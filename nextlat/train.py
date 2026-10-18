"""Single-node DDP trainer for joint Qwen3 + residual NextLat training."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import shlex
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

import torch
import torch.distributed as dist
from datasets import load_from_disk
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import __version__ as transformers_version

from .core import (
    NextLatDynamics,
    NextLatLossConfig,
    NextLatQwen,
    nextlat_position_counts,
)
from .data import split_receipt
from .dmt import iter_deep_detached_repair_losses
from .eval_recursive import teacher_forced_recursive_metrics
from .run_utils import (
    assert_claim_source_committed,
    capture_rng_state,
    git_commit,
    initialize_run_directory,
    load_dynamics_checkpoint,
    restore_rng_state,
    save_checkpoint,
    write_jsonl,
    write_status,
)
from .tracking import WandbTracker, start_wandb_tracker
from .countdown_policy_gradient import (
    PROTOCOL as PG_PROTOCOL, POLICY as PG_POLICY, SCALED_POLICY as PG_SCALED_POLICY,
    conditional_group_backward, phase_token_budget,
)


RESUME_FRESH = "fresh"
RESUME_SAME_SOURCE = "same_source"
RESUME_JOINT_D4_FORK = "joint_d4_from_joint_d16"
RESUME_JOINT_HYBRID_FORK = "joint_hybrid_d4_h16_from_joint_d16"
RESUME_JOINT_D4_SYSTEMS_FORK = "joint_d4_systems_fork"
RESUME_JOINT_D4_NOISE_FORK = "joint_d4_anchor_noise_pilot"
RESUME_JOINT_D4_DATA_FORK = "joint_d4_fresh_data_pilot"
RESUME_COUNTDOWN_PG_FORK = "countdown_recurrent_pg_phase"
SEALED_JOINT_FORKS = (
    RESUME_JOINT_D4_FORK,
    RESUME_JOINT_HYBRID_FORK,
    RESUME_JOINT_D4_SYSTEMS_FORK,
    RESUME_JOINT_D4_NOISE_FORK,
    RESUME_JOINT_D4_DATA_FORK,
    RESUME_COUNTDOWN_PG_FORK,
)


def _validate_transformers_runtime(
    config: Dict[str, Any], *, installed_version: Optional[str] = None
) -> None:
    runtime = config.get("runtime")
    if not isinstance(runtime, dict):
        raise RuntimeError("training config requires an explicit runtime contract")
    required = runtime.get("required_transformers_version")
    if not isinstance(required, str) or not required:
        raise RuntimeError("runtime contract requires a Transformers version")
    observed = transformers_version if installed_version is None else installed_version
    if observed != required:
        raise RuntimeError(f"Transformers {required} required; got {observed}")


def _validate_loaded_rope_theta(config: Dict[str, Any], model_config: Any) -> None:
    runtime = config.get("runtime")
    if not isinstance(runtime, dict) or "expected_rope_theta" not in runtime:
        raise RuntimeError("runtime contract requires expected_rope_theta")
    expected = float(runtime["expected_rope_theta"])
    observed: list[tuple[str, float]] = []
    rope_parameters = getattr(model_config, "rope_parameters", None)
    if isinstance(rope_parameters, dict) and "rope_theta" in rope_parameters:
        observed.append(
            ("rope_parameters.rope_theta", float(rope_parameters["rope_theta"]))
        )
    legacy_rope_theta = getattr(model_config, "rope_theta", None)
    if legacy_rope_theta is not None:
        observed.append(("rope_theta", float(legacy_rope_theta)))
    if not observed:
        raise RuntimeError("loaded model config has no RoPE theta")
    mismatched = [(name, value) for name, value in observed if value != expected]
    if mismatched:
        details = ", ".join(f"{name}={value:g}" for name, value in observed)
        raise RuntimeError(f"RoPE theta misloaded: {details}; expected {expected:g}")


def _validate_embedding_tying(model: torch.nn.Module) -> None:
    expected = bool(model.config.tie_word_embeddings)
    observed = model.get_input_embeddings().weight is model.lm_head.weight
    if observed != expected:
        raise RuntimeError(f"embedding tying mismatch: config={expected}, loaded={observed}")


def _prepared_data_authority(config: Dict[str, Any]) -> Dict[str, Any]:
    if config.get("continuation") is not None:
        authority = _continuation_contract(config)["prepared_data"]
    else:
        authority = config.get("prepared_data")
    if not isinstance(authority, dict):
        raise RuntimeError("prepared data requires an explicit prepared_data authority")
    return authority


def _setup_distributed() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("the joint FFT path requires CUDA")
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size, torch.device("cuda", local_rank)


def _barrier(world_size: int) -> None:
    if world_size > 1:
        dist.barrier()


def _all_reduce_sum(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    result = tensor.clone()
    if world_size > 1:
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
    return result


def _resolve_training_sdpa_backend(
    model_config: Dict[str, Any],
) -> Optional[SDPBackend]:
    """Resolve an optional forced SDPA backend without silent fallback."""

    requested = model_config.get("sdpa_backend")
    if requested is None:
        return None
    if requested != "cudnn":
        raise RuntimeError(
            "model.sdpa_backend must be omitted or equal to 'cudnn'"
        )
    if model_config.get("attention") != "sdpa":
        raise RuntimeError("model.sdpa_backend requires model.attention='sdpa'")
    if not torch.backends.cudnn.is_available():
        raise RuntimeError("cuDNN SDPA was requested but cuDNN is unavailable")
    if not torch.backends.cuda.cudnn_sdp_enabled():
        raise RuntimeError("cuDNN SDPA was requested but disabled at runtime")
    return SDPBackend.CUDNN_ATTENTION


def _training_sdpa_context(backend: Optional[SDPBackend]):
    if backend is None:
        return nullcontext()
    # A singleton backend list makes unsupported shapes fail instead of
    # silently falling back to FlashAttention, memory-efficient, or math SDPA.
    return sdpa_kernel(backend)


class NextLatCollator:
    def __init__(self, pad_token_id: int, max_length: int) -> None:
        self.pad_token_id = pad_token_id
        self.max_length = max_length

    def __call__(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        longest = max(len(record["input_ids"]) for record in records)
        length = min(self.max_length, 128 * math.ceil(longest / 128))
        batch_size = len(records)
        input_ids = torch.full(
            (batch_size, length), self.pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros((batch_size, length), dtype=torch.bool)
        assistant_loss_mask = torch.zeros((batch_size, length), dtype=torch.bool)
        for row, record in enumerate(records):
            ids = record["input_ids"][:length]
            loss_mask = record["assistant_loss_mask"][:length]
            width = len(ids)
            input_ids[row, :width] = torch.tensor(ids, dtype=torch.long)
            attention_mask[row, :width] = True
            assistant_loss_mask[row, :width] = torch.tensor(
                loss_mask, dtype=torch.bool
            )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "assistant_loss_mask": assistant_loss_mask,
            "problem_hash": [record["problem_hash"] for record in records],
        }


def _to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        name: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for name, value in batch.items()
    }


def _resolve_execution_micro_batch_size(training_config: Dict[str, Any]) -> int:
    """Resolve optional physical batching without changing the logical batch clock."""

    execution_size = training_config.get("execution_micro_batch_size", 1)
    if isinstance(execution_size, bool) or not isinstance(execution_size, int):
        raise RuntimeError("execution_micro_batch_size must be an integer")
    if execution_size < 1:
        raise RuntimeError("execution_micro_batch_size must be positive")

    logical_size = int(training_config["micro_batch_size"])
    accumulation_steps = int(training_config["gradient_accumulation_steps"])
    if logical_size != 1:
        raise RuntimeError(
            "execution microbatch coalescing requires logical micro_batch_size=1"
        )
    if accumulation_steps % execution_size != 0:
        raise RuntimeError(
            "gradient_accumulation_steps must be divisible by "
            "execution_micro_batch_size"
        )
    return execution_size


def _resolve_execution_pad_to_multiple(
    training_config: Dict[str, Any], *, systems_fork: bool
) -> Optional[int]:
    requested = training_config.get("execution_pad_to_multiple")
    if requested is None:
        return None
    if not systems_fork:
        raise RuntimeError(
            "execution_pad_to_multiple is admitted only by the joint d=4 systems fork"
        )
    if isinstance(requested, bool) or not isinstance(requested, int):
        raise RuntimeError("execution_pad_to_multiple must be an integer")
    if requested not in (64, 128, 256):
        raise RuntimeError("execution_pad_to_multiple must be one of 64, 128, or 256")
    if int(training_config.get("execution_micro_batch_size", 1)) != 2:
        raise RuntimeError("execution_pad_to_multiple requires execution B2")
    return requested


def _logical_batch_token_length(batch: Dict[str, Any]) -> int:
    attention_mask = batch.get("attention_mask")
    if not isinstance(attention_mask, torch.Tensor):
        raise RuntimeError("logical microbatch requires an attention_mask tensor")
    if attention_mask.ndim != 2 or attention_mask.shape[0] != 1:
        raise RuntimeError(
            "execution microbatch coalescing requires logical batch dimension 1"
        )
    true_length = int(attention_mask.sum().item())
    if true_length <= 0:
        raise RuntimeError("logical microbatch has no non-padding tokens")
    if not bool(attention_mask[0, :true_length].all()) or bool(
        attention_mask[0, true_length:].any()
    ):
        raise RuntimeError("logical microbatch attention_mask must be a true prefix")
    return true_length


def _merge_logical_micro_batches(
    batches: List[Dict[str, Any]],
    *,
    pad_token_id: int,
    pad_to_multiple: Optional[int] = None,
) -> Dict[str, Any]:
    """Right-pad and concatenate already-collated logical B1 microbatches."""

    if not batches:
        raise RuntimeError("cannot merge an empty execution microbatch")
    required_fields = {
        "input_ids",
        "attention_mask",
        "assistant_loss_mask",
        "problem_hash",
    }
    for batch in batches:
        if set(batch) != required_fields:
            raise RuntimeError(
                "logical microbatch fields do not match the NextLat collator contract"
            )
        for name in ("input_ids", "attention_mask", "assistant_loss_mask"):
            value = batch[name]
            if (
                not isinstance(value, torch.Tensor)
                or value.ndim != 2
                or value.shape[0] != 1
            ):
                raise RuntimeError(
                    f"logical microbatch {name} must have shape [1, sequence]"
                )
        if not isinstance(batch["problem_hash"], list) or len(
            batch["problem_hash"]
        ) != 1:
            raise RuntimeError(
                "logical microbatch problem_hash must contain exactly one sample"
            )
        if batch["input_ids"].shape != batch["attention_mask"].shape or batch[
            "input_ids"
        ].shape != batch["assistant_loss_mask"].shape:
            raise RuntimeError("logical microbatch tensor shapes do not match")
        if bool(
            (
                batch["assistant_loss_mask"]
                & ~batch["attention_mask"].to(dtype=torch.bool)
            ).any()
        ):
            raise RuntimeError(
                "assistant_loss_mask must be contained in attention_mask"
            )

    reference_ids = batches[0]["input_ids"]
    reference_attention = batches[0]["attention_mask"]
    reference_assistant = batches[0]["assistant_loss_mask"]
    for batch in batches[1:]:
        for name, reference in (
            ("input_ids", reference_ids),
            ("attention_mask", reference_attention),
            ("assistant_loss_mask", reference_assistant),
        ):
            value = batch[name]
            if value.dtype != reference.dtype or value.device != reference.device:
                raise RuntimeError(
                    f"logical microbatch {name} dtype/device does not match"
                )

    batch_size = len(batches)
    if pad_to_multiple is None:
        width = max(int(batch["input_ids"].shape[1]) for batch in batches)
    else:
        if len(batches) != 2:
            raise RuntimeError("physical execution padding requires B2")
        if pad_to_multiple not in (64, 128, 256):
            raise RuntimeError("merged execution padding multiple is unsupported")
        true_width = max(_logical_batch_token_length(batch) for batch in batches)
        width = pad_to_multiple * math.ceil(true_width / pad_to_multiple)
    allocate = {
        "device": reference_ids.device,
        "pin_memory": reference_ids.device.type == "cpu"
        and all(batch["input_ids"].is_pinned() for batch in batches),
    }
    input_ids = torch.full(
        (batch_size, width),
        pad_token_id,
        dtype=reference_ids.dtype,
        **allocate,
    )
    attention_mask = torch.zeros(
        (batch_size, width), dtype=reference_attention.dtype, **allocate
    )
    assistant_loss_mask = torch.zeros(
        (batch_size, width), dtype=reference_assistant.dtype, **allocate
    )
    problem_hash: list[Any] = []
    for row, batch in enumerate(batches):
        row_width = min(int(batch["input_ids"].shape[1]), width)
        input_ids[row, :row_width].copy_(batch["input_ids"][0, :row_width])
        attention_mask[row, :row_width].copy_(
            batch["attention_mask"][0, :row_width]
        )
        assistant_loss_mask[row, :row_width].copy_(
            batch["assistant_loss_mask"][0, :row_width]
        )
        problem_hash.extend(batch["problem_hash"])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "assistant_loss_mask": assistant_loss_mask,
        "problem_hash": problem_hash,
    }


def _coalesce_execution_micro_batches(
    logical_batches: List[Dict[str, Any]],
    *,
    execution_micro_batch_size: int,
    pad_token_id: int,
    pad_to_multiple: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Pack one fixed accumulation group; never sort across optimizer updates."""

    if execution_micro_batch_size < 1:
        raise RuntimeError("execution_micro_batch_size must be positive")
    if len(logical_batches) % execution_micro_batch_size != 0:
        raise RuntimeError(
            "logical accumulation group is not divisible by execution microbatch size"
        )
    if pad_to_multiple is not None and execution_micro_batch_size != 2:
        raise RuntimeError("physical execution padding requires B2")
    if execution_micro_batch_size == 1:
        return logical_batches

    ordered = sorted(
        enumerate(logical_batches),
        key=lambda indexed: (_logical_batch_token_length(indexed[1]), indexed[0]),
    )
    sorted_batches = [batch for _, batch in ordered]
    return [
        _merge_logical_micro_batches(
            sorted_batches[start : start + execution_micro_batch_size],
            pad_token_id=pad_token_id,
            pad_to_multiple=pad_to_multiple,
        )
        for start in range(0, len(sorted_batches), execution_micro_batch_size)
    ]


def _optimizer_groups(
    module: torch.nn.Module,
    learning_rate: float,
    weight_decay: float,
    group_name: str,
) -> List[Dict[str, Any]]:
    decay: List[torch.nn.Parameter] = []
    no_decay: List[torch.nn.Parameter] = []
    for parameter in module.parameters():
        if not parameter.requires_grad:
            continue
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    return [
        {
            "params": decay,
            "lr": learning_rate,
            "initial_lr": learning_rate,
            "weight_decay": weight_decay,
            "name": f"{group_name}_decay",
        },
        {
            "params": no_decay,
            "lr": learning_rate,
            "initial_lr": learning_rate,
            "weight_decay": 0.0,
            "name": f"{group_name}_no_decay",
        },
    ]


def _set_warmup_lr(optimizer: torch.optim.Optimizer, tokens: int, warmup: int) -> None:
    scale = 1.0 if warmup <= 0 else min(1.0, max(1, tokens) / warmup)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * scale


def _continuation_contract(config: Dict[str, Any]) -> Dict[str, Any]:
    contract = config.get("continuation")
    if not isinstance(contract, dict):
        raise RuntimeError("continued joint d=4 training requires a continuation contract")
    if contract.get("protocol") not in (
        "optimizer_state_preserving_joint_d16_to_joint_d4_v1",
        "optimizer_state_preserving_joint_d16_to_hybrid_d4_h16_v1",
        "optimizer_state_preserving_joint_d4_systems_fork_v1",
        "optimizer_state_preserving_joint_d4_anchor_noise_pilot_v1",
        "optimizer_state_preserving_joint_d4_fresh_data_pilot_v1",
        PG_PROTOCOL,
    ):
        raise RuntimeError("unsupported joint d=4 continuation protocol")
    required = {"parent", "prepared_data"}
    missing = required - set(contract)
    if missing:
        raise RuntimeError(f"continuation contract is missing: {sorted(missing)}")
    return contract


def _hybrid_correction_contract(
    config: Dict[str, Any], *, world_size: int
) -> Optional[tuple[int, float]]:
    """Validate the one supported same-forward H16 correction intervention."""

    raw = config.get("hybrid_correction")
    if raw is None:
        return None
    if not isinstance(raw, dict) or raw.get("enabled") is not True:
        raise RuntimeError("hybrid_correction must be an enabled object")
    if world_size != 1:
        raise RuntimeError(
            "same-forward hybrid correction is currently admitted only at world_size=1"
        )
    required_values = {
        "mode": "dense_teacher_forced",
        "repair_horizon": 16,
        "loss": "smooth_l1",
        "reuse_connected_depth": 4,
        "recurrent_input_detached_each_step": True,
        "detached_one_step_loss": True,
        "exact_states_from_joint_forward": True,
        "theta_gradient": False,
    }
    for key, expected in required_values.items():
        if raw.get(key) != expected:
            raise RuntimeError(
                f"hybrid_correction {key} must be {expected!r}; got {raw.get(key)!r}"
            )
    loss_weight = float(raw.get("loss_weight", 0.0))
    if not math.isfinite(loss_weight) or loss_weight <= 0:
        raise RuntimeError("hybrid_correction loss_weight must be finite and positive")
    return int(raw["repair_horizon"]), loss_weight


def _require_resumable_checkpoint(checkpoint_dir: Path) -> None:
    required = (
        checkpoint_dir / "COMPLETE",
        checkpoint_dir / "sha256.json",
        checkpoint_dir / "train_state.json",
        checkpoint_dir / "dynamics.safetensors",
        checkpoint_dir / "optimizer.pt",
        checkpoint_dir / "rng_state.pt",
        checkpoint_dir / "model",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"resume checkpoint is incomplete: {missing}")


def _require_mapping_equal_except(
    *,
    parent: Dict[str, Any],
    child: Dict[str, Any],
    allowed_changes: set[str],
    label: str,
) -> None:
    parent_scientific = {
        key: value for key, value in parent.items() if key not in allowed_changes
    }
    child_scientific = {
        key: value for key, value in child.items() if key not in allowed_changes
    }
    if parent_scientific != child_scientific:
        changed = sorted(
            key
            for key in set(parent_scientific) | set(child_scientific)
            if parent_scientific.get(key) != child_scientific.get(key)
        )
        raise RuntimeError(
            f"joint d=4 systems fork changed non-implementation {label}: {changed}"
        )


def _validate_joint_d4_systems_fork(
    *,
    config: Dict[str, Any],
    parent_config: Dict[str, Any],
    parent_config_path: Path,
    resume_path: Path,
    resume_state: Dict[str, Any],
    contract: Dict[str, Any],
) -> str:
    """Admit an exact joint-d4 checkpoint onto implementation-only backends."""

    parent = contract["parent"]
    if not isinstance(parent, dict):
        raise RuntimeError("continuation parent must be an object")
    required_parent_fields = {
        "source_code_commit",
        "config_sha256",
        "checkpoint_receipt_sha256",
        "train_state_sha256",
        "optimizer_sha256",
        "rng_sha256",
        "tokens_seen",
        "update",
        "data_batches_seen",
        "sampler_epoch",
        "sampler_offset",
        "position_totals",
        "prepared_manifest_sha256",
        "optimizer_group_parameter_counts",
        "theta_adam_steps",
        "psi_adam_steps",
    }
    missing_parent_fields = required_parent_fields - set(parent)
    if missing_parent_fields:
        raise RuntimeError(
            "joint d=4 systems parent is missing: "
            f"{sorted(missing_parent_fields)}"
        )

    receipt_path = resume_path / "sha256.json"
    state_path = resume_path / "train_state.json"
    rng_path = resume_path / "rng_state.pt"
    optimizer_path = resume_path / "optimizer.pt"
    exact_files = (
        (parent_config_path, "config_sha256", "config"),
        (receipt_path, "checkpoint_receipt_sha256", "checkpoint receipt"),
        (state_path, "train_state_sha256", "train state"),
        (rng_path, "rng_sha256", "RNG state"),
        (optimizer_path, "optimizer_sha256", "optimizer state"),
    )
    for path, contract_key, label in exact_files:
        if not path.is_file():
            raise RuntimeError(f"joint d=4 systems parent is missing {label}: {path}")
        if _sha256_path(path) != parent[contract_key]:
            raise RuntimeError(f"joint d=4 systems parent {label} hash mismatch")

    receipt = json.loads(receipt_path.read_text())
    if not isinstance(receipt, dict):
        raise RuntimeError("joint d=4 systems checkpoint receipt must be an object")
    receipt_contract = {
        "train_state.json": "train_state_sha256",
        "rng_state.pt": "rng_sha256",
        "optimizer.pt": "optimizer_sha256",
    }
    for relative_path, parent_key in receipt_contract.items():
        if receipt.get(relative_path) != parent[parent_key]:
            raise RuntimeError(
                "joint d=4 systems checkpoint receipt mismatch for "
                f"{relative_path}"
            )

    counter_fields = (
        "source_code_commit",
        "tokens_seen",
        "update",
        "data_batches_seen",
        "sampler_epoch",
        "sampler_offset",
        "position_totals",
        "prepared_manifest_sha256",
    )
    for key in counter_fields:
        if resume_state.get(key) != parent[key]:
            raise RuntimeError(f"joint d=4 systems parent state mismatch for {key}")
    if resume_state.get("optimizer_included") is not True:
        raise RuntimeError("joint d=4 systems parent lacks optimizer state")
    if resume_state.get("rng_state_included") is not True:
        raise RuntimeError("joint d=4 systems parent lacks RNG state")

    ignored_top_level = {
        "run",
        "continuation",
        "model",
        "nextlat",
        "training",
        "tracking",
        "scale_ladder",
    }
    _require_mapping_equal_except(
        parent=parent_config,
        child=config,
        allowed_changes=ignored_top_level,
        label="top-level sections",
    )

    parent_model = parent_config.get("model")
    child_model = config.get("model")
    if not isinstance(parent_model, dict) or not isinstance(child_model, dict):
        raise RuntimeError("joint d=4 systems fork requires model objects")
    _require_mapping_equal_except(
        parent=parent_model,
        child=child_model,
        allowed_changes={"gradient_checkpointing", "sdpa_backend"},
        label="model fields",
    )
    for label, model in (("parent", parent_model), ("child", child_model)):
        gradient_checkpointing = model.get("gradient_checkpointing", True)
        if not isinstance(gradient_checkpointing, bool):
            raise RuntimeError(
                f"joint d=4 systems {label} gradient_checkpointing must be boolean"
            )

    parent_nextlat = parent_config.get("nextlat")
    child_nextlat = config.get("nextlat")
    if not isinstance(parent_nextlat, dict) or not isinstance(child_nextlat, dict):
        raise RuntimeError("joint d=4 systems fork requires NextLat objects")
    _require_mapping_equal_except(
        parent=parent_nextlat,
        child=child_nextlat,
        allowed_changes={"kl_implementation", "vocab_chunk_tokens"},
        label="NextLat fields",
    )
    if parent_nextlat.get("horizon") != 4 or child_nextlat.get("horizon") != 4:
        raise RuntimeError("joint d=4 systems fork requires identical d=4 objectives")
    for label, nextlat in (("parent", parent_nextlat), ("child", child_nextlat)):
        kl_implementation = nextlat.get("kl_implementation", "autograd")
        if kl_implementation not in (
            "autograd",
            "forward_gradient",
            "shared_teacher_forward_gradient",
            "shared_teacher_horizon_stacked",
            "shared_teacher_horizon_stacked_triton",
        ):
            raise RuntimeError(
                f"joint d=4 systems {label} KL implementation is unsupported"
            )
        vocab_chunk_tokens = nextlat.get("vocab_chunk_tokens")
        if (
            isinstance(vocab_chunk_tokens, bool)
            or not isinstance(vocab_chunk_tokens, int)
            or vocab_chunk_tokens <= 0
        ):
            raise RuntimeError(
                f"joint d=4 systems {label} vocab_chunk_tokens must be positive"
            )

    parent_training = parent_config.get("training")
    child_training = config.get("training")
    if not isinstance(parent_training, dict) or not isinstance(child_training, dict):
        raise RuntimeError("joint d=4 systems fork requires training objects")
    schedule_fields = {
        "decision_tokens",
        "registered_token_rungs",
        "log_every_updates",
        "recursive_eval_every_tokens",
        "recursive_eval_examples",
        "checkpoint_tokens",
        "resumable_checkpoint_tokens",
        "wandb_artifact_tokens",
        "wandb_receipt_artifact_tokens",
    }
    _require_mapping_equal_except(
        parent=parent_training,
        child=child_training,
        allowed_changes=schedule_fields
        | {"execution_micro_batch_size", "execution_pad_to_multiple"},
        label="training fields",
    )
    optimizer_invariants = (
        "seed",
        "expected_world_size",
        "micro_batch_size",
        "gradient_accumulation_steps",
        "base_learning_rate",
        "dynamics_learning_rate",
        "warmup_tokens",
        "weight_decay",
        "adam_beta1",
        "adam_beta2",
        "adam_eps",
        "gradient_clip",
        "freeze_theta",
    )
    for key in optimizer_invariants:
        if parent_training.get(key) != child_training.get(key):
            raise RuntimeError(
                f"joint d=4 systems fork changed optimizer invariant {key}"
            )
    if (
        parent_training.get("micro_batch_size") != 1
        or child_training.get("micro_batch_size") != 1
        or parent_training.get("gradient_accumulation_steps") != 8
        or child_training.get("gradient_accumulation_steps") != 8
    ):
        raise RuntimeError(
            "joint d=4 systems fork requires logical B1 and accumulation 8"
        )
    if (
        parent_training.get("freeze_theta", False) is not False
        or child_training.get("freeze_theta", False) is not False
    ):
        raise RuntimeError("joint d=4 systems fork requires trainable theta")
    if (
        parent_training.get("teacher_forced_corpus_tokens") is not True
        or child_training.get("teacher_forced_corpus_tokens") is not True
    ):
        raise RuntimeError(
            "joint d=4 systems fork requires teacher-forced corpus tokens"
        )
    _resolve_execution_micro_batch_size(parent_training)
    _resolve_execution_micro_batch_size(child_training)
    parent_continuation = parent_config.get("continuation")
    parent_is_systems_fork = (
        isinstance(parent_continuation, dict)
        and parent_continuation.get("protocol")
        == "optimizer_state_preserving_joint_d4_systems_fork_v1"
    )
    _resolve_execution_pad_to_multiple(
        parent_training, systems_fork=parent_is_systems_fork
    )
    _resolve_execution_pad_to_multiple(child_training, systems_fork=True)

    return RESUME_JOINT_D4_SYSTEMS_FORK


def _validate_parent_continuation(
    *,
    config: Dict[str, Any],
    parent_config: Dict[str, Any],
    parent_config_path: Path,
    resume_path: Path,
    resume_state: Dict[str, Any],
    current_source_commit: str,
) -> str:
    """Admit exact same-source resumes and explicitly sealed continuation forks."""

    same_source = resume_state.get("source_code_commit") == current_source_commit
    if same_source:
        mismatch: Optional[str] = None
        for section in (
            "model",
            "runtime",
            "dataset",
            "data",
            "upstream_nextlat",
            "nextlat",
            "training",
            "scale_ladder",
            "tracking",
            "recursive_gate",
            "continuation",
            "hybrid_correction",
            "recurrent_policy_gradient",
        ):
            if parent_config.get(section) != config.get(section):
                mismatch = section
                break
        if mismatch is None:
            return RESUME_SAME_SOURCE
        requested_continuation = config.get("continuation")
        requested_protocol = (
            requested_continuation.get("protocol")
            if isinstance(requested_continuation, dict)
            else None
        )
        if requested_protocol not in ("optimizer_state_preserving_joint_d4_systems_fork_v1", PG_PROTOCOL):
            raise RuntimeError(f"resume config mismatch in section {mismatch}")

    contract = _continuation_contract(config)
    protocol = contract["protocol"]
    if protocol == PG_PROTOCOL:
        if parent_config.get("recurrent_policy_gradient") is not None:
            raise RuntimeError("new PG phase must start from the registered SFT endpoint")
        if int(resume_state["tokens_seen"]) < int(parent_config["training"]["decision_tokens"]):
            raise RuntimeError("PG phase requires the completed SFT token budget, not an intermediate checkpoint")
        if config["data"].get("corpus_policy") != "countdown_verified_expression_v1":
            raise RuntimeError("PG phase requires verified Countdown data")
        if (contract["prepared_data"] != parent_config["continuation"]["prepared_data"]
            or contract["fresh_sampler_origin"] != resume_state.get("fresh_sampler_origin")
            or contract["prepared_data"]["manifest_sha256"] != resume_state["prepared_manifest_sha256"]):
            raise RuntimeError("PG phase must preserve corpus and existing sampler origin")
        compatible = copy.deepcopy(config)
        compatible.pop("recurrent_policy_gradient")
        for section in ("claim", "evidence_gate"):
            compatible[section] = parent_config[section]
        _validate_joint_d4_systems_fork(
            config=compatible, parent_config=parent_config, parent_config_path=parent_config_path,
            resume_path=resume_path, resume_state=resume_state, contract=contract)
        return RESUME_COUNTDOWN_PG_FORK
    if protocol == "optimizer_state_preserving_joint_d4_fresh_data_pilot_v1":
        if resume_state.get("tokens_seen") != contract["parent"]["tokens_seen"]:
            raise RuntimeError("fresh-data fork requires its exact registered parent")
        if config["nextlat"].get("anchor_noise_relative") != 0:
            raise RuntimeError("fresh-data fork must not also change anchor noise")
        countdown = config["data"].get("corpus_policy") == "countdown_verified_expression_v1"
        if countdown and contract.get("task_fork") != {
            "task": "countdown_verified_expression_v1",
            "parent_sequence_length": parent_config["data"]["sequence_length"],
            "sequence_length": 512,
        }:
            raise RuntimeError("Countdown requires explicit sequence-length task-fork admission")
        if countdown and config["data"]["sequence_length"] != 512:
            raise RuntimeError("Countdown task fork requires sequence_length 512")
        _require_mapping_equal_except(
            parent=parent_config["data"], child=config["data"],
            allowed_changes={"prepared_name", "prepared_train_tokens", "corpus_policy"}
            | ({"sequence_length", "recursive_holdout_problems", "stress_candidate_problems"} if countdown else set()), label="fresh data fields",
        )
        corpus_policy = config["data"].get("corpus_policy", "verified_openr1_v1")
        if corpus_policy not in ("verified_openr1_v1", "smoltalk_openr1_rehearsal_v1", "countdown_verified_expression_v1"):
            raise RuntimeError("unsupported fresh-data corpus policy")
        if config["data"]["prepared_train_tokens"] < (
            config["training"]["decision_tokens"] - contract["parent"]["tokens_seen"]
            + config["data"]["sequence_length"] * config["training"]["gradient_accumulation_steps"]
        ):
            raise RuntimeError("fresh-data pool must exceed the full budget plus one update")
        if contract.get("parent_admission", "original_data_ce") not in (
            "original_data_ce", "content_verified_runtime_finite_initial_ce"
        ):
            raise RuntimeError("fresh-data fork requires an explicit supported admission mode")
        if contract.get("fresh_sampler_origin") != {
            key: resume_state[key] for key in ("tokens_seen", "update", "data_batches_seen")
        }:
            raise RuntimeError("fresh-data sampler origin differs from the imported parent")
        if contract["prepared_data"]["manifest_sha256"] == resume_state["prepared_manifest_sha256"]:
            raise RuntimeError("fresh-data fork cannot reuse the old prepared corpus")
        compatible = copy.deepcopy(config)
        compatible["data"] = parent_config["data"]
        if corpus_policy == "smoltalk_openr1_rehearsal_v1" or countdown:
            # The explicitly registered data intervention includes mixed-source
            # provenance. Model, objective, Adam and runtime invariants below
            # remain unchanged; the prepared manifest is checked independently.
            compatible["dataset"] = parent_config["dataset"]
        if "anchor_noise_relative" not in parent_config["nextlat"]:
            compatible["nextlat"].pop("anchor_noise_relative")
        for section in ("claim", "evidence_gate"):
            compatible[section] = parent_config[section]
        _validate_joint_d4_systems_fork(
            config=compatible, parent_config=parent_config,
            parent_config_path=parent_config_path, resume_path=resume_path,
            resume_state=resume_state, contract=contract,
        )
        return RESUME_JOINT_D4_DATA_FORK
    if protocol == "optimizer_state_preserving_joint_d4_anchor_noise_pilot_v1":
        # Reuse the established parent/data/Adam admission. Only the initial
        # anchor perturbation is a new scientific axis; do not weaken the
        # original implementation-only systems-fork contract.
        amplitude = config["nextlat"].get("anchor_noise_relative")
        if amplitude not in (0.0, 0.1):
            raise RuntimeError("pilot permits only matched clean or relative-0.1 noise")
        if parent_config["nextlat"].get("anchor_noise_relative", 0.0) != 0.0:
            raise RuntimeError("noise pilot must start from the clean 800M parent")
        if resume_state.get("tokens_seen") != 800007805:
            raise RuntimeError("noise pilot requires the exact 800007805-token parent")
        compatible = copy.deepcopy(config)
        compatible["nextlat"].pop("anchor_noise_relative")
        for section in ("claim", "evidence_gate"):
            compatible[section] = parent_config[section]
        _validate_joint_d4_systems_fork(
            config=compatible, parent_config=parent_config,
            parent_config_path=parent_config_path, resume_path=resume_path,
            resume_state=resume_state, contract=contract,
        )
        return RESUME_JOINT_D4_NOISE_FORK
    if protocol == "optimizer_state_preserving_joint_d4_systems_fork_v1":
        return _validate_joint_d4_systems_fork(
            config=config,
            parent_config=parent_config,
            parent_config_path=parent_config_path,
            resume_path=resume_path,
            resume_state=resume_state,
            contract=contract,
        )
    hybrid_fork = (
        protocol == "optimizer_state_preserving_joint_d16_to_hybrid_d4_h16_v1"
    )
    if hybrid_fork != (config.get("hybrid_correction") is not None):
        raise RuntimeError(
            "continuation protocol and hybrid_correction presence do not match"
        )
    parent = contract["parent"]
    if not isinstance(parent, dict):
        raise RuntimeError("continuation parent must be an object")
    if _sha256_path(parent_config_path) != parent.get("config_sha256"):
        raise RuntimeError("sealed 400M parent config hash mismatch")
    receipt_path = resume_path / "sha256.json"
    state_path = resume_path / "train_state.json"
    rng_path = resume_path / "rng_state.pt"
    if _sha256_path(receipt_path) != parent.get("checkpoint_receipt_sha256"):
        raise RuntimeError("sealed 400M checkpoint receipt hash mismatch")
    if _sha256_path(state_path) != parent.get("train_state_sha256"):
        raise RuntimeError("sealed 400M train-state hash mismatch")
    if _sha256_path(rng_path) != parent.get("rng_sha256"):
        raise RuntimeError("sealed 400M RNG-state hash mismatch")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("optimizer.pt") != parent.get("optimizer_sha256"):
        raise RuntimeError("sealed 400M optimizer receipt mismatch")
    if receipt.get("rng_state.pt") != parent.get("rng_sha256"):
        raise RuntimeError("sealed 400M RNG receipt mismatch")

    for section in ("model", "dataset", "data", "upstream_nextlat"):
        if parent_config.get(section) != config.get(section):
            raise RuntimeError(f"joint d=4 fork changed immutable section {section}")
    parent_nextlat = dict(parent_config["nextlat"])
    nextlat = dict(config["nextlat"])
    if parent_nextlat.pop("horizon") != 16 or nextlat.pop("horizon") != 4:
        raise RuntimeError("sealed fork must change the objective from d=16 to d=4")
    if parent_nextlat != nextlat:
        raise RuntimeError("joint d=4 fork changed non-horizon NextLat settings")
    optimizer_keys = (
        "seed",
        "expected_world_size",
        "micro_batch_size",
        "gradient_accumulation_steps",
        "base_learning_rate",
        "dynamics_learning_rate",
        "warmup_tokens",
        "weight_decay",
        "adam_beta1",
        "adam_beta2",
        "adam_eps",
        "gradient_clip",
    )
    for key in optimizer_keys:
        if parent_config["training"].get(key) != config["training"].get(key):
            raise RuntimeError(f"joint d=4 fork changed optimizer invariant {key}")
    if parent_config["training"].get("freeze_theta", False) is not False:
        raise RuntimeError("sealed parent is not the joint theta+psi endpoint")
    if config["training"].get("freeze_theta", False) is not False:
        raise RuntimeError("joint d=4 continuation must unfreeze theta")

    for key in (
        "source_code_commit",
        "tokens_seen",
        "update",
        "data_batches_seen",
        "sampler_epoch",
        "sampler_offset",
        "prepared_manifest_sha256",
    ):
        if resume_state.get(key) != parent.get(key):
            raise RuntimeError(f"sealed 400M parent state mismatch for {key}")
    if resume_state.get("optimizer_included") is not True:
        raise RuntimeError("sealed 400M parent lacks optimizer state")
    if resume_state.get("rng_state_included") is not True:
        raise RuntimeError("sealed 400M parent lacks RNG state")
    if resume_state.get("freeze_theta", False) is not False:
        raise RuntimeError("sealed 400M parent is not joint theta+psi")
    return RESUME_JOINT_HYBRID_FORK if hybrid_fork else RESUME_JOINT_D4_FORK


def _tracker_resume_mode(resume_path: Optional[Path], resume_kind: str) -> str:
    return "must" if resume_path is not None and resume_kind == RESUME_SAME_SOURCE else "never"


def _optimizer_state_policy(
    config: Dict[str, Any], *, hybrid_correction: bool
) -> str:
    continuation = config.get("continuation")
    if isinstance(continuation, dict) and continuation.get("protocol") == "optimizer_state_preserving_joint_d4_fresh_data_pilot_v1":
        return "joint_adam_state_preserved_d4_fresh_corpus_v1"
    if hybrid_correction:
        return "joint_adam_state_preserved_d16_to_d4_plus_deep_repair_v1"
    if isinstance(continuation, dict) and continuation.get("protocol") == PG_PROTOCOL:
        return "joint_adam_state_preserved_countdown_policy_phase_v1"
    if not isinstance(continuation, dict):
        return "joint_adam_v1"
    if continuation.get("protocol") == "optimizer_state_preserving_joint_d4_anchor_noise_pilot_v1":
        return "joint_adam_state_preserved_anchor_noise_pilot_v1"
    if (
        continuation.get("protocol")
        == "optimizer_state_preserving_joint_d4_systems_fork_v1"
    ):
        return "joint_adam_state_preserved_d4_systems_fork_v1"
    return "joint_adam_state_preserved_d16_to_d4_v1"


def _run_completed(
    *, tokens_seen: int, token_budget: int, qualification_updates: int
) -> bool:
    return qualification_updates == 0 and tokens_seen >= token_budget


def _validate_loaded_optimizer_state(
    *,
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
    dynamics: torch.nn.Module,
    expected_group_parameter_counts: Optional[List[int]] = None,
    expected_theta_steps: Optional[List[int]] = None,
    expected_psi_steps: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Prove the loaded four-group Adam state covers both theta and psi."""

    names = [str(group.get("name")) for group in optimizer.param_groups]
    if names != ["theta_decay", "theta_no_decay", "psi_decay", "psi_no_decay"]:
        raise RuntimeError(f"optimizer group topology mismatch: {names}")
    counts = [len(group["params"]) for group in optimizer.param_groups]
    if expected_group_parameter_counts is not None and counts != expected_group_parameter_counts:
        raise RuntimeError(
            f"optimizer group parameter counts mismatch: {counts} != "
            f"{expected_group_parameter_counts}"
        )

    def steps(parameters: Iterable[torch.nn.Parameter], role: str) -> List[int]:
        observed: set[int] = set()
        missing = 0
        for parameter in parameters:
            state = optimizer.state.get(parameter)
            if not state:
                missing += 1
                continue
            for name in ("step", "exp_avg", "exp_avg_sq"):
                if name not in state:
                    raise RuntimeError(f"{role} Adam state lacks {name}")
            if state["exp_avg"].shape != parameter.shape:
                raise RuntimeError(f"{role} exp_avg shape mismatch")
            if state["exp_avg_sq"].shape != parameter.shape:
                raise RuntimeError(f"{role} exp_avg_sq shape mismatch")
            step = state["step"]
            observed.add(int(step.item() if torch.is_tensor(step) else step))
        if missing:
            raise RuntimeError(f"parent optimizer is missing {missing} {role} states")
        return sorted(observed)

    theta_steps = steps(model.parameters(), "theta")
    psi_steps = steps(dynamics.parameters(), "psi")
    if expected_theta_steps is not None and theta_steps != expected_theta_steps:
        raise RuntimeError(
            f"theta Adam steps mismatch: {theta_steps} != {expected_theta_steps}"
        )
    if expected_psi_steps is not None and psi_steps != expected_psi_steps:
        raise RuntimeError(f"psi Adam steps mismatch: {psi_steps} != {expected_psi_steps}")
    return {
        "group_parameter_counts": counts,
        "theta_steps": theta_steps,
        "psi_steps": psi_steps,
    }


def _optimizer_checkpoint_tokens(training_config: Dict[str, Any]) -> set[int]:
    checkpoint_tokens = {int(value) for value in training_config["checkpoint_tokens"]}
    resumable = {
        int(value)
        for value in training_config.get(
            "resumable_checkpoint_tokens", training_config["checkpoint_tokens"]
        )
    }
    if not resumable <= checkpoint_tokens:
        raise RuntimeError("resumable checkpoint tokens must be checkpoint tokens")
    registered = {
        int(value) for value in training_config.get("registered_token_rungs", [])
    }
    if registered and not registered <= resumable:
        raise RuntimeError("every registered token rung must be optimizer-resumable")
    return resumable


def _parameter_grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    squares = [
        parameter.grad.detach().float().pow(2).sum()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not squares:
        return 0.0
    return float(torch.stack(squares).sum().sqrt().item())


def _should_validate_forward_counts(
    *,
    resume_kind: str,
    update: int,
    initial_update: int,
    qualification_updates: int,
) -> bool:
    """Keep count parity fail-closed at resume admission/qualification only."""

    return update == initial_update and (
        resume_kind != RESUME_FRESH or qualification_updates == 1
    )


def _step_requires_materialized_metrics(
    *,
    next_update: int,
    next_tokens: int,
    token_budget: int,
    log_every_updates: int,
    checkpoint_thresholds: Sequence[int],
    admission_update: bool,
    qualification_complete: bool,
) -> bool:
    """Whether this completed step's detached metrics must reach the host."""

    checkpoint_due = bool(
        checkpoint_thresholds and next_tokens >= checkpoint_thresholds[0]
    )
    return (
        next_update == 1
        or next_update % log_every_updates == 0
        or checkpoint_due
        or next_tokens >= token_budget
        or admission_update
        or qualification_complete
    )


def _position_count_vector(
    global_counts: torch.Tensor,
    *,
    horizon: int,
    global_repair_counts: Optional[torch.Tensor],
) -> torch.Tensor:
    """Return exact per-step position deltas without reading device scalars."""

    values = [
        global_counts[0],
        global_counts[1 : 1 + horizon].sum(),
        global_counts[1 + horizon :].sum(),
    ]
    if global_repair_counts is not None:
        values.append(global_repair_counts.sum())
    return torch.stack(values).to(dtype=torch.float64)


def _flush_position_totals(
    *,
    position_totals: Dict[str, int],
    position_total_names: Sequence[str],
    pending_position_counts: torch.Tensor,
) -> None:
    """Materialize accumulated exact integer counts at reporting boundaries."""

    if pending_position_counts.numel() != len(position_total_names):
        raise RuntimeError("pending position count shape does not match names")
    deltas = pending_position_counts.detach().cpu().tolist()
    for name, delta in zip(position_total_names, deltas):
        position_totals[name] += int(round(float(delta)))
    pending_position_counts.zero_()


def _materialize_global_metrics(
    *,
    global_values: torch.Tensor,
    horizon: int,
    lambda_h: float,
    lambda_kl: float,
    correction_loss_weight: float,
    global_repair_values: Optional[torch.Tensor],
) -> Dict[str, Any]:
    """Copy one compact report tensor while preserving existing reductions."""

    report_parts = [
        global_values[0].reshape(1),
        global_values[1 : 1 + horizon].mean().reshape(1),
        global_values[1 + horizon :].mean().reshape(1),
        global_values[1 : 1 + horizon],
        global_values[1 + horizon :],
    ]
    if global_repair_values is not None:
        report_parts.extend(
            (
                global_repair_values.mean().reshape(1),
                global_repair_values,
            )
        )
    host_values = torch.cat(report_parts).detach().cpu().tolist()
    cursor = 0
    metrics: Dict[str, Any] = {
        "ce": float(host_values[cursor]),
        "smooth_l1": float(host_values[cursor + 1]),
        "kl": float(host_values[cursor + 2]),
    }
    cursor += 3
    metrics["smooth_l1_by_horizon"] = [
        float(value) for value in host_values[cursor : cursor + horizon]
    ]
    cursor += horizon
    metrics["kl_by_horizon"] = [
        float(value) for value in host_values[cursor : cursor + horizon]
    ]
    cursor += horizon
    if global_repair_values is not None:
        repair_depths = int(global_repair_values.numel())
        metrics["deep_detached_repair_smooth_l1"] = float(host_values[cursor])
        cursor += 1
        metrics["deep_detached_repair_by_depth"] = [
            float(value) for value in host_values[cursor : cursor + repair_depths]
        ]
        cursor += repair_depths
    if cursor != len(host_values):
        raise RuntimeError("global metric report shape mismatch")
    metrics["loss"] = (
        metrics["ce"]
        + lambda_h * metrics["smooth_l1"]
        + lambda_kl * metrics["kl"]
        + correction_loss_weight
        * metrics.get("deep_detached_repair_smooth_l1", 0.0)
    )
    return metrics


def _infinite_batches(
    loader: DataLoader, sampler: DistributedSampler, batches_seen: int = 0
) -> Iterator[Dict[str, Any]]:
    batches_per_epoch = len(loader)
    if batches_per_epoch <= 0:
        raise RuntimeError("training loader has zero batches")
    epoch, offset = divmod(batches_seen, batches_per_epoch)
    while True:
        sampler.set_epoch(epoch)
        iterator = iter(loader)
        for _ in range(offset):
            next(iterator)
        yield from iterator
        epoch += 1
        offset = 0


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sampler_origins(config, resume_state, resume_kind):
    """A new corpus resets its own cursor, never the imported optimizer clock."""
    contract = config.get("continuation", {})
    if contract.get("protocol") not in ("optimizer_state_preserving_joint_d4_fresh_data_pilot_v1", PG_PROTOCOL):
        return {"tokens_seen": 0, "update": 0, "data_batches_seen": 0}
    origins = contract["fresh_sampler_origin"]
    if resume_kind != RESUME_JOINT_D4_DATA_FORK:
        if resume_state.get("fresh_sampler_origin") != origins:
            raise RuntimeError("fresh-data resume lost its registered sampler origins")
    stream_batches = int(resume_state["data_batches_seen"]) - origins["data_batches_seen"]
    stream_updates = int(resume_state["update"]) - origins["update"]
    if stream_batches < 0 or stream_batches != stream_updates * config["training"]["gradient_accumulation_steps"]:
        raise RuntimeError("fresh-data stream cursor differs from continuation updates")
    return origins


@torch.no_grad()
def _admit_parent_on_original_data(model, prepared_root, config, collator, device):
    """Keep the memorized-parent CE guard on its original corpus, not fresh data."""
    if prepared_root is None:
        raise RuntimeError("fresh-data initial fork requires --parent-admission-data")
    parent = config["continuation"]["parent"]
    if _sha256_path(prepared_root / "manifest.json") != parent["prepared_manifest_sha256"]:
        raise RuntimeError("parent admission corpus manifest differs from the original")
    rows = load_from_disk(str(prepared_root / "train"))
    manifest = json.loads((prepared_root / "manifest.json").read_text())
    if split_receipt(rows) != manifest["splits"]["train"]:
        raise RuntimeError("parent admission corpus contents differ from its receipt")
    sampler = DistributedSampler(rows, num_replicas=1, rank=0, shuffle=True,
                                 seed=config["training"]["seed"], drop_last=True)
    sampler.set_epoch(parent["sampler_epoch"])
    indices = list(iter(sampler))[parent["sampler_offset"]:parent["sampler_offset"] + 8]
    if len(indices) != 8:
        raise RuntimeError("registered old-data admission update crosses an epoch")
    rng = capture_rng_state()
    was_training = model.training
    model.eval()
    ce_sum, count = 0.0, 0
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for index in indices:
                batch = collator([rows[index]])
                ids = batch["input_ids"].to(device)
                mask = batch["assistant_loss_mask"][:, 1:].to(device).bool()
                hidden = model.model(input_ids=ids, attention_mask=batch["attention_mask"].to(device),
                                     use_cache=False, return_dict=True).last_hidden_state[:, :-1][mask]
                targets = ids[:, 1:][mask]
                for start in range(0, targets.numel(), 128):
                    logits = model.lm_head(hidden[start:start + 128]).float()
                    ce_sum += float(torch.nn.functional.cross_entropy(
                        logits, targets[start:start + 128], reduction="sum"))
                count += targets.numel()
    finally:
        model.train(was_training)
        restore_rng_state(rng)
    ce = ce_sum / count
    if not math.isfinite(ce) or ce > config["runtime"]["maximum_parent_admission_ce"]:
        raise RuntimeError(f"original-data parent CE admission failed: {ce}")
    return {"old_corpus_ce": ce, "assistant_positions": count,
            "problem_hashes": [rows[index]["problem_hash"] for index in indices],
            "fresh_corpus_ce_not_subject_to_memorized_threshold": True}


def _validate_countdown_prepared_data(prepared_root, config, manifest, prepared_config, contract):
    """Verify the task's real rows; never reuse OpenR1 correctness declarations."""
    from search.countdown_task import CountdownTask, task_hash, verify_expression

    authority = contract["prepared_data"]
    if manifest.get("git_commit") != authority["source_code_commit"]:
        raise RuntimeError("Countdown generator source commit mismatch")
    if config["dataset"]["id"] != "countdown4-legal-reduction-v1" or manifest.get("source") != {
        "dataset": config["dataset"]["id"], "split_seed": config["dataset"]["split_seed"]
    }:
        raise RuntimeError("Countdown dataset identity mismatch")
    if prepared_config["seed"] != config["dataset"]["split_seed"]:
        raise RuntimeError("Countdown split seed mismatch")
    if manifest.get("split_unit") != "sorted input number multiset plus target; one witness per problem":
        raise RuntimeError("Countdown mathematical split unit mismatch")
    if prepared_config["sequence_length"] != config["data"]["sequence_length"]:
        raise RuntimeError("Countdown prepared sequence length mismatch")
    tokenizer = manifest["tokenizer"]
    for field, bound in (("backend_sha256", "tokenizer_backend_sha256"),
                         ("chat_template_sha256", "tokenizer_chat_template_sha256")):
        if tokenizer.get(field) != authority.get(bound) or not authority.get(bound):
            raise RuntimeError(f"Countdown tokenizer {field} mismatch")
    if tokenizer["sequence_length"] != 512 or tokenizer.get("enable_thinking") is not False:
        raise RuntimeError("Countdown tokenizer protocol mismatch")
    repo = Path(__file__).resolve().parents[1]
    required_sources = {"nextlat/prepare_countdown_joint.py", "search/countdown_task.py",
                        "search/countdown_best_of_k.py", "nextlat/data.py"}
    if set(manifest.get("source_files", {})) != required_sources:
        raise RuntimeError("Countdown source receipts missing")
    for relative, expected in manifest["source_files"].items():
        if _sha256_path(repo / relative) != expected:
            raise RuntimeError(f"Countdown source changed: {relative}")
    excluded = manifest["exclusions"]["problem_hashes"]
    if (len(set(excluded)) != len(excluded)
        or len(excluded) != manifest["exclusions"]["count"]
        or hashlib.sha256("\n".join(sorted(excluded)).encode()).hexdigest()
        != manifest["exclusions"]["identity_sha256"]):
        raise RuntimeError("Countdown exclusion identity receipt mismatch")
    seen = set(excluded)
    total = 0
    split_counts = {"train": prepared_config["train_records"],
                    "recursive_holdout": config["data"]["recursive_holdout_problems"],
                    "stress_candidate": config["data"]["stress_candidate_problems"],
                    "composition_holdout": prepared_config["composition_records"]}
    for split, expected_count in split_counts.items():
        dataset = load_from_disk(str(prepared_root / split))
        receipt = manifest["splits"][split]
        observed = split_receipt(dataset)
        if observed != {key: receipt[key] for key in observed} or observed["problems"] != expected_count:
            raise RuntimeError(f"Countdown {split} record receipt mismatch")
        if _sha256_path(prepared_root / receipt["jsonl"]) != receipt["jsonl_sha256"]:
            raise RuntimeError(f"Countdown {split} JSONL receipt mismatch")
        for row in dataset:
            task = CountdownTask(tuple(row["inputs"]), row["target"], row["expression"], 0)
            identity = task_hash(task)
            if identity != row["problem_hash"] or identity in seen:
                raise RuntimeError("Countdown mathematical identity overlap or mismatch")
            seen.add(identity)
            check = verify_expression(row["expression"], task)
            if not check["passed"] or row["verifier"] != check:
                raise RuntimeError("Countdown expression verification failed")
            ids, mask = row["input_ids"], row["assistant_loss_mask"]
            start, stop = row["expression_start"], row["expression_stop"]
            if (row["source"] != "countdown4-legal-reduction-v1" or row["was_truncated"]
                or not 0 < start < stop == len(ids) - 1 < 512
                or len(ids) != row["token_count"] or len(mask) != len(ids)
                or list(mask) != [False] * start + [True] * (len(ids) - start)
                or sum(mask) != row["assistant_token_count"]
                or row["enable_thinking"] is not False or row["forced_prefix"] != "FINAL: "):
                raise RuntimeError("Countdown expression supervision boundary mismatch")
            signature = "".join(sorted(c for c in row["expression"] if c in "+-*/"))
            if (signature == "*-/") != (split == "composition_holdout"):
                raise RuntimeError("Countdown witness-composition split mismatch")
            total += 1
    counts = manifest["verifier_counts"]
    if (counts.get("verifier_calls") != total or counts.get("verifier_passed") != total
        or manifest.get("encoding_verifier_calls") != total
        or manifest.get("encoding_verifier_passed") != total):
        raise RuntimeError("Countdown verifier pass counts mismatch")
    if manifest["splits"]["train"]["tokens"] < config["data"]["prepared_train_tokens"]:
        raise RuntimeError("Countdown training pool below registered token target")
    manifest["manifest_sha256"] = _sha256_path(prepared_root / "manifest.json")
    return manifest


def _validate_prepared_data(
    prepared_root: Path,
    config: Dict[str, Any],
    datasets_by_split: Dict[str, Any],
    *,
    expected_source_commit: Optional[str] = None,
) -> Dict[str, Any]:
    manifest_path = prepared_root / "manifest.json"
    if not (prepared_root / "COMPLETE").is_file() or (
        prepared_root / "INCOMPLETE"
    ).exists():
        raise RuntimeError(f"prepared data is incomplete: {prepared_root}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing prepared-data manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    validation_config = config
    bound_source_commit = expected_source_commit or git_commit()
    if (config.get("continuation") is not None
        or config["data"].get("corpus_policy") == "countdown_verified_expression_v1"):
        prepared_contract = _prepared_data_authority(config)
        config_receipt = prepared_root / "config.resolved.json"
        if not config_receipt.is_file():
            raise RuntimeError("prepared data has no config receipt")
        if _sha256_path(config_receipt) != prepared_contract.get("config_sha256"):
            raise RuntimeError("prepared-data config hash mismatch")
        if _sha256_path(manifest_path) != prepared_contract.get("manifest_sha256"):
            raise RuntimeError("prepared-data manifest hash mismatch")
        validation_config = json.loads(config_receipt.read_text())
        bound_source_commit = str(prepared_contract["source_code_commit"])
        if config["data"].get("corpus_policy") == "countdown_verified_expression_v1":
            return _validate_countdown_prepared_data(
                prepared_root, config, manifest, validation_config,
                {"prepared_data": prepared_contract}
            )
    expected = validation_config["dataset"]
    expected_config_digest = hashlib.sha256(
        json.dumps(validation_config, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    if manifest.get("config_digest") != expected_config_digest:
        raise RuntimeError("prepared-data config digest does not match current config")
    source = manifest["source"]
    for key in ("dataset", "config", "split", "revision"):
        expected_key = "id" if key == "dataset" else key
        if source[key] != expected[expected_key]:
            raise RuntimeError(
                f"prepared-data {key} mismatch: {source[key]} != {expected[expected_key]}"
            )
    expected_tokenizer = {
        "model": validation_config["model"]["id"],
        "revision": validation_config["model"]["revision"],
        "sequence_length": int(validation_config["data"]["sequence_length"]),
    }
    if manifest.get("tokenizer") != expected_tokenizer:
        raise RuntimeError(
            f"prepared-data tokenizer mismatch: {manifest.get('tokenizer')} "
            f"!= {expected_tokenizer}"
        )
    selection = manifest.get("selection", {})
    corpus_policy = validation_config["data"].get("corpus_policy", "verified_openr1_v1")
    if config["data"].get("corpus_policy", "verified_openr1_v1") != corpus_policy:
        raise RuntimeError("prepared and training corpus policies differ")
    if corpus_policy not in ("verified_openr1_v1", "smoltalk_openr1_rehearsal_v1"):
        raise RuntimeError("unsupported prepared-data corpus policy")
    required_selection = {
        "require_correctness_math_verify": True,
        "require_reasoning_complete": True,
        "one_trajectory_per_problem": True,
        "reject_trajectories_over_sequence_length": True,
        "problem_key": "sha256(normalized problem text)",
        "split_seed": int(validation_config["dataset"]["split_seed"]),
        "stress_candidate_is_not_yet_search_selected": True,
    }
    if corpus_policy == "smoltalk_openr1_rehearsal_v1":
        if config["dataset"] != expected or not expected.get("components"):
            raise RuntimeError("rehearsal requires the registered mixed-source dataset")
        if source.get("components") != expected["components"]:
            raise RuntimeError("rehearsal source components differ from the config")
        required_selection.update({
            "require_correctness_math_verify": False,
            "require_reasoning_complete": False,
            "problem_key": "sha256(canonical prompt messages) for SmolTalk; sha256(normalized problem text) for OpenR1",
            "verification_by_source": {
                "openr1": "math_verify_and_reasoning_complete",
                "smoltalk": "synthetic_not_answer_verified",
            },
            "first_response_only": True,
            "smoltalk_test_reserved": True,
            "diagnostic_prompts_excluded": True,
            "math_disjoint_all_previous_pools": True,
        })
    for key, value in required_selection.items():
        if selection.get(key) != value:
            raise RuntimeError(
                f"prepared-data selection mismatch for {key}: "
                f"{selection.get(key)} != {value}"
            )
    expected_targets = {
        "prepared_train_tokens": int(
            validation_config["data"]["prepared_train_tokens"]
        ),
        "recursive_holdout_problems": int(
            validation_config["data"]["recursive_holdout_problems"]
        ),
        "stress_candidate_problems": int(
            validation_config["data"]["stress_candidate_problems"]
        ),
    }
    if manifest.get("targets") != expected_targets:
        raise RuntimeError(
            f"prepared-data targets mismatch: {manifest.get('targets')} "
            f"!= {expected_targets}"
        )
    if manifest.get("source_code_commit") != bound_source_commit:
        raise RuntimeError(
            "prepared data source commit does not match its bound authority: "
            f"{manifest.get('source_code_commit')} != {bound_source_commit}"
        )

    observed_hashes: Dict[str, set[str]] = {}
    for split, dataset in datasets_by_split.items():
        observed = split_receipt(dataset)
        recorded = manifest.get("splits", {}).get(split)
        if observed != recorded:
            raise RuntimeError(
                f"saved {split} receipt mismatch: {observed} != {recorded}"
            )
        observed_hashes[split] = {str(value) for value in dataset["problem_hash"]}
    for left, right in (
        ("train", "recursive_holdout"),
        ("train", "stress_candidate"),
        ("recursive_holdout", "stress_candidate"),
    ):
        overlap = observed_hashes[left] & observed_hashes[right]
        if overlap:
            raise RuntimeError(
                f"saved prepared data leaks {len(overlap)} problems between "
                f"{left} and {right}"
            )
    if manifest["splits"]["train"]["tokens"] < expected_targets[
        "prepared_train_tokens"
    ]:
        raise RuntimeError("prepared train split is below its configured token target")
    for split, target_key in (
        ("recursive_holdout", "recursive_holdout_problems"),
        ("stress_candidate", "stress_candidate_problems"),
    ):
        if manifest["splits"][split]["problems"] != expected_targets[target_key]:
            raise RuntimeError(f"prepared {split} problem count is not exact")
    manifest["manifest_sha256"] = _sha256_path(manifest_path)
    return manifest


def _distributed_validate_prepared_data(
    *,
    prepared_root: Path,
    config: Dict[str, Any],
    datasets_by_split: Dict[str, Any],
    rank: int,
    world_size: int,
) -> Dict[str, Any]:
    payload: list[Any] = [None, None]
    if rank == 0:
        try:
            payload[0] = _validate_prepared_data(
                prepared_root, config, datasets_by_split
            )
        except Exception as error:  # propagated so other ranks cannot hang
            payload[1] = f"{type(error).__name__}: {error}"
    if world_size > 1:
        dist.broadcast_object_list(payload, src=0)
    if payload[1] is not None:
        raise RuntimeError(f"prepared-data validation failed: {payload[1]}")
    if not isinstance(payload[0], dict):
        raise RuntimeError("prepared-data validation returned no manifest")
    return payload[0]


@torch.inference_mode()
def _quick_recursive_eval(
    *,
    model: torch.nn.Module,
    dynamics: NextLatDynamics,
    holdout: Any,
    collator: NextLatCollator,
    device: torch.device,
    depths: List[int],
    examples: int,
) -> Dict[str, Any]:
    model_was_training = model.training
    dynamics_was_training = dynamics.training
    model.eval()
    dynamics.eval()
    totals: Dict[str, Dict[str, float]] = {}
    total_anchors = 0
    requested = min(examples, len(holdout))
    evaluated = 0
    for index in range(len(holdout)):
        if evaluated >= requested:
            break
        batch = _to_device(collator([holdout[index]]), device)
        try:
            metrics = teacher_forced_recursive_metrics(
                model=model,
                dynamics=dynamics,
                batch=batch,
                depths=depths,
                anchors_per_example=2,
            )
        except RuntimeError as error:
            if "no held-out response has room" in str(error):
                continue
            raise
        evaluated += 1
        anchors = int(metrics["anchors"])
        total_anchors += anchors
        for depth, values in metrics["depths"].items():
            target = totals.setdefault(depth, {name: 0.0 for name in values})
            for name, value in values.items():
                target[name] += float(value) * anchors
    if model_was_training:
        model.train()
    if dynamics_was_training:
        dynamics.train()
    if total_anchors == 0:
        raise RuntimeError("quick recursive evaluation produced no anchors")
    return {
        "mode": "teacher_forced_random_response_anchor",
        "examples": evaluated,
        "anchors": total_anchors,
        "depths": {
            depth: {name: value / total_anchors for name, value in values.items()}
            for depth, values in totals.items()
        },
    }


def train(args: argparse.Namespace) -> None:
    source_commit = assert_claim_source_committed([args.config])
    config = json.loads(args.config.read_text())
    _validate_transformers_runtime(config)
    training_config = config["training"]
    tracking_config = config.get("tracking")
    horizon = int(config["nextlat"]["horizon"])
    if horizon not in (1, 4):
        raise RuntimeError(
            "the claim contract permits only the primary d=1 path and its single d=4 repair"
        )
    if config.get("continuation") is not None:
        _continuation_contract(config)
    rank, local_rank, world_size, device = _setup_distributed()
    hybrid_correction = _hybrid_correction_contract(
        config, world_size=world_size
    )
    if hybrid_correction is not None and horizon != 4:
        raise RuntimeError("hybrid correction requires the connected d=4 objective")
    correction_horizon = (
        hybrid_correction[0] if hybrid_correction is not None else None
    )
    correction_loss_weight = (
        hybrid_correction[1] if hybrid_correction is not None else 0.0
    )
    if config.get("continuation") is not None and args.resume is None:
        raise RuntimeError("continued joint d=4 training requires --resume")
    if args.resume_config is not None and args.resume is None:
        raise RuntimeError("--resume-config requires --resume")
    if args.qualification_updates not in (0, 1):
        raise RuntimeError("qualification mode permits exactly zero or one update")
    token_budget = int(
        training_config["decision_tokens"]
        if args.token_budget is None
        else args.token_budget
    )
    if token_budget <= 0:
        raise ValueError("token_budget must be positive")
    registered_rungs = {
        int(value) for value in training_config.get("registered_token_rungs", [])
    }
    if registered_rungs and token_budget not in registered_rungs:
        raise RuntimeError(
            f"token budget {token_budget:,} is not a registered rung: "
            f"{sorted(registered_rungs)}"
        )
    optimizer_checkpoint_tokens = _optimizer_checkpoint_tokens(training_config)
    artifact_checkpoint_tokens = {
        int(value) for value in training_config.get("wandb_artifact_tokens", [])
    }
    receipt_artifact_tokens = {
        int(value)
        for value in training_config.get("wandb_receipt_artifact_tokens", [])
    }
    if not artifact_checkpoint_tokens <= optimizer_checkpoint_tokens:
        raise RuntimeError("W&B artifact tokens must be optimizer-resumable checkpoints")
    if not receipt_artifact_tokens <= optimizer_checkpoint_tokens:
        raise RuntimeError(
            "W&B receipt artifact tokens must be optimizer-resumable checkpoints"
        )
    if artifact_checkpoint_tokens & receipt_artifact_tokens:
        raise RuntimeError("full and receipt-only W&B artifact tokens must be disjoint")
    if tracking_config is not None and args.wandb_run_id is None:
        raise RuntimeError("--wandb-run-id is required by this config")
    if tracking_config is None and args.wandb_run_id is not None:
        raise RuntimeError("--wandb-run-id was provided but tracking is not configured")
    seed = int(training_config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    training_sdpa_backend = _resolve_training_sdpa_backend(config["model"])

    accumulation_steps = int(training_config["gradient_accumulation_steps"])
    if accumulation_steps < 1:
        raise RuntimeError("gradient_accumulation_steps must be positive")
    execution_micro_batch_size = _resolve_execution_micro_batch_size(
        training_config
    )
    continuation = config.get("continuation")
    pg_config = config.get("recurrent_policy_gradient")
    if pg_config is not None:
        if (not isinstance(pg_config, dict)
            or pg_config.get("lambda_pg") not in (0, 1)
            or {**pg_config, "lambda_pg": 1.0} not in (PG_POLICY, PG_SCALED_POLICY)
            or _continuation_contract(config)["protocol"] != PG_PROTOCOL
            or world_size != 1 or hybrid_correction is not None):
            raise RuntimeError("unsupported Countdown PG phase configuration")
    elif isinstance(continuation, dict) and continuation.get("protocol") == PG_PROTOCOL:
        raise RuntimeError("PG phase has no explicit policy configuration")
    pg_control_first_only = bool(pg_config is not None and pg_config["lambda_pg"] == 0
                                 and pg_config.get("control_auxiliary") == "first_group_only")
    execution_pad_to_multiple = _resolve_execution_pad_to_multiple(
        training_config,
        systems_fork=(
            isinstance(continuation, dict)
            and continuation.get("protocol")
            == "optimizer_state_preserving_joint_d4_systems_fork_v1"
        ),
    )
    expected_world_size = training_config.get("expected_world_size")
    if expected_world_size is not None and int(expected_world_size) != world_size:
        raise RuntimeError(
            f"registered world size {expected_world_size} != runtime {world_size}"
        )
    resume_state: Dict[str, Any] = {}
    resume_kind = RESUME_FRESH
    if args.resume is not None:
        if rank == 0:
            _require_resumable_checkpoint(args.resume)
        _barrier(world_size)
        resume_state = json.loads((args.resume / "train_state.json").read_text())
        parent_config_path = (
            args.resume_config
            if args.resume_config is not None
            else args.resume.parent / "config.resolved.json"
        )
        if not parent_config_path.exists():
            raise RuntimeError(
                f"resume checkpoint has no parent config receipt: {parent_config_path}"
            )
        parent_config = json.loads(parent_config_path.read_text())
        resume_kind = _validate_parent_continuation(
            config=config,
            parent_config=parent_config,
            parent_config_path=parent_config_path,
            resume_path=args.resume,
            resume_state=resume_state,
            current_source_commit=source_commit,
        )
        required_resume_fields = {
            "update",
            "tokens_seen",
            "data_batches_seen",
            "world_size",
            "micro_batch_size",
            "gradient_accumulation_steps",
            "sampler_seed",
            "batches_per_epoch",
            "sampler_epoch",
            "sampler_offset",
            "prepared_manifest_sha256",
            "train_record_digest",
            "position_totals",
            "source_code_commit",
            "optimizer_included",
            "rng_state_included",
        }
        if tracking_config is not None:
            required_resume_fields.add("wandb_run_id")
        if pg_config is not None and resume_kind == RESUME_SAME_SOURCE:
            required_resume_fields.update({"pg_first_mixed_audited", "pg_groups_seen"})
            if pg_config == PG_SCALED_POLICY or pg_control_first_only:
                required_resume_fields.add("pg_auxiliary_skipped_updates")
        missing = required_resume_fields - set(resume_state)
        if missing:
            raise RuntimeError(f"resume checkpoint is missing state: {sorted(missing)}")
        if int(resume_state["world_size"]) != world_size:
            raise RuntimeError(
                f"resume world_size mismatch: {resume_state['world_size']} != {world_size}"
            )
        if int(resume_state["gradient_accumulation_steps"]) != accumulation_steps:
            raise RuntimeError(
                "resume gradient_accumulation_steps mismatch: "
                f"{resume_state['gradient_accumulation_steps']} != {accumulation_steps}"
            )
        if (
            resume_kind == RESUME_SAME_SOURCE
            and resume_state["source_code_commit"] != source_commit
        ):
            raise RuntimeError(
                "resume source commit mismatch: "
                f"{resume_state['source_code_commit']} != {source_commit}"
            )
        if resume_state["optimizer_included"] is not True:
            raise RuntimeError("resume checkpoint does not include optimizer state")
        if not (args.resume / "optimizer.pt").is_file():
            raise RuntimeError("resume checkpoint is missing optimizer.pt")
        if (
            tracking_config is not None
            and resume_kind == RESUME_SAME_SOURCE
            and resume_state["wandb_run_id"] != args.wandb_run_id
        ):
            raise RuntimeError("resume W&B run ID does not match --wandb-run-id")
    run_config = json.loads(json.dumps(config))
    run_config["run"] = {
        "token_budget": token_budget,
        "resume_from": str(args.resume) if args.resume is not None else None,
        "source_code_commit": source_commit,
        "world_size": world_size,
        "wandb_run_id": args.wandb_run_id,
        "resume_kind": resume_kind,
        "qualification_updates": args.qualification_updates,
    }
    if rank == 0:
        initialize_run_directory(
            args.run_dir,
            run_config,
            " ".join(shlex.quote(part) for part in sys.argv),
        )
    _barrier(world_size)

    tracker: Optional[WandbTracker] = None
    tracking_error: list[Optional[str]] = [None]
    if rank == 0 and tracking_config is not None:
        try:
            tracker = start_wandb_tracker(
                tracking_config=tracking_config,
                run_dir=args.run_dir,
                run_id=str(args.wandb_run_id),
                resume_mode=_tracker_resume_mode(args.resume, resume_kind),
                experiment_config=config,
                source_code_commit=source_commit,
                phase="train",
            )
        except Exception as error:
            tracking_error[0] = f"{type(error).__name__}: {error}"
    if world_size > 1:
        dist.broadcast_object_list(tracking_error, src=0)
    if tracking_error[0] is not None:
        raise RuntimeError(f"required W&B initialization failed: {tracking_error[0]}")
    _barrier(world_size)

    # W&B initialization must not perturb the registered model/data RNG streams.
    random.seed(seed + rank)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model_source: Any = (
        args.resume / "model" if args.resume is not None else config["model"]["id"]
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        revision=None if args.resume is not None else config["model"]["revision"],
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        raise RuntimeError("Qwen tokenizer has no pad token")
    if config["data"].get("corpus_policy") == "countdown_verified_expression_v1":
        authority = _prepared_data_authority(config)
        for value, key in ((tokenizer.backend_tokenizer.to_str(), "tokenizer_backend_sha256"),
                           (tokenizer.chat_template, "tokenizer_chat_template_sha256")):
            if hashlib.sha256(value.encode()).hexdigest() != authority.get(key):
                raise RuntimeError(f"Countdown runtime tokenizer mismatch: {key}")
    train_data = load_from_disk(str(args.prepared_root / "train"))
    holdout = load_from_disk(str(args.prepared_root / "recursive_holdout"))
    stress_candidate = load_from_disk(
        str(args.prepared_root / "stress_candidate")
    )
    prepared_manifest = _distributed_validate_prepared_data(
        prepared_root=args.prepared_root,
        config=config,
        datasets_by_split={
            "train": train_data,
            "recursive_holdout": holdout,
            "stress_candidate": stress_candidate,
        },
        rank=rank,
        world_size=world_size,
    )
    del stress_candidate
    collator = NextLatCollator(
        tokenizer.pad_token_id, int(config["data"]["sequence_length"])
    )
    sampler = DistributedSampler(
        train_data,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=seed,
        drop_last=True,
    )
    loader = DataLoader(
        train_data,
        batch_size=int(training_config["micro_batch_size"]),
        sampler=sampler,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )

    sampler_origins = _sampler_origins(config, resume_state, resume_kind)
    if args.resume is not None:
        expected_resume_values = {
            "micro_batch_size": int(training_config["micro_batch_size"]),
            "gradient_accumulation_steps": accumulation_steps,
            "sampler_seed": seed,
            "batches_per_epoch": len(loader),
            "prepared_manifest_sha256": prepared_manifest["manifest_sha256"],
            "train_record_digest": prepared_manifest["splits"]["train"][
                "record_digest"
            ],
        }
        if resume_kind == RESUME_JOINT_D4_DATA_FORK:
            for key in ("batches_per_epoch", "prepared_manifest_sha256", "train_record_digest"):
                expected_resume_values.pop(key)
        for key, expected_value in expected_resume_values.items():
            if resume_state[key] != expected_value:
                raise RuntimeError(
                    f"resume {key} mismatch: {resume_state[key]} != {expected_value}"
                )
        expected_epoch, expected_offset = divmod(
            int(resume_state["data_batches_seen"]) - sampler_origins["data_batches_seen"], len(loader)
        )
        if resume_kind != RESUME_JOINT_D4_DATA_FORK and int(resume_state["sampler_epoch"]) != expected_epoch:
            raise RuntimeError("resume sampler_epoch does not match batches seen")
        if resume_kind != RESUME_JOINT_D4_DATA_FORK and int(resume_state["sampler_offset"]) != expected_offset:
            raise RuntimeError("resume sampler_offset does not match batches seen")
        expected_batches_seen = int(resume_state["update"]) * accumulation_steps
        if int(resume_state["data_batches_seen"]) != expected_batches_seen:
            raise RuntimeError(
                "resume data_batches_seen must equal optimizer updates times "
                "gradient_accumulation_steps"
            )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_source,
        revision=None if args.resume is not None else config["model"]["revision"],
        torch_dtype=torch.float32,
        attn_implementation=config["model"]["attention"],
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to(device)
    _validate_loaded_rope_theta(config, base_model.config)
    base_model.config.use_cache = False
    if bool(config["model"].get("gradient_checkpointing", True)):
        base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    _validate_embedding_tying(base_model)
    dynamics = NextLatDynamics(
        int(base_model.config.hidden_size),
        projection_factor=float(config["nextlat"]["projection_factor"]),
        dropout=float(config["nextlat"]["dropout"]),
    ).to(device=device, dtype=torch.float32)
    if args.resume is not None:
        load_dynamics_checkpoint(dynamics, args.resume, device, verify=False)
    loss_config = NextLatLossConfig(
        lambda_h=float(config["nextlat"]["lambda_h"]),
        lambda_kl=float(config["nextlat"]["lambda_kl"]),
        vocab_chunk_tokens=int(config["nextlat"]["vocab_chunk_tokens"]),
        horizon=horizon,
        anchor_noise_relative=float(config["nextlat"].get("anchor_noise_relative", 0.0)),
        kl_implementation=str(
            config["nextlat"].get("kl_implementation", "autograd")
        ),
    )
    joint: torch.nn.Module = NextLatQwen(base_model, dynamics, loss_config).to(device)
    if world_size > 1:
        joint = DistributedDataParallel(
            joint,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )
    core: NextLatQwen = joint.module if isinstance(joint, DistributedDataParallel) else joint
    joint.train()
    if resume_kind == RESUME_JOINT_D4_DATA_FORK and config["continuation"].get("parent_admission", "original_data_ce") == "original_data_ce":
        receipt = _admit_parent_on_original_data(
            core.model, args.parent_admission_data, config, collator, device,
        )
        (args.run_dir / "original-data-parent-admission.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps({"original_data_parent_admission": receipt}), flush=True)
    if not core.model.training or not core.dynamics.training:
        raise RuntimeError("Qwen and dynamics must both be in training mode")
    frozen_theta = [name for name, parameter in core.model.named_parameters() if not parameter.requires_grad]
    frozen_psi = [name for name, parameter in core.dynamics.named_parameters() if not parameter.requires_grad]
    if frozen_theta or frozen_psi:
        raise RuntimeError(
            "joint continuation requires every theta and psi parameter trainable; "
            f"frozen theta={len(frozen_theta)} psi={len(frozen_psi)}"
        )

    parameter_counts = {
        "theta": sum(p.numel() for p in core.model.parameters()),
        "psi": sum(p.numel() for p in core.dynamics.parameters()),
        "hidden_size": int(base_model.config.hidden_size),
        "projection_size": dynamics.projection_size,
        "tied_embeddings": bool(base_model.config.tie_word_embeddings),
    }
    expected_psi = config["nextlat"].get("expected_parameters")
    if expected_psi is not None and parameter_counts["psi"] != expected_psi:
        raise RuntimeError(f"sidecar parameter count mismatch: {parameter_counts['psi']} != {expected_psi}")
    if rank == 0:
        write_jsonl(args.run_dir / "metrics.jsonl", {"type": "parameter_counts", **parameter_counts})
        print(json.dumps({"parameter_counts": parameter_counts}), flush=True)

    groups = _optimizer_groups(
        core.model,
        float(training_config["base_learning_rate"]),
        float(training_config["weight_decay"]),
        "theta",
    ) + _optimizer_groups(
        core.dynamics,
        float(training_config["dynamics_learning_rate"]),
        float(training_config["weight_decay"]),
        "psi",
    )
    optimizer = torch.optim.AdamW(
        groups,
        betas=(
            float(training_config["adam_beta1"]),
            float(training_config["adam_beta2"]),
        ),
        eps=float(training_config["adam_eps"]),
        fused=True,
    )
    optimizer_import: Optional[Dict[str, Any]] = None
    rng_resume_mode = "fresh_seed"
    if args.resume is not None:
        optimizer_state = torch.load(
            args.resume / "optimizer.pt", map_location="cpu", weights_only=False
        )
        optimizer.load_state_dict(optimizer_state)
        del optimizer_state
        torch.cuda.empty_cache()
        continuation_parent = (
            _continuation_contract(config)["parent"]
            if config.get("continuation") is not None
            else {}
        )
        optimizer_import = _validate_loaded_optimizer_state(
            optimizer=optimizer,
            model=core.model,
            dynamics=core.dynamics,
            expected_group_parameter_counts=(
                [
                    int(value)
                    for value in continuation_parent[
                        "optimizer_group_parameter_counts"
                    ]
                ]
                if continuation_parent
                else None
            ),
            expected_theta_steps=(
                [int(value) for value in continuation_parent["theta_adam_steps"]]
                if resume_kind in SEALED_JOINT_FORKS
                else None
            ),
            expected_psi_steps=(
                [int(value) for value in continuation_parent["psi_adam_steps"]]
                if resume_kind in SEALED_JOINT_FORKS
                else None
            ),
        )
        rng_path = args.resume / "rng_state.pt"
        if not rng_path.is_file():
            raise RuntimeError("resume checkpoint is missing rng_state.pt")
        restore_rng_state(torch.load(rng_path, map_location="cpu", weights_only=False))
        rng_resume_mode = (
            "restored_countdown_same_corpus_policy_phase_v1"
            if resume_kind == RESUME_COUNTDOWN_PG_FORK
            else
            "restored_original_parent_with_registered_new_corpus_sampler_v1"
            if resume_kind == RESUME_JOINT_D4_DATA_FORK
            else
            "restored_sealed_joint_d4_anchor_noise_parent_v1"
            if resume_kind == RESUME_JOINT_D4_NOISE_FORK
            else
            "restored_sealed_joint_d4_systems_parent_v1"
            if resume_kind == RESUME_JOINT_D4_SYSTEMS_FORK
            else (
                "restored_sealed_joint_d16_parent_v1"
                if resume_kind in SEALED_JOINT_FORKS
                else "restored_checkpoint"
            )
        )
    optimizer.zero_grad(set_to_none=True)

    tokens_seen = int(resume_state.get("tokens_seen", 0))
    update = int(resume_state.get("update", 0))
    position_total_names = ["ce", "transition", "kl"]
    if hybrid_correction is not None:
        position_total_names.append("deep_detached_repair")
    position_totals = {
        name: int(resume_state.get("position_totals", {}).get(name, 0))
        for name in position_total_names
    }
    pending_position_counts = (
        torch.zeros(len(position_total_names), device=device, dtype=torch.float64)
        if rank == 0
        else None
    )
    if tokens_seen >= token_budget:
        raise RuntimeError(
            f"resume already has {tokens_seen:,} tokens, not below budget {token_budget:,}"
        )
    eval_interval = int(training_config.get("recursive_eval_every_tokens", 0))
    eval_examples = int(training_config.get("recursive_eval_examples", 0))
    if (eval_interval > 0) != (eval_examples > 0):
        raise RuntimeError(
            "recursive eval interval and example count must both be positive or both be zero"
        )
    recursive_eval_enabled = eval_interval > 0
    next_eval = (
        ((tokens_seen // eval_interval) + 1) * eval_interval
        if recursive_eval_enabled
        else None
    )
    checkpoint_thresholds = sorted(
        threshold
        for threshold in (int(value) for value in training_config["checkpoint_tokens"])
        if tokens_seen < threshold <= token_budget
    )
    data_batches_seen = int(resume_state.get("data_batches_seen", 0))
    batches = _infinite_batches(loader, sampler, batches_seen=data_batches_seen - sampler_origins["data_batches_seen"])
    started = time.time()
    initial_tokens = tokens_seen
    initial_update = update
    pg_audited = bool(resume_state.get("pg_first_mixed_audited", False))
    pg_groups_seen = int(resume_state.get("pg_groups_seen", 0))
    pg_skipped_updates = int(resume_state.get("pg_auxiliary_skipped_updates", 0))
    if pg_config is not None:
        phase_updates = update - int(continuation["parent"]["update"])
        expected_groups = min(phase_updates, 1) if pg_control_first_only else phase_updates
        if (not 0 <= phase_updates < pg_config["updates"] or pg_groups_seen != expected_groups
            or pg_skipped_updates != phase_updates - expected_groups):
            raise RuntimeError("PG phase update/group clock mismatch")
        expected_budget = tokens_seen + phase_token_budget(
            train_data, seed, data_batches_seen - sampler_origins["data_batches_seen"],
            pg_config["updates"] - phase_updates)
        if token_budget != expected_budget:
            raise RuntimeError(f"{pg_config['updates']}-update PG phase requires exact token budget {expected_budget}")
        pg_record_indices = {value: i for i, value in enumerate(train_data["problem_hash"])}
    initial_position_totals = dict(position_totals)
    last_global_metrics: Dict[str, Any] = {}
    last_checkpoint_tokens: Optional[int] = None
    qualification_optimizer_state: Optional[Dict[str, Any]] = None

    def checkpoint_state(
        requested_checkpoint_tokens: int, optimizer_included: bool
    ) -> Dict[str, Any]:
        sampler_epoch, sampler_offset = divmod(data_batches_seen - sampler_origins["data_batches_seen"], len(loader))
        return {
            "update": update,
            "tokens_seen": tokens_seen,
            "requested_checkpoint_tokens": requested_checkpoint_tokens,
            "token_budget": token_budget,
            "data_batches_seen": data_batches_seen,
            "fresh_sampler_origin": sampler_origins,
            **({"pg_first_mixed_audited": pg_audited, "pg_groups_seen": pg_groups_seen}
               if pg_config is not None else {}),
            **({"pg_auxiliary_skipped_updates": pg_skipped_updates}
               if pg_config is not None and pg_config["updates"] == 4096 else {}),
            "world_size": world_size,
            "micro_batch_size": int(training_config["micro_batch_size"]),
            "execution_micro_batch_size": execution_micro_batch_size,
            "execution_pad_to_multiple": execution_pad_to_multiple,
            "gradient_accumulation_steps": accumulation_steps,
            "sampler_seed": seed,
            "batches_per_epoch": len(loader),
            "sampler_epoch": sampler_epoch,
            "sampler_offset": sampler_offset,
            "prepared_manifest_sha256": prepared_manifest["manifest_sha256"],
            "train_record_digest": prepared_manifest["splits"]["train"][
                "record_digest"
            ],
            "position_totals": dict(position_totals),
            "parameter_counts": parameter_counts,
            "corpus_tokens_seen": tokens_seen,
            "assistant_tokens_seen": position_totals["ce"],
            "source_code_commit": source_commit,
            "optimizer_included": optimizer_included,
            "rng_state_included": optimizer_included,
            "rng_resume_mode": rng_resume_mode,
            "resume_kind": resume_kind,
            "optimizer_state_policy": _optimizer_state_policy(
                config, hybrid_correction=hybrid_correction is not None
            ),
            "optimizer_import": optimizer_import,
            "wandb_run_id": args.wandb_run_id,
            "parent_checkpoint": str(args.resume)
            if args.resume is not None
            else None,
        }

    if training_config.get("save_initial_checkpoint", False) and args.resume is None:
        _barrier(world_size)
        if rank == 0:
            rng_state = capture_rng_state()
            save_checkpoint(
                checkpoint_dir=args.run_dir / "checkpoint-000000000-tokens",
                model=core.model, dynamics=core.dynamics, tokenizer=tokenizer,
                train_state=checkpoint_state(0, False),
            )
            restore_rng_state(rng_state)
        _barrier(world_size)

    # Establish the recursive baseline before any update (or at the resume point).
    if recursive_eval_enabled:
        _barrier(world_size)
    if recursive_eval_enabled and rank == 0:
        with _training_sdpa_context(training_sdpa_backend), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            recursive = _quick_recursive_eval(
                model=core.model,
                dynamics=core.dynamics,
                holdout=holdout,
                collator=collator,
                device=device,
                depths=[int(value) for value in config["recursive_gate"]["depths"]],
                examples=eval_examples,
            )
        record = {
            "type": "recursive_eval",
            "phase": "pre_update" if args.resume is None else "resume_pre_update",
            "update": update,
            "tokens_seen": tokens_seen,
            **recursive,
        }
        write_jsonl(args.run_dir / "metrics.jsonl", record)
        if tracker is not None:
            tracker.log_record("recursive", record)
        print(json.dumps(record, sort_keys=True), flush=True)
    if recursive_eval_enabled:
        _barrier(world_size)

    while tokens_seen < token_budget:
        update_started = time.time()
        update_initial_tokens = tokens_seen
        logical_micro_batches = [next(batches) for _ in range(accumulation_steps)]
        micro_batches = [
            _to_device(batch, device)
            for batch in _coalesce_execution_micro_batches(
                logical_micro_batches,
                execution_micro_batch_size=execution_micro_batch_size,
                pad_token_id=tokenizer.pad_token_id,
                pad_to_multiple=execution_pad_to_multiple,
            )
        ]
        local_count_rows = [
            nextlat_position_counts(
                batch["attention_mask"], batch["assistant_loss_mask"], horizon
            ).to(dtype=torch.float32)
            for batch in micro_batches
        ]
        local_counts = torch.stack(local_count_rows, dim=0).sum(dim=0)
        global_counts = _all_reduce_sum(local_counts, world_size)
        if bool((global_counts <= 0).any()):
            raise RuntimeError(f"global loss count is zero: {global_counts.tolist()}")
        local_repair_count_rows: list[torch.Tensor] = []
        global_repair_counts: Optional[torch.Tensor] = None
        if correction_horizon is not None:
            local_repair_count_rows = [
                nextlat_position_counts(
                    batch["attention_mask"],
                    batch["assistant_loss_mask"],
                    correction_horizon,
                )[1 + horizon : 1 + correction_horizon].to(dtype=torch.float32)
                for batch in micro_batches
            ]
            global_repair_counts = _all_reduce_sum(
                torch.stack(local_repair_count_rows, dim=0).sum(dim=0),
                world_size,
            )
            if bool((global_repair_counts <= 0).any()):
                raise RuntimeError(
                    "global deep-repair count is zero: "
                    f"{global_repair_counts.tolist()}"
                )
        local_tokens = torch.stack(
            [
                batch["attention_mask"].sum().to(dtype=torch.float64)
                for batch in micro_batches
            ]
        ).sum()
        global_step_tokens = int(
            _all_reduce_sum(local_tokens, world_size).item()
        )
        _set_warmup_lr(
            optimizer,
            tokens_seen + global_step_tokens,
            int(training_config["warmup_tokens"]),
        )

        next_update = update + 1
        next_tokens = tokens_seen + global_step_tokens
        admission_update = (
            resume_kind in SEALED_JOINT_FORKS and update == initial_update
        )
        qualification_complete = (
            args.qualification_updates == 1
            and next_update == initial_update + 1
        )
        validate_forward_counts = _should_validate_forward_counts(
            resume_kind=resume_kind,
            update=update,
            initial_update=initial_update,
            qualification_updates=args.qualification_updates,
        )
        materialize_step_metrics = _step_requires_materialized_metrics(
            next_update=next_update,
            next_tokens=next_tokens,
            token_budget=token_budget,
            log_every_updates=int(training_config["log_every_updates"]),
            checkpoint_thresholds=checkpoint_thresholds,
            admission_update=admission_update,
            qualification_complete=qualification_complete,
        )

        local_component_sums = (
            torch.zeros_like(global_counts, dtype=torch.float64)
            if materialize_step_metrics
            else None
        )
        local_repair_sums = (
            torch.zeros_like(global_repair_counts, dtype=torch.float64)
            if global_repair_counts is not None and materialize_step_metrics
            else None
        )
        for micro_index, (batch, micro_counts) in enumerate(
            zip(micro_batches, local_count_rows)
        ):
            synchronize = micro_index == len(micro_batches) - 1
            sync_context = (
                nullcontext()
                if synchronize or not isinstance(joint, DistributedDataParallel)
                else joint.no_sync()
            )
            with sync_context:
                with _training_sdpa_context(
                    training_sdpa_backend
                ), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = joint(
                        batch["input_ids"],
                        batch["attention_mask"],
                        batch["assistant_loss_mask"],
                        return_training_states=(hybrid_correction is not None
                                                or (admission_update and micro_index == 0)),
                    )
                if admission_update and micro_index == 0 and rank == 0:
                    valid = batch["attention_mask"].bool()
                    input_scale = {
                        "type": "first_batch_input_scale",
                        "tokens_seen": tokens_seen,
                        "hidden_rms": float(outputs["exact_hidden_states"].detach()[valid].float().square().mean().sqrt()),
                        "embedding_rms": float(outputs["token_embeddings"].detach()[valid].float().square().mean().sqrt()),
                        "anchor_noise_relative_half_width": loss_config.anchor_noise_relative,
                    }
                    write_jsonl(args.run_dir / "metrics.jsonl", input_scale)
                    print(json.dumps(input_scale), flush=True)
                    if tracker is not None:
                        tracker.log_record("input_scale", input_scale)
                    if hybrid_correction is None:
                        outputs.pop("exact_hidden_states")
                        outputs.pop("token_embeddings")
                if validate_forward_counts:
                    observed_counts = torch.cat(
                        (
                            outputs["ce_positions"].reshape(1),
                            outputs["transition_positions_by_horizon"].reshape(-1),
                            outputs["kl_positions_by_horizon"].reshape(-1),
                        )
                    ).to(dtype=torch.float32)
                    if not torch.equal(observed_counts, micro_counts):
                        raise RuntimeError(
                            "precomputed NextLat position counts do not match the forward"
                        )
                scales = world_size * micro_counts / global_counts
                loss = (
                    outputs["ce"] * scales[0]
                    + loss_config.lambda_h
                    * (
                        outputs["smooth_l1_by_horizon"]
                        * scales[1 : 1 + horizon]
                    ).mean()
                    + loss_config.lambda_kl
                    * (
                        outputs["kl_by_horizon"] * scales[1 + horizon :]
                    ).mean()
                )
                if hybrid_correction is not None:
                    repair_connected_states = outputs[
                        "predicted_states_by_horizon"
                    ][-1].detach()
                    repair_exact_states = outputs["exact_hidden_states"].detach()
                    repair_token_embeddings = outputs["token_embeddings"].detach()
                loss.backward()
                if hybrid_correction is not None:
                    if global_repair_counts is None:
                        raise RuntimeError("deep-repair counts were not initialized")
                    repair_micro_counts = local_repair_count_rows[micro_index]
                    repair_scales = (
                        world_size * repair_micro_counts / global_repair_counts
                    )
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        repair_iterator = iter_deep_detached_repair_losses(
                            dynamics=core.dynamics,
                            connected_states=repair_connected_states,
                            exact_hidden_states=repair_exact_states,
                            token_embeddings=repair_token_embeddings,
                            attention_mask=batch["attention_mask"],
                            assistant_loss_mask=batch["assistant_loss_mask"],
                            connected_depth=horizon,
                            repair_horizon=correction_horizon,
                        )
                        repair_depths = correction_horizon - horizon
                        observed_repair_steps = 0
                        for repair_index, repair_output in enumerate(repair_iterator):
                            observed_depth = int(repair_output["depth"])
                            expected_depth = horizon + 1 + repair_index
                            if observed_depth != expected_depth:
                                raise RuntimeError(
                                    "deep-repair depth mismatch: "
                                    f"{observed_depth} != {expected_depth}"
                                )
                            if validate_forward_counts:
                                observed_count = repair_output[
                                    "transition_positions"
                                ].to(dtype=torch.float32)
                                if not torch.equal(
                                    observed_count,
                                    repair_micro_counts[repair_index],
                                ):
                                    raise RuntimeError(
                                        "precomputed deep-repair position count does "
                                        f"not match depth {observed_depth}"
                                    )
                            repair_loss = (
                                correction_loss_weight
                                * repair_output["smooth_l1"]
                                * repair_scales[repair_index]
                                / repair_depths
                            )
                            repair_loss.backward()
                            if local_repair_sums is not None:
                                local_repair_sums[repair_index] += (
                                    repair_output["smooth_l1"].detach()
                                    * repair_micro_counts[repair_index]
                                ).to(dtype=torch.float64)
                            observed_repair_steps += 1
                            del repair_output, repair_loss
                    if observed_repair_steps != correction_horizon - horizon:
                        raise RuntimeError(
                            "deep-repair iterator returned the wrong number of depths"
                        )
            if local_component_sums is not None:
                local_component_sums += torch.cat(
                    (
                        (outputs["ce"].detach() * micro_counts[0]).reshape(1),
                        outputs["smooth_l1_by_horizon"].detach()
                        * micro_counts[1 : 1 + horizon],
                        outputs["kl_by_horizon"].detach()
                        * micro_counts[1 + horizon :],
                    )
                ).to(dtype=torch.float64)
            if hybrid_correction is not None:
                del (
                    repair_connected_states,
                    repair_exact_states,
                    repair_token_embeddings,
                    repair_iterator,
                )
            del outputs, loss
        data_batches_seen += accumulation_steps
        global_values: Optional[torch.Tensor] = None
        if local_component_sums is not None:
            global_sums = _all_reduce_sum(local_component_sums, world_size)
            global_values = global_sums / global_counts.to(dtype=torch.float64)
        global_repair_values: Optional[torch.Tensor] = None
        if local_repair_sums is not None:
            if global_repair_counts is None:
                raise RuntimeError("deep-repair counts were not initialized")
            global_repair_sums = _all_reduce_sum(local_repair_sums, world_size)
            global_repair_values = global_repair_sums / global_repair_counts.to(
                dtype=torch.float64
            )
        if admission_update:
            if global_values is None:
                raise RuntimeError("admission update has no global component metrics")
            maximum_admission_ce = float(
                config["runtime"]["maximum_parent_admission_ce"]
            )
            observed_admission_ce = float(global_values[0].item())
            if (
                not math.isfinite(observed_admission_ce)
                or (resume_kind not in (RESUME_JOINT_D4_DATA_FORK, RESUME_COUNTDOWN_PG_FORK)
                    and observed_admission_ce > maximum_admission_ce)
            ):
                raise RuntimeError(
                    "joint d=4 parent admission CE exceeds the runtime contract: "
                    f"{observed_admission_ce:.6f} > {maximum_admission_ce:.6f}"
                )
        if pg_config is not None and not (pg_control_first_only and pg_groups_seen == 1):
            pg_row = train_data[pg_record_indices[micro_batches[0]["problem_hash"][0]]]
            with _training_sdpa_context(training_sdpa_backend), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                pg_metrics = conditional_group_backward(
                    core, tokenizer, pg_row, audit_gradients=not pg_audited,
                    lambda_pg=pg_config["lambda_pg"], samples=pg_config["samples"])
            pg_audited |= pg_metrics["pg_first_mixed_audited"]
            pg_groups_seen += 1
            pg_record = {"type": "recurrent_pg", "update": next_update,
                         "tokens_seen": next_tokens, "pg_groups_seen": pg_groups_seen,
                         **pg_metrics}
            write_jsonl(args.run_dir / "metrics.jsonl", pg_record)
            if tracker is not None:
                tracker.log_record("recurrent_pg", pg_record)
        elif pg_control_first_only:
            pg_skipped_updates += 1
        measure_component_norms = update == 0 or (
            (update + 1) % int(training_config["log_every_updates"]) == 0
        ) or admission_update or (
            args.qualification_updates == 1 and update == initial_update
        )
        theta_grad_norm = (
            _parameter_grad_norm(core.model.parameters())
            if measure_component_norms
            else None
        )
        psi_grad_norm = (
            _parameter_grad_norm(core.dynamics.parameters())
            if measure_component_norms
            else None
        )
        if theta_grad_norm is not None and (
            not math.isfinite(theta_grad_norm) or not math.isfinite(psi_grad_norm)
        ):
            raise RuntimeError(
                f"non-finite gradient norm theta={theta_grad_norm} psi={psi_grad_norm}"
            )
        if (admission_update or update == 0) and (
            theta_grad_norm is None
            or psi_grad_norm is None
            or theta_grad_norm <= 0.0
            or psi_grad_norm <= 0.0
        ):
            raise RuntimeError(
                "joint d=4 admission requires nonzero theta and psi gradients"
            )
        if hybrid_correction is not None:
            theta_total_grad_norm = torch.nn.utils.clip_grad_norm_(
                core.model.parameters(), float(training_config["gradient_clip"])
            )
            psi_total_grad_norm = torch.nn.utils.clip_grad_norm_(
                core.dynamics.parameters(), float(training_config["gradient_clip"])
            )
            total_grad_norm = torch.sqrt(
                theta_total_grad_norm.float().square()
                + psi_total_grad_norm.float().square()
            )
        else:
            theta_total_grad_norm = None
            psi_total_grad_norm = None
            total_grad_norm = torch.nn.utils.clip_grad_norm_(
                core.parameters(), float(training_config["gradient_clip"])
            )
        if not bool(torch.isfinite(total_grad_norm)):
            raise RuntimeError(f"non-finite total gradient norm {total_grad_norm}")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if pending_position_counts is not None:
            pending_position_counts += _position_count_vector(
                global_counts,
                horizon=horizon,
                global_repair_counts=global_repair_counts,
            )
        if materialize_step_metrics:
            if global_values is None:
                raise RuntimeError("reporting step has no global component metrics")
            if rank == 0:
                last_global_metrics = _materialize_global_metrics(
                    global_values=global_values,
                    horizon=horizon,
                    lambda_h=loss_config.lambda_h,
                    lambda_kl=loss_config.lambda_kl,
                    correction_loss_weight=correction_loss_weight,
                    global_repair_values=global_repair_values,
                )
                if pending_position_counts is None:
                    raise RuntimeError("rank zero has no pending position counts")
                _flush_position_totals(
                    position_totals=position_totals,
                    position_total_names=position_total_names,
                    pending_position_counts=pending_position_counts,
                )
        tokens_seen = next_tokens
        update = next_update
        if qualification_complete and hybrid_correction is not None:
            qualification_optimizer_state = _validate_loaded_optimizer_state(
                optimizer=optimizer,
                model=core.model,
                dynamics=core.dynamics,
                expected_group_parameter_counts=(
                    optimizer_import["group_parameter_counts"]
                    if optimizer_import is not None
                    else None
                ),
                expected_theta_steps=[update],
                expected_psi_steps=[update],
            )

        if rank == 0 and (
            update == 1
            or update % int(training_config["log_every_updates"]) == 0
            or qualification_complete
            or admission_update
        ):
            torch.cuda.synchronize(device)
            update_seconds = time.time() - update_started
            elapsed = time.time() - started
            record = {
                "type": "train",
                "update": update,
                "tokens_seen": tokens_seen,
                "token_budget": token_budget,
                "update_seconds": update_seconds,
                "update_tokens_per_second": (tokens_seen - update_initial_tokens) / max(update_seconds, 1e-6),
                "tokens_per_second": (tokens_seen - initial_tokens)
                / max(elapsed, 1e-6),
                **last_global_metrics,
                **({"pg_groups_seen": pg_groups_seen,
                    "pg_auxiliary_skipped_updates": pg_skipped_updates}
                   if pg_config is not None else {}),
                "theta_grad_norm": theta_grad_norm,
                "psi_grad_norm": psi_grad_norm,
                "total_grad_norm_pre_clip": float(total_grad_norm.item()),
                "theta_total_grad_norm_pre_clip": (
                    float(theta_total_grad_norm.item())
                    if theta_total_grad_norm is not None
                    else None
                ),
                "psi_total_grad_norm_pre_clip": (
                    float(psi_total_grad_norm.item())
                    if psi_total_grad_norm is not None
                    else None
                ),
                "theta_lr": optimizer.param_groups[0]["lr"],
                "psi_lr": optimizer.param_groups[2]["lr"],
                "max_cuda_memory_gib": torch.cuda.max_memory_allocated(device)
                / 2**30,
                "position_totals": dict(position_totals),
                "corpus_tokens_seen": tokens_seen,
                "assistant_tokens_seen": position_totals["ce"],
                "training_pool_unique_problems": prepared_manifest["splits"]["train"]["problems"],
                "dataset_epochs": (tokens_seen - sampler_origins["tokens_seen"])
                / float(prepared_manifest["splits"]["train"]["tokens"]),
                "continuation_tokens": tokens_seen - initial_tokens,
                "admission_update": admission_update,
                "qualification_optimizer_state": qualification_optimizer_state,
                "position_deltas": {
                    name: position_totals[name] - initial_position_totals[name]
                    for name in position_totals
                },
            }
            write_jsonl(args.run_dir / "metrics.jsonl", record)
            if tracker is not None:
                tracker.log_record("train", record)
            print(json.dumps(record, sort_keys=True), flush=True)

        if qualification_complete:
            break

        if recursive_eval_enabled and (
            tokens_seen >= int(next_eval) or tokens_seen >= token_budget
        ):
            _barrier(world_size)
            if rank == 0:
                with _training_sdpa_context(
                    training_sdpa_backend
                ), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    recursive = _quick_recursive_eval(
                        model=core.model,
                        dynamics=core.dynamics,
                        holdout=holdout,
                        collator=collator,
                        device=device,
                        depths=[
                            int(value) for value in config["recursive_gate"]["depths"]
                        ],
                        examples=int(training_config["recursive_eval_examples"]),
                    )
                record = {
                    "type": "recursive_eval",
                    "update": update,
                    "tokens_seen": tokens_seen,
                    **recursive,
                }
                write_jsonl(args.run_dir / "metrics.jsonl", record)
                if tracker is not None:
                    tracker.log_record("recursive", record)
                print(json.dumps(record, sort_keys=True), flush=True)
            _barrier(world_size)
            while int(next_eval) <= tokens_seen:
                next_eval = int(next_eval) + eval_interval

        while checkpoint_thresholds and tokens_seen >= checkpoint_thresholds[0]:
            threshold = checkpoint_thresholds.pop(0)
            _barrier(world_size)
            if rank == 0:
                checkpoint_dir = args.run_dir / f"checkpoint-{threshold:09d}-tokens"
                include_optimizer = threshold in optimizer_checkpoint_tokens
                rng_state = capture_rng_state() if include_optimizer else None
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    model=core.model,
                    dynamics=core.dynamics,
                    tokenizer=tokenizer,
                    optimizer=optimizer if include_optimizer else None,
                    train_state=checkpoint_state(threshold, include_optimizer),
                    rng_state=rng_state,
                )
                if rng_state is not None:
                    restore_rng_state(rng_state)
                metadata = {
                    "tokens_seen": tokens_seen,
                    "requested_checkpoint_tokens": threshold,
                    "continuation_tokens": tokens_seen - initial_tokens,
                    "update": update,
                    "source_code_commit": source_commit,
                    "objective_horizon": horizon,
                    "deep_detached_repair_horizon": correction_horizon,
                    "theta_trainable": True,
                    "psi_trainable": True,
                }
                aliases = [
                    f"tokens-{tokens_seen}",
                    f"continuation-{tokens_seen - initial_tokens}",
                    f"requested-{threshold}",
                ]
                if threshold in artifact_checkpoint_tokens:
                    if tracker is None:
                        raise RuntimeError("registered W&B checkpoint has no tracker")
                    tracker.log_checkpoint_artifact(
                        checkpoint_dir,
                        artifact_name=f"{args.wandb_run_id}-checkpoint",
                        aliases=aliases,
                        metadata=metadata,
                    )
                if threshold in receipt_artifact_tokens:
                    if tracker is None:
                        raise RuntimeError("registered W&B receipt has no tracker")
                    tracker.log_checkpoint_receipt_artifact(
                        checkpoint_dir,
                        artifact_name=f"{args.wandb_run_id}-checkpoint-receipt",
                        aliases=aliases,
                        metadata={
                            **metadata,
                            "full_checkpoint_location": str(checkpoint_dir),
                            "full_checkpoint_upload": "deferred",
                        },
                    )
            last_checkpoint_tokens = tokens_seen
            _barrier(world_size)

    if last_checkpoint_tokens != tokens_seen:
        _barrier(world_size)
        if rank == 0:
            rng_state = capture_rng_state()
            save_checkpoint(
                checkpoint_dir=args.run_dir
                / f"checkpoint-final-{tokens_seen:09d}-tokens",
                model=core.model,
                dynamics=core.dynamics,
                tokenizer=tokenizer,
                optimizer=optimizer,
                train_state=checkpoint_state(
                    tokens_seen if args.qualification_updates else token_budget,
                    True,
                ),
                rng_state=rng_state,
            )
            restore_rng_state(rng_state)
        last_checkpoint_tokens = tokens_seen
        _barrier(world_size)

    if rank == 0:
        completed = _run_completed(
            tokens_seen=tokens_seen,
            token_budget=token_budget,
            qualification_updates=args.qualification_updates,
        )
        pg_signal = pg_config is None or pg_config["lambda_pg"] == 0 or pg_audited
        result_state = "observed" if completed and pg_signal else "diagnostic"
        summary = {
            "result_state": result_state,
            "completed": completed,
            "qualification_updates": args.qualification_updates,
            "updates": update,
            "tokens_seen": tokens_seen,
            "token_budget": token_budget,
            "elapsed_seconds": time.time() - started,
            "world_size": world_size,
            "source_code_commit": source_commit,
            "position_totals": position_totals,
            "last_train_metrics": last_global_metrics,
            **({"pg_groups_seen": pg_groups_seen, "pg_first_mixed_audited": pg_audited,
                "pg_auxiliary_skipped_updates": pg_skipped_updates,
                "recurrent_policy_gradient": pg_config} if pg_config is not None else {}),
            "prepared_data_manifest": prepared_manifest,
            "dataset_epochs": (tokens_seen - sampler_origins["tokens_seen"])
            / float(prepared_manifest["splits"]["train"]["tokens"]),
            "wandb_run_id": args.wandb_run_id,
            "resume_kind": resume_kind,
            "objective": {
                "horizon": horizon,
                "teacher_forced_corpus_tokens": True,
                "theta_trainable": True,
                "psi_trainable": True,
                "ce_retained": True,
                "deep_detached_repair": (
                    {
                        "depths": list(range(horizon + 1, correction_horizon + 1)),
                        "loss": "smooth_l1",
                        "loss_weight": correction_loss_weight,
                        "same_exact_forward": True,
                        "theta_gradient": False,
                        "recurrent_input_detached_each_step": True,
                    }
                    if correction_horizon is not None
                    else None
                ),
            },
            "optimizer_state_policy": _optimizer_state_policy(
                config, hybrid_correction=hybrid_correction is not None
            ),
            "optimizer_import": optimizer_import,
            "qualification_optimizer_state": qualification_optimizer_state,
            "continuation": {
                "parent_tokens": initial_tokens,
                "tokens": tokens_seen - initial_tokens,
                "updates": update - initial_update,
                "position_deltas": {
                    name: position_totals[name] - initial_position_totals[name]
                    for name in position_totals
                },
            },
        }
        (args.run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        status_objective = (
            f"joint d4 plus conditional recurrent PG suffix{pg_config['suffix_tokens']} lambda={pg_config['lambda_pg']}"
            if pg_config is not None
            else
            f"joint d4 anchor-noise pilot relative={loss_config.anchor_noise_relative}"
            if config.get("continuation", {}).get("protocol") == "optimizer_state_preserving_joint_d4_anchor_noise_pilot_v1"
            else
            "joint d=4 plus deep detached repair"
            if hybrid_correction is not None
            else "original joint d=4"
        )
        requested_continuation_tokens = max(token_budget - initial_tokens, 0)
        write_status(
            args.run_dir,
            result_state=result_state,
            passed=(
                [f"{status_objective} completed requested rung"]
                if completed
                else [
                    f"one production-path {status_objective} optimizer update completed"
                ]
            ),
            failed_or_missing=[
                *(
                    []
                    if completed
                    else [
                        f"{requested_continuation_tokens:,}-token decision continuation"
                    ]
                ),
                "full recursive-state gate",
                "latent-search gate",
                "matched CE-only attribution",
            ],
            main_metric=(
                f"trained {tokens_seen - initial_tokens:,} continuation tokens"
                if completed
                else f"qualified exactly one {status_objective} theta+psi update"
            ),
            baseline=(f"sealed {config['continuation']['parent']['tokens_seen']:,}-token completed Countdown SFT parent; same corpus and sampler"
                      if pg_config is not None
                      else f"sealed {config['continuation']['parent']['tokens_seen']:,}-token joint d4 parent; new registered corpus"
                      if config.get("continuation", {}).get("protocol") == "optimizer_state_preserving_joint_d4_fresh_data_pilot_v1"
                      else "sealed 800,007,805-token joint theta+psi d4 parent and matched clean continuation"
                      if config.get("continuation", {}).get("protocol") == "optimizer_state_preserving_joint_d4_anchor_noise_pilot_v1"
                      else "sealed 400,004,379-token joint theta+psi d=16 endpoint"),
            caveats=(
                ["recursive evaluation intentionally runs out of process"]
                if not recursive_eval_enabled
                else ["quick recursive metrics are diagnostic"]
            ),
            next_target=(
                "run the locked recursive-state evaluation"
                if completed
                else (
                    "restart the "
                    f"{requested_continuation_tokens:,}-token decision run from the sealed parent"
                )
            ),
        )
        if tracker is not None:
            if world_size == 1:
                # Full model/Adam/RNG files are already complete. Artifact sync
                # only reads files; let the next workload use this GPU meanwhile.
                optimizer.zero_grad(set_to_none=True)
                optimizer.state.clear()
                core.to("cpu")
                torch.cuda.empty_cache()
                print(json.dumps({"type": "gpu_released_for_artifact_sync", "tokens_seen": tokens_seen}), flush=True)
            tracker.finish(summary_namespace="final_train", summary=summary)
    _barrier(world_size)
    if world_size > 1:
        dist.destroy_process_group()


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--token-budget", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-config", type=Path)
    parser.add_argument("--parent-admission-data", type=Path)
    parser.add_argument("--wandb-run-id")
    parser.add_argument("--qualification-updates", type=int, default=0)
    args = parser.parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()
