"""No-draft-KV NextLat worker for pinned SGLang V2.

The worker uses the built-in no-draft-KV scheduler allocation path, DFlash's
linear ragged target-verify input, and SGLang's canonical KV mover.  It does
not own, mirror, or roll back a second KV allocator.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F

from sglang.srt.speculative.base_spec_worker import BaseSpecWorker

from .admission import admit_sglang_source, admit_target_checkpoint
from .beam_tree import (
    TREE_BEAM_WIDTH,
    TREE_HORIZON,
    TREE_NODE_BUDGET,
    pack_top2_prefix_tree,
    reference_greedy_tree_acceptance,
    recurrent_beam_lines,
)
from .checkpoint import load_dynamics_sidecar, read_sidecar_metadata
from .confidence import CONFIDENCE_MAX_DRAFT_TOKENS, top1_log_probability
from .config import top2_tree_verify_enabled, validate_server_args
from .constants import (
    HOT_VOCAB_SIZE,
    MAX_DRAFT_TOKENS,
    QWEN3_1_7B_TOKENIZER_VOCAB_SIZE,
)
from .decoder import (
    NextLatProposal,
    finish_sampled_verification,
    mask_embedding_only_logits,
    propose_greedy,
    propose_temperature_sampled,
    scatter_hot_probabilities,
    verify_greedy,
)
from .diagnostic_precision import near_tie_lm_head_probe
from .diagonal_decode_verify import (
    assert_captured_runner as assert_diagonal_captured_runner,
    diagonal_decode_max_rows,
    diagonal_decode_verify_enabled,
    diagonalize_tree_verify_forward_batch,
    diagonalize_verify_forward_batch,
    fixed_b1_variable_verify_enabled,
    fixed_b32_variable_verify_enabled,
    replay_snapshot as diagonal_replay_snapshot,
    target_bf16_logits_buffer_enabled,
)
from .fixed_b8_m8_verify import (
    FixedB8M8VerifyGeometry,
    fixed_b1_m4_verify_enabled,
    fixed_b32_m4_verify_enabled,
    fixed_b8_m8_verify_enabled,
    fixed_m16_verify_enabled,
)
from .native_ragged_verify import (
    BATCH_SIZE as NATIVE_RAGGED_BATCH_SIZE,
    DRAFT_TOKENS as NATIVE_RAGGED_DRAFT_TOKENS,
    VERIFY_ROWS as NATIVE_RAGGED_VERIFY_ROWS,
    assert_native_captured_runner,
    native_ragged_verify_enabled,
    native_replay_snapshot,
)
from .top2_tree_verify_graph import (
    assert_tree_captured_runner,
    tree_replay_snapshot,
)


_LOW_MARGIN_LOGIT_GAP = 0.125
_DRAFT_CUDA_GRAPH_ENV = "NEXTLAT_DRAFT_CUDA_GRAPH"
_DRAFT_FP8_ENV = "NEXTLAT_DRAFT_FP8"
_B1_FP8_SLIM_ENV = "NEXTLAT_B1_FP8_SLIM"
_ZERO_COPY_LINEAR_KV_COMMIT_ENV = "NEXTLAT_ZERO_COPY_LINEAR_KV_COMMIT"
_RAW_ROLLOUT_TOKENS = 128


def _has_decode_cuda_graph_runner(model_runner: Any) -> bool:
    """Return whether target decode graphs exist without assuming the attribute."""

    return getattr(model_runner, "decode_cuda_graph_runner", None) is not None


def _decode_graph_requested(server_args: Any) -> bool:
    """Read the resolved decode-graph mode before SGLang constructs its runner."""

    graph_config = getattr(server_args, "cuda_graph_config", None)
    decode_config = (
        graph_config.get("decode")
        if isinstance(graph_config, dict)
        else getattr(graph_config, "decode", None)
    )
    backend = (
        decode_config.get("backend")
        if isinstance(decode_config, dict)
        else getattr(decode_config, "backend", None)
    )
    if backend is None:
        backend = getattr(server_args, "cuda_graph_backend_decode", None)
    backend = getattr(backend, "value", backend)
    return bool(
        isinstance(backend, str)
        and backend.strip().lower() == "full"
        and getattr(server_args, "disable_cuda_graph", False) is not True
    )


@dataclass
class NextLatDraftInput:
    """Exact recurrent seed carried from one scheduler iteration to the next."""

    bonus_tokens: torch.Tensor
    hidden_states: torch.Tensor
    new_seq_lens: torch.Tensor
    future_indices: Optional[torch.Tensor] = None
    # Generic FutureMap relay fields. NextLat does not consume EAGLE top-k or
    # DSA state, but the pinned generic payload constructor reads the common
    # SpecInput surface before dispatching on need_topk().
    topk_p: Optional[torch.Tensor] = None
    topk_index: Optional[torch.Tensor] = None
    draft_probs: Optional[torch.Tensor] = None
    dsa_topk_indices: Optional[torch.Tensor] = None
    future_dsa_topk_indices_available: bool = False

    # Scheduler/ForwardBatch-compatible shape defaults. This object is replaced
    # by a DFlashVerifyInput before any model forward consumes ``spec_info``.
    num_tokens_per_req: int = 1
    num_tokens_for_logprob_per_req: int = 1
    ragged_verify_layout: None = None

    @classmethod
    def empty(cls, *, device: torch.device, hidden_size: int) -> "NextLatDraftInput":
        return cls(
            bonus_tokens=torch.empty((0,), dtype=torch.int64, device=device),
            hidden_states=torch.empty(
                (0, hidden_size), dtype=torch.bfloat16, device=device
            ),
            new_seq_lens=torch.empty((0,), dtype=torch.int64, device=device),
        )

    def filter_batch(self, new_indices, new_indices_cpu=None) -> None:
        del new_indices_cpu
        if self.future_indices is not None:
            self.future_indices = self.future_indices[new_indices]
            return
        self.bonus_tokens = self.bonus_tokens[new_indices]
        self.hidden_states = self.hidden_states[new_indices]
        self.new_seq_lens = self.new_seq_lens[new_indices]

    def merge_batch(self, other: "NextLatDraftInput") -> None:
        if not isinstance(other, NextLatDraftInput):
            raise TypeError("NEXTLAT can only merge another NextLatDraftInput")
        if self.future_indices is not None:
            if other.future_indices is None:
                raise ValueError("NEXTLAT future-index merge is asymmetric")
            self.future_indices = torch.cat((self.future_indices, other.future_indices))
            return
        if other.future_indices is not None:
            raise ValueError("NEXTLAT future-index merge is asymmetric")
        self.bonus_tokens = torch.cat((self.bonus_tokens, other.bonus_tokens))
        self.hidden_states = torch.cat((self.hidden_states, other.hidden_states))
        self.new_seq_lens = torch.cat((self.new_seq_lens, other.new_seq_lens))


class NextLatWorker(BaseSpecWorker):
    """Greedy NextLat decoder with a configured max window up to 32 drafts."""

    def __init__(
        self,
        server_args,
        gpu_id: int,
        ps,
        nccl_port: int,
        target_worker,
    ) -> None:
        super().__init__()
        del nccl_port
        validate_server_args(server_args)
        self._top2_tree_verify = top2_tree_verify_enabled()

        from sglang.srt.runtime_context import get_device, get_spec

        started = time.perf_counter()
        self.server_args = server_args
        self.gpu_id = gpu_id
        self.ps = ps
        self._target_worker = target_worker
        self._draft_worker = None
        self.model_runner = target_worker.model_runner
        self.device = torch.device(get_device().device)
        if self.device.type != "cuda":
            raise RuntimeError(f"NEXTLAT requires CUDA, got {self.device}")

        spec = get_spec()
        self.max_draft_tokens = int(spec.speculative_num_steps)
        self.verify_width = int(spec.speculative_num_draft_tokens)
        self._use_rejection_sampling = bool(
            spec.speculative_use_rejection_sampling
        )
        # Keep the BaseSpecWorker ecosystem's conventional attribute names;
        # these are maxima, while each request carries its own proposal length.
        self.speculative_num_steps = self.max_draft_tokens
        self.speculative_num_draft_tokens = self.verify_width
        if not 1 <= self.max_draft_tokens <= MAX_DRAFT_TOKENS:
            raise RuntimeError("NEXTLAT runtime context drifted after argument admission")
        if (
            not self._top2_tree_verify
            and self.verify_width != self.max_draft_tokens + 1
        ):
            raise RuntimeError("NEXTLAT verify width drifted after argument admission")
        variable_b1_verify = fixed_b1_variable_verify_enabled()
        variable_b32_verify = fixed_b32_variable_verify_enabled()
        draft_graph_value = os.environ.get(_DRAFT_CUDA_GRAPH_ENV)
        if draft_graph_value not in (None, "1"):
            raise RuntimeError(f"{_DRAFT_CUDA_GRAPH_ENV} must be exactly '1'")
        self._draft_cuda_graph_enabled = draft_graph_value == "1"
        if self._use_rejection_sampling and self._draft_cuda_graph_enabled:
            raise RuntimeError(
                "NEXTLAT sampled proposals currently require eager draft execution "
                "because q_hot must be retained for rejection sampling"
            )
        if self._draft_cuda_graph_enabled and not (
            self.max_draft_tokens in (4, 8, 16)
            or variable_b1_verify
            or (variable_b32_verify and 1 <= self.max_draft_tokens <= 16)
        ):
            raise RuntimeError(
                "NEXTLAT draft CUDA graph supports M4/M8/M16 or variable "
                "B1/B32 M1-M16"
            )
        self._draft_graph_batch_size = 1 if self._top2_tree_verify else 8
        if self._draft_cuda_graph_enabled and diagonal_decode_verify_enabled():
            verify_rows = diagonal_decode_max_rows()
            if verify_rows % self.verify_width != 0:
                raise RuntimeError(
                    "NEXTLAT diagonal verifier rows must divide by the verify width"
                )
            self._draft_graph_batch_size = verify_rows // self.verify_width
            if variable_b1_verify and self._draft_graph_batch_size != 1:
                raise RuntimeError(
                    "NEXTLAT variable B1 verifier geometry requires exactly B1"
                )
            if variable_b32_verify and self._draft_graph_batch_size != 32:
                raise RuntimeError(
                    "NEXTLAT variable verifier geometry requires exactly B32"
                )
        draft_fp8_value = os.environ.get(_DRAFT_FP8_ENV)
        if draft_fp8_value not in (None, "1"):
            raise RuntimeError(f"{_DRAFT_FP8_ENV} must be exactly '1'")
        self._draft_fp8_enabled = draft_fp8_value == "1"
        if self._draft_fp8_enabled and not self._draft_cuda_graph_enabled:
            raise RuntimeError(
                "NEXTLAT_DRAFT_FP8 requires NEXTLAT_DRAFT_CUDA_GRAPH=1"
            )
        slim_value = os.environ.get(_B1_FP8_SLIM_ENV)
        if slim_value not in (None, "1"):
            raise RuntimeError(f"{_B1_FP8_SLIM_ENV} must be exactly '1'")
        self._b1_fp8_slim = slim_value == "1"
        if self._b1_fp8_slim and not (
            self._draft_fp8_enabled
            and variable_b1_verify
            and self._draft_graph_batch_size == 1
            and self.max_draft_tokens == 8
            and not self._top2_tree_verify
            and not self._use_rejection_sampling
            and not os.environ.get("NEXTLAT_CONFIDENCE_TRACE")
            and not os.environ.get("NEXTLAT_PIPELINE_DRAFT_DEVICE")
        ):
            raise RuntimeError(
                "NEXTLAT_B1_FP8_SLIM requires greedy linear B1/M8 FP8 graphs "
                "and excludes BF16 confidence tracing or a separate draft device"
            )
        self._draft_cuda_graph: torch.cuda.CUDAGraph | None = None
        self._draft_graph_state: torch.Tensor | None = None
        self._draft_graph_anchor: torch.Tensor | None = None
        self._draft_graph_tokens: torch.Tensor | None = None
        self._draft_graph_log_q: torch.Tensor | None = None

        sidecar_path = server_args.speculative_draft_model_path
        metadata = read_sidecar_metadata(sidecar_path)
        self.contract = metadata.contract
        self.source_admission = admit_sglang_source()
        self.target_admission = admit_target_checkpoint(
            server_args.model_path,
            expected_manifest_sha256=metadata.target_checkpoint_sha256,
            contract=metadata.contract,
        )
        loaded = load_dynamics_sidecar(sidecar_path, device=self.device)
        self.sidecar_metadata = loaded.metadata
        # Source artifacts are FP32 by contract; the one admitted serving tier
        # makes an explicit BF16 runtime copy, recorded in its receipt.
        self.dynamics = loaded.dynamics.to(dtype=torch.bfloat16)
        self._preloaded_weights_bytes = sum(
            parameter.numel() * parameter.element_size()
            for parameter in self.dynamics.parameters()
        )
        self._weight_load_time = time.perf_counter() - started

        self.target_model = self.model_runner.model
        if type(self.target_model).__name__ != "Qwen3ForCausalLM":
            raise RuntimeError(
                "NEXTLAT target runtime must instantiate Qwen3ForCausalLM, got "
                f"{type(self.target_model).__name__}"
            )
        for attribute in ("get_input_embeddings", "logits_processor", "lm_head"):
            if not hasattr(self.target_model, attribute):
                raise RuntimeError(f"NEXTLAT target lacks required API {attribute}")
        embeddings = self.target_model.get_input_embeddings()
        if tuple(embeddings.weight.shape) != (
            self.contract.vocab_size,
            self.contract.hidden_size,
        ):
            raise RuntimeError(
                "NEXTLAT target embedding shape differs from the admitted contract"
            )
        hf_config = self.model_runner.model_config.hf_config
        if getattr(hf_config, "tie_word_embeddings", None) is not True:
            raise RuntimeError("NEXTLAT hot head requires Qwen tied embeddings")
        from sglang.srt.speculative.spec_utils import load_token_map

        hot_token_id = load_token_map(server_args.speculative_token_map)
        if (
            hot_token_id.dtype != torch.int64
            or tuple(hot_token_id.shape) != (HOT_VOCAB_SIZE,)
            or int(torch.unique(hot_token_id).numel()) != HOT_VOCAB_SIZE
            or int(hot_token_id.min()) < 0
            or int(hot_token_id.max()) >= QWEN3_1_7B_TOKENIZER_VOCAB_SIZE
        ):
            raise RuntimeError(
                "NEXTLAT speculative token map must contain 32,768 unique "
                "Qwen tokenizer token ids"
            )
        self.hot_token_id = hot_token_id.to(self.device, non_blocking=True)
        full_lm_head_weight = self.target_model.lm_head.weight
        if tuple(full_lm_head_weight.shape) != (
            self.contract.vocab_size,
            self.contract.hidden_size,
        ):
            raise RuntimeError("NEXTLAT tied lm_head shape differs from Qwen contract")
        # One serving-lifetime gather.  Every recurrent step thereafter projects
        # only the FR-Spec hot vocabulary and maps the reduced winner back to a
        # canonical Qwen token id.
        self.hot_lm_head_weight = full_lm_head_weight.index_select(
            0, self.hot_token_id
        ).contiguous()
        self.draft_vocab_size = HOT_VOCAB_SIZE
        self._preloaded_weights_bytes += (
            self.hot_lm_head_weight.numel() * self.hot_lm_head_weight.element_size()
        )
        self._draft_fp8_quant = None
        self._fp8_mlp0_weight = self._fp8_mlp0_scale = None
        self._fp8_mlp2_weight = self._fp8_mlp2_scale = None
        self._fp8_mlp4_weight = self._fp8_mlp4_scale = None
        self._fp8_head_weight = self._fp8_head_scale = None
        if self._draft_fp8_enabled:
            self._initialize_draft_fp8_weights()
        if self._b1_fp8_slim:
            # Every proposal, including a bounded completion tail, will use
            # the FP8 graph. Keep the small norm on GPU, but stop retaining
            # unused BF16 copies of the matrices and gathered vocabulary head.
            released_bytes = sum(
                parameter.numel() * parameter.element_size()
                for parameter in self.dynamics.mlp.parameters()
            ) + self.hot_lm_head_weight.numel() * self.hot_lm_head_weight.element_size()
            self.dynamics.mlp.to(device="cpu")
            self.hot_lm_head_weight = None
            self._preloaded_weights_bytes -= released_bytes
            torch.cuda.empty_cache()
        self._weight_load_time = time.perf_counter() - started

        # Pools are injected by the scheduler after target-pool construction.
        self.req_to_token_pool = None
        self.token_to_kv_pool_allocator = None
        self._diagnostic_trace_path: Path | None = None
        self._diagnostic_request_ids: tuple[str, ...] | None = None
        trace_value = os.environ.get("NEXTLAT_DIAGNOSTIC_TRACE")
        phase_trace_value = os.environ.get("NEXTLAT_PHASE_TRACE")
        confidence_trace_value = os.environ.get("NEXTLAT_CONFIDENCE_TRACE")
        if self._use_rejection_sampling and confidence_trace_value:
            raise RuntimeError(
                "NEXTLAT_CONFIDENCE_TRACE currently measures greedy top-1 proposals"
            )
        if trace_value and phase_trace_value:
            raise RuntimeError(
                "NEXTLAT_DIAGNOSTIC_TRACE and NEXTLAT_PHASE_TRACE cannot be enabled "
                "together because diagnostic logits would contaminate phase timing"
            )
        if trace_value and confidence_trace_value:
            raise RuntimeError(
                "NEXTLAT_DIAGNOSTIC_TRACE and NEXTLAT_CONFIDENCE_TRACE cannot be "
                "enabled together"
            )
        if trace_value:
            request_id = os.environ.get("NEXTLAT_DIAGNOSTIC_RID")
            request_ids_json = os.environ.get("NEXTLAT_DIAGNOSTIC_RIDS")
            if request_id and request_ids_json:
                raise RuntimeError(
                    "NEXTLAT_DIAGNOSTIC_RID and NEXTLAT_DIAGNOSTIC_RIDS are "
                    "mutually exclusive"
                )
            if request_ids_json:
                try:
                    parsed_request_ids = json.loads(request_ids_json)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        "NEXTLAT_DIAGNOSTIC_RIDS must be a JSON string array"
                    ) from exc
                if (
                    not isinstance(parsed_request_ids, list)
                    or not parsed_request_ids
                    or any(
                        not isinstance(value, str) or not value
                        for value in parsed_request_ids
                    )
                    or len(set(parsed_request_ids)) != len(parsed_request_ids)
                ):
                    raise RuntimeError(
                        "NEXTLAT_DIAGNOSTIC_RIDS must contain unique non-empty strings"
                    )
                diagnostic_request_ids = tuple(parsed_request_ids)
            elif request_id:
                diagnostic_request_ids = (request_id,)
            else:
                raise RuntimeError(
                    "NEXTLAT_DIAGNOSTIC_TRACE requires NEXTLAT_DIAGNOSTIC_RID or "
                    "NEXTLAT_DIAGNOSTIC_RIDS"
                )
            trace_arg = Path(trace_value).expanduser()
            if not trace_arg.is_absolute() or not trace_arg.name:
                raise RuntimeError("NEXTLAT_DIAGNOSTIC_TRACE must be an absolute new file")
            trace_parent = trace_arg.parent.resolve()
            if not trace_parent.is_dir() or trace_arg.exists() or trace_arg.is_symlink():
                raise RuntimeError(
                    "NEXTLAT_DIAGNOSTIC_TRACE parent must exist and destination must be new"
                )
            self._diagnostic_trace_path = trace_parent / trace_arg.name
            self._diagnostic_request_ids = diagnostic_request_ids
            self._write_diagnostic_trace(
                {
                    "event": "header",
                    "request_ids": list(diagnostic_request_ids),
                    "max_draft_tokens": self.max_draft_tokens,
                    "verify_width": self.verify_width,
                },
                exclusive=True,
            )
        self._raw_rollout_trace_path: Path | None = None
        self._raw_rollout_request_id: str | None = None
        raw_trace_value = os.environ.get("NEXTLAT_RAW_ROLLOUT_TRACE")
        raw_request_id = os.environ.get("NEXTLAT_RAW_ROLLOUT_RID")
        if bool(raw_trace_value) != bool(raw_request_id):
            raise RuntimeError(
                "NEXTLAT_RAW_ROLLOUT_TRACE and NEXTLAT_RAW_ROLLOUT_RID must be set together"
            )
        if raw_trace_value and raw_request_id:
            if self._diagnostic_request_ids and raw_request_id in self._diagnostic_request_ids:
                raise RuntimeError("raw rollout RID must differ from diagnostic trace RIDs")
            raw_trace_arg = Path(raw_trace_value).expanduser()
            if not raw_trace_arg.is_absolute() or not raw_trace_arg.name:
                raise RuntimeError(
                    "NEXTLAT_RAW_ROLLOUT_TRACE must be an absolute new file"
                )
            raw_trace_parent = raw_trace_arg.parent.resolve()
            if (
                not raw_trace_parent.is_dir()
                or raw_trace_arg.exists()
                or raw_trace_arg.is_symlink()
            ):
                raise RuntimeError(
                    "raw rollout trace parent must exist and destination must be new"
                )
            self._raw_rollout_trace_path = raw_trace_parent / raw_trace_arg.name
            self._raw_rollout_request_id = raw_request_id
            self._write_raw_rollout_trace(
                {
                    "event": "header",
                    "request_id": raw_request_id,
                    "output_tokens": _RAW_ROLLOUT_TOKENS,
                    "semantics": "one exact anchor followed by unverified recurrent top1",
                },
                exclusive=True,
            )
        self._confidence_trace_path: Path | None = None
        self._confidence_request_ids: frozenset[str] | None = None
        if confidence_trace_value:
            if self.max_draft_tokens != CONFIDENCE_MAX_DRAFT_TOKENS:
                raise RuntimeError(
                    "NEXTLAT_CONFIDENCE_TRACE is registered only for fixed M=16"
                )
            request_ids_json = os.environ.get("NEXTLAT_CONFIDENCE_RIDS")
            if not request_ids_json:
                raise RuntimeError(
                    "NEXTLAT_CONFIDENCE_TRACE requires NEXTLAT_CONFIDENCE_RIDS"
                )
            try:
                parsed_request_ids = json.loads(request_ids_json)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "NEXTLAT_CONFIDENCE_RIDS must be a JSON string array"
                ) from exc
            if (
                not isinstance(parsed_request_ids, list)
                or not parsed_request_ids
                or any(
                    not isinstance(value, str) or not value
                    for value in parsed_request_ids
                )
                or len(set(parsed_request_ids)) != len(parsed_request_ids)
            ):
                raise RuntimeError(
                    "NEXTLAT_CONFIDENCE_RIDS must contain unique non-empty strings"
                )
            confidence_trace_arg = Path(confidence_trace_value).expanduser()
            if not confidence_trace_arg.is_absolute() or not confidence_trace_arg.name:
                raise RuntimeError(
                    "NEXTLAT_CONFIDENCE_TRACE must be an absolute new file"
                )
            confidence_trace_parent = confidence_trace_arg.parent.resolve()
            if (
                not confidence_trace_parent.is_dir()
                or confidence_trace_arg.exists()
                or confidence_trace_arg.is_symlink()
            ):
                raise RuntimeError(
                    "NEXTLAT_CONFIDENCE_TRACE parent must exist and destination must be new"
                )
            self._confidence_trace_path = confidence_trace_parent / confidence_trace_arg.name
            self._confidence_request_ids = frozenset(parsed_request_ids)
            self._write_confidence_trace(
                {
                    "event": "header",
                    "request_ids": parsed_request_ids,
                    "max_draft_tokens": self.max_draft_tokens,
                    "score": "s_k=sum_{j<=k} log p_recurrent(argmax_j)",
                    "trace_write_contract": "header fsync; event append without fsync",
                },
                exclusive=True,
            )
        self._phase_trace_path: Path | None = None
        if phase_trace_value:
            phase_trace_arg = Path(phase_trace_value).expanduser()
            if not phase_trace_arg.is_absolute() or not phase_trace_arg.name:
                raise RuntimeError("NEXTLAT_PHASE_TRACE must be an absolute new file")
            phase_trace_parent = phase_trace_arg.parent.resolve()
            if (
                not phase_trace_parent.is_dir()
                or phase_trace_arg.exists()
                or phase_trace_arg.is_symlink()
            ):
                raise RuntimeError(
                    "NEXTLAT_PHASE_TRACE parent must exist and destination must be new"
                )
            self._phase_trace_path = phase_trace_parent / phase_trace_arg.name
            self._write_phase_trace(
                {
                    "event": "header",
                    "max_draft_tokens": self.max_draft_tokens,
                    "verify_width": self.verify_width,
                    "device": str(self.device),
                    "timing_authority": "torch.cuda.Event on the production stream",
                },
                exclusive=True,
            )
        self._diagonal_verify_graph = diagonal_decode_verify_enabled()
        self._native_ragged_verify_graph = native_ragged_verify_enabled()
        if self._diagonal_verify_graph and self._native_ragged_verify_graph:
            raise RuntimeError(
                "diagonal and native ragged verifier graphs are mutually exclusive"
            )
        tree_graph_requested = _decode_graph_requested(self.server_args)
        self._top2_tree_diagonal_verify_graph = bool(
            self._top2_tree_verify
            and self._diagonal_verify_graph
            and tree_graph_requested
        )
        self._top2_tree_verify_graph = bool(
            self._top2_tree_verify
            and not self._diagonal_verify_graph
            and tree_graph_requested
        )
        self._target_verify_graph_diagnostic = bool(
            self._diagonal_verify_graph
            or self._native_ragged_verify_graph
            or self._top2_tree_verify_graph
        )
        self._fixed_dense_m8_verify: FixedB8M8VerifyGeometry | None = None
        self._tree_diagonal_req_pool_rows: list[int] | None = None
        self._tree_diagonal_req_pool_indices: torch.Tensor | None = None
        fixed_m8 = fixed_b8_m8_verify_enabled()
        fixed_b1_m4 = fixed_b1_m4_verify_enabled()
        fixed_b32_m4 = fixed_b32_m4_verify_enabled()
        fixed_m16 = fixed_m16_verify_enabled()
        if sum(
            (
                fixed_m8,
                fixed_b1_m4,
                fixed_b32_m4,
                fixed_m16,
                variable_b1_verify,
                variable_b32_verify,
            )
        ) > 1:
            raise RuntimeError("fixed verifier lanes are mutually exclusive")
        fixed_m4 = fixed_b1_m4 or fixed_b32_m4
        if (
            fixed_m8
            or fixed_m4
            or fixed_m16
            or variable_b1_verify
            or variable_b32_verify
        ):
            if not self._diagonal_verify_graph:
                raise RuntimeError(
                    "fixed dense verifier geometry requires diagonal DECODE"
                )
            verify_rows = diagonal_decode_max_rows()
            if fixed_b1_m4 and verify_rows != 5:
                raise RuntimeError(
                    "fixed B1/M4 verifier geometry requires exactly five rows"
                )
            if fixed_b32_m4 and verify_rows != 160:
                raise RuntimeError(
                    "fixed B32/M4 verifier geometry requires exactly 160 rows"
                )
            if fixed_m16 and verify_rows not in (17, 544):
                raise RuntimeError(
                    "fixed M16 verifier geometry requires exactly 17 or 544 rows"
                )
            if variable_b1_verify and verify_rows != self.verify_width:
                raise RuntimeError(
                    "variable B1 verifier geometry requires exactly M+1 rows"
                )
            if variable_b32_verify and verify_rows != 32 * self.verify_width:
                raise RuntimeError(
                    "variable B32 verifier geometry requires exactly 32 x (M+1) rows"
                )
            expected_drafts = (
                self.max_draft_tokens
                if variable_b1_verify or variable_b32_verify
                else 4
                if fixed_m4
                else 16
                if fixed_m16
                else 8
            )
            if (
                self.max_draft_tokens != expected_drafts
                or self.verify_width != expected_drafts + 1
            ):
                raise RuntimeError(
                    f"fixed dense verifier geometry requires M{expected_drafts}"
                )
            self._fixed_dense_m8_verify = FixedB8M8VerifyGeometry(
                verify_rows=verify_rows,
                draft_tokens=expected_drafts,
            )
        zero_copy_kv_value = os.environ.get(_ZERO_COPY_LINEAR_KV_COMMIT_ENV)
        if zero_copy_kv_value not in (None, "1"):
            raise RuntimeError(
                f"{_ZERO_COPY_LINEAR_KV_COMMIT_ENV} must be exactly '1'"
            )
        self._zero_copy_linear_kv_commit = zero_copy_kv_value == "1"
        if (
            self._zero_copy_linear_kv_commit
            and self._fixed_dense_m8_verify is None
        ):
            raise RuntimeError(
                f"{_ZERO_COPY_LINEAR_KV_COMMIT_ENV}=1 requires the fixed dense "
                "verifier geometry"
            )
        self._target_verify_graph_trace_path: Path | None = None
        self._target_verify_graph_pointer_signature: dict[str, int] | None = None
        self._target_verify_graph_geometry_sha256: str | None = None
        self._target_verify_graph_replay_count = 0
        self._target_verify_graph_tail_eager_count = 0
        graph_trace_value = os.environ.get("NEXTLAT_TARGET_VERIFY_GRAPH_TRACE")
        if graph_trace_value and not self._target_verify_graph_diagnostic:
            raise RuntimeError(
                "NEXTLAT_TARGET_VERIFY_GRAPH_TRACE requires the graph diagnostic"
            )
        if graph_trace_value:
            graph_trace_arg = Path(graph_trace_value).expanduser()
            if not graph_trace_arg.is_absolute() or not graph_trace_arg.name:
                raise RuntimeError(
                    "NEXTLAT_TARGET_VERIFY_GRAPH_TRACE must be an absolute new file"
                )
            graph_trace_parent = graph_trace_arg.parent.resolve()
            if (
                not graph_trace_parent.is_dir()
                or graph_trace_arg.exists()
                or graph_trace_arg.is_symlink()
            ):
                raise RuntimeError(
                    "NEXTLAT_TARGET_VERIFY_GRAPH_TRACE parent must exist and "
                    "destination must be new"
                )
            self._target_verify_graph_trace_path = (
                graph_trace_parent / graph_trace_arg.name
            )
            self._write_target_verify_graph_trace(
                {
                    "event": "header",
                    "scope": (
                        "B1/N14 irregular tree through q_len=1 ordinary DECODE"
                        if self._top2_tree_diagonal_verify_graph
                        else "native B1/N14 irregular-tree TARGET_VERIFY"
                        if self._top2_tree_verify_graph
                        else "native compact-ragged TARGET_VERIFY"
                        if self._native_ragged_verify_graph
                        else (
                            "compact verification through q_len=1 ordinary DECODE; "
                            "TARGET_VERIFY is never executed"
                        )
                    ),
                    "graph_max_rows": (
                        self.verify_width
                        if self._top2_tree_verify_graph
                        else NATIVE_RAGGED_VERIFY_ROWS
                        if self._native_ragged_verify_graph
                        else diagonal_decode_max_rows()
                    ),
                    "configured_max_drafts": self.max_draft_tokens,
                },
                exclusive=True,
            )

    def _trace_this_batch(self, batch) -> bool:
        batch_request_ids = tuple(request.rid for request in batch.reqs)
        return (
            self._diagnostic_trace_path is not None
            and self._diagnostic_request_ids is not None
            and len(batch_request_ids) == len(self._diagnostic_request_ids)
            and set(batch_request_ids) == set(self._diagnostic_request_ids)
        )

    def _trace_raw_rollout_batch(self, batch) -> bool:
        return (
            self._raw_rollout_trace_path is not None
            and self._raw_rollout_request_id is not None
            and len(batch.reqs) == 1
            and batch.reqs[0].rid == self._raw_rollout_request_id
        )

    def _trace_confidence_batch(self, batch) -> bool:
        batch_request_ids = tuple(request.rid for request in batch.reqs)
        return (
            self._confidence_trace_path is not None
            and self._confidence_request_ids is not None
            and bool(batch_request_ids)
            and all(
                request_id in self._confidence_request_ids
                for request_id in batch_request_ids
            )
        )

    @property
    def draft_worker(self):
        return None

    @property
    def preloaded_weights_bytes(self) -> int:
        return self._preloaded_weights_bytes

    @property
    def weight_load_time(self) -> float:
        return self._weight_load_time

    def alloc_memory_pool(
        self,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ) -> None:
        del memory_pool_config
        if req_to_token_pool is None or token_to_kv_pool_allocator is None:
            raise RuntimeError("NEXTLAT requires the canonical target memory pools")
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator

    def init_attention_backends(self) -> None:
        # Target attention was initialized by the target worker; no draft backend.
        return None

    def _initialize_draft_fp8_weights(self) -> None:
        from sglang.kernels.ops.quantization.fp8_kernel import (
            per_token_group_quant_fp8,
        )
        from sglang.srt.layers.quantization.fp8_utils import apply_fp8_linear

        def quantize_transposed(weight: torch.Tensor):
            canonical = weight.detach().contiguous()
            # SGLang's SM90 path consumes [K,N] FP8 weights with one scale per
            # output channel.  Quantizing each canonical [N,K] row as one group
            # preserves substantially more range than the old scalar-weight
            # path and dispatches the native CUTLASS fp8_scaled_mm kernel.
            quantized, scale = per_token_group_quant_fp8(
                canonical, group_size=canonical.shape[-1]
            )
            return quantized.T, scale.T.contiguous()

        with torch.no_grad():
            self._fp8_mlp0_weight, self._fp8_mlp0_scale = quantize_transposed(
                self.dynamics.mlp[0].weight
            )
            self._fp8_mlp2_weight, self._fp8_mlp2_scale = quantize_transposed(
                self.dynamics.mlp[2].weight
            )
            self._fp8_mlp4_weight, self._fp8_mlp4_scale = quantize_transposed(
                self.dynamics.mlp[4].weight
            )
            self._fp8_head_weight, self._fp8_head_scale = quantize_transposed(
                self.hot_lm_head_weight
            )
        self._draft_fp8_quant = apply_fp8_linear
        fp8_tensors = (
            self._fp8_mlp0_weight,
            self._fp8_mlp0_scale,
            self._fp8_mlp2_weight,
            self._fp8_mlp2_scale,
            self._fp8_mlp4_weight,
            self._fp8_mlp4_scale,
            self._fp8_head_weight,
            self._fp8_head_scale,
        )
        self._preloaded_weights_bytes += sum(
            tensor.numel() * tensor.element_size() for tensor in fp8_tensors
        )

    def _draft_fp8_linear(
        self,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> torch.Tensor:
        if self._draft_fp8_quant is None:
            raise RuntimeError("NEXTLAT FP8 draft weights are not initialized")
        return self._draft_fp8_quant(
            inputs,
            weight,
            weight_scale,
            cutlass_fp8_supported=True,
            use_per_token_if_dynamic=True,
            pad_output=False,
        )

    def _draft_fp8_dynamics(
        self, current_states: torch.Tensor, token_embeddings: torch.Tensor
    ) -> torch.Tensor:
        joined = torch.cat((token_embeddings, current_states), dim=-1)
        hidden = self.dynamics.norm(joined)
        hidden = F.gelu(
            self._draft_fp8_linear(
                hidden, self._fp8_mlp0_weight, self._fp8_mlp0_scale
            )
        )
        hidden = F.gelu(
            self._draft_fp8_linear(
                hidden, self._fp8_mlp2_weight, self._fp8_mlp2_scale
            )
        )
        return current_states + self._draft_fp8_linear(
            hidden, self._fp8_mlp4_weight, self._fp8_mlp4_scale
        )

    def _dense_m8_draft_chain(
        self, current_states: torch.Tensor, anchor_tokens: torch.Tensor
    ) -> torch.Tensor:
        state = current_states
        input_tokens = anchor_tokens
        outputs: list[torch.Tensor] = []
        embeddings = self.target_model.get_input_embeddings()
        for _ in range(self.max_draft_tokens):
            token_embeddings = embeddings(input_tokens)
            if self._draft_fp8_enabled:
                state = self._draft_fp8_dynamics(state, token_embeddings)
                logits = self._draft_fp8_linear(
                    state, self._fp8_head_weight, self._fp8_head_scale
                )
            else:
                state = self.dynamics(state, token_embeddings)
                logits = F.linear(state, self.hot_lm_head_weight)
            reduced_ids = torch.argmax(logits, dim=-1).to(torch.int64)
            input_tokens = self.hot_token_id[reduced_ids]
            outputs.append(input_tokens)
        return torch.stack(outputs, dim=1)

    def _raw_recurrent_chain(
        self, current_states: torch.Tensor, anchor_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Free-run one exact anchor plus 127 unverified recurrent tokens."""

        if current_states.shape != (1, self.contract.hidden_size):
            raise RuntimeError("raw recurrent rollout requires one exact hidden state")
        if anchor_tokens.shape != (1,):
            raise RuntimeError("raw recurrent rollout requires one exact anchor token")
        state = current_states
        input_tokens = anchor_tokens.to(torch.int64)
        outputs = [input_tokens]
        embeddings = self.target_model.get_input_embeddings()
        for _ in range(_RAW_ROLLOUT_TOKENS - 1):
            token_embeddings = embeddings(input_tokens)
            if self._draft_fp8_enabled:
                state = self._draft_fp8_dynamics(state, token_embeddings)
                logits = self._draft_fp8_linear(
                    state, self._fp8_head_weight, self._fp8_head_scale
                )
            else:
                state = self.dynamics(state, token_embeddings)
                logits = F.linear(state, self.hot_lm_head_weight)
            reduced_ids = torch.argmax(logits, dim=-1).to(torch.int64)
            input_tokens = self.hot_token_id[reduced_ids]
            outputs.append(input_tokens)
        return torch.stack(outputs, dim=1)

    def _top2_draft_lines(
        self, current_states: torch.Tensor, anchor_tokens: torch.Tensor, horizon: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Both precision rails use the same protected greedy beam and FP32
        # hot-vocabulary log-probability scoring. Only matrix arithmetic changes.
        project = self._project_logits
        if self._draft_fp8_enabled:
            def project(states):
                return self._draft_fp8_linear(
                    states, self._fp8_head_weight, self._fp8_head_scale
                )

        return recurrent_beam_lines(
            current_state=current_states,
            anchor_token=anchor_tokens,
            dynamics=(
                self._draft_fp8_dynamics if self._draft_fp8_enabled else self.dynamics
            ),
            embed_tokens=self.target_model.get_input_embeddings(),
            project_hot_logits=project,
            hot_token_ids=self.hot_token_id,
            horizon=horizon,
            beam_width=TREE_BEAM_WIDTH,
        )

    def _capture_dense_draft_cuda_graph(self) -> None:
        if not self._draft_cuda_graph_enabled:
            return
        if self._draft_cuda_graph is not None:
            raise RuntimeError("NEXTLAT dense draft CUDA graph was captured twice")

        static_state = torch.zeros(
            (self._draft_graph_batch_size, self.contract.hidden_size),
            dtype=torch.bfloat16,
            device=self.device,
        )
        static_anchor = torch.zeros(
            (self._draft_graph_batch_size,), dtype=torch.int64, device=self.device
        )
        def draft():
            if self._top2_tree_verify:
                return self._top2_draft_lines(static_state, static_anchor, TREE_HORIZON)
            return self._dense_m8_draft_chain(static_state, static_anchor), None

        capture_stream = torch.cuda.Stream(device=self.device)
        capture_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(capture_stream):
            # Prime cuBLAS and the caching allocator on the capture stream.
            for _ in range(2):
                draft()
        capture_stream.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            static_tokens, static_log_q = draft()
        capture_stream.synchronize()
        self._draft_graph_state = static_state
        self._draft_graph_anchor = static_anchor
        self._draft_graph_tokens = static_tokens
        self._draft_graph_log_q = static_log_q
        self._draft_cuda_graph = graph

    def _replay_dense_draft_cuda_graph(
        self, current_states: torch.Tensor, anchor_tokens: torch.Tensor
    ) -> torch.Tensor:
        if (
            self._draft_cuda_graph is None
            or self._draft_graph_state is None
            or self._draft_graph_anchor is None
            or self._draft_graph_tokens is None
        ):
            raise RuntimeError("NEXTLAT dense draft CUDA graph is not initialized")
        self._draft_graph_state.copy_(current_states)
        self._draft_graph_anchor.copy_(anchor_tokens)
        self._draft_cuda_graph.replay()
        return self._draft_graph_tokens

    def init_cuda_graphs(self) -> None:
        runner = self.model_runner.decode_cuda_graph_runner
        if self._target_verify_graph_diagnostic:
            capture_contract = (
                assert_tree_captured_runner(runner)
                if self._top2_tree_verify_graph
                else assert_native_captured_runner(runner)
                if self._native_ragged_verify_graph
                else assert_diagonal_captured_runner(runner)
            )
            if self._target_verify_graph_trace_path is not None:
                self._write_target_verify_graph_trace(
                    {"event": "capture_contract", **capture_contract}
                )
            if target_bf16_logits_buffer_enabled():
                logits_buffer = getattr(
                    getattr(runner, "buffers", None),
                    "next_token_logits_buffer",
                    None,
                )
                expected_shape = (
                    diagonal_decode_max_rows(),
                    self.contract.vocab_size,
                )
                if (
                    not isinstance(logits_buffer, torch.Tensor)
                    or logits_buffer.dtype != torch.bfloat16
                    or tuple(logits_buffer.shape) != expected_shape
                ):
                    observed = None
                    if isinstance(logits_buffer, torch.Tensor):
                        observed = {
                            "shape": tuple(logits_buffer.shape),
                            "dtype": str(logits_buffer.dtype),
                        }
                    raise RuntimeError(
                        "NEXTLAT target BF16 logits buffer was not installed: "
                        f"expected={expected_shape}/torch.bfloat16 observed={observed}"
                    )
            self._capture_dense_draft_cuda_graph()
            return
        # Production admission requires eager execution. A graph is drift.
        if runner is not None:
            raise RuntimeError("NEXTLAT eager tier found an initialized CUDA graph runner")
        self._capture_dense_draft_cuda_graph()

    def clear_cache_pool(self) -> None:
        # No draft KV or persistent request-table rows survive a decode call.
        return None

    def _write_diagnostic_trace(
        self, payload: dict[str, Any], *, exclusive: bool = False
    ) -> None:
        if self._diagnostic_trace_path is None:
            return
        body = {"schema_version": 1, **payload}
        mode = "x" if exclusive else "a"
        with self._diagnostic_trace_path.open(mode, encoding="utf-8") as handle:
            handle.write(json.dumps(body, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _write_raw_rollout_trace(
        self, payload: dict[str, Any], *, exclusive: bool = False
    ) -> None:
        if self._raw_rollout_trace_path is None:
            return
        body = {"schema_version": 1, **payload}
        mode = "x" if exclusive else "a"
        with self._raw_rollout_trace_path.open(mode, encoding="utf-8") as handle:
            handle.write(json.dumps(body, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _write_phase_trace(
        self, payload: dict[str, Any], *, exclusive: bool = False
    ) -> None:
        if self._phase_trace_path is None:
            return
        body = {"schema_version": 1, **payload}
        mode = "x" if exclusive else "a"
        with self._phase_trace_path.open(mode, encoding="utf-8") as handle:
            handle.write(json.dumps(body, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _write_confidence_trace(
        self, payload: dict[str, Any], *, exclusive: bool = False
    ) -> float:
        if self._confidence_trace_path is None:
            return 0.0
        started_ns = time.perf_counter_ns()
        body = {"schema_version": 1, **payload}
        mode = "x" if exclusive else "a"
        with self._confidence_trace_path.open(mode, encoding="utf-8") as handle:
            handle.write(json.dumps(body, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            if exclusive:
                os.fsync(handle.fileno())
        return (time.perf_counter_ns() - started_ns) / 1_000_000.0

    def _write_target_verify_graph_trace(
        self, payload: dict[str, Any], *, exclusive: bool = False
    ) -> None:
        if self._target_verify_graph_trace_path is None:
            return
        body = {"schema_version": 1, **payload}
        mode = "x" if exclusive else "a"
        with self._target_verify_graph_trace_path.open(
            mode, encoding="utf-8"
        ) as handle:
            handle.write(json.dumps(body, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _diagnostic_top_logits(self, logits: torch.Tensor) -> list[dict[str, Any]]:
        values, indices = torch.topk(
            logits[:, : self.contract.vocab_size].float(), k=4, dim=-1
        )
        values_cpu = values.detach().cpu().tolist()
        indices_cpu = indices.detach().cpu().tolist()
        return [
            {
                "top_token_ids": token_ids,
                "top_logits": row,
                "top1_minus_top2_logit": row[0] - row[1],
            }
            for row, token_ids in zip(values_cpu, indices_cpu)
        ]

    def _reject_weight_update(self, *_args, **_kwargs):
        return False, (
            "NEXTLAT target and dynamics are digest-bound as one serving artifact; "
            "live weight updates are not admitted"
        )

    update_weights_from_disk = _reject_weight_update
    update_weights_from_ipc = _reject_weight_update
    update_weights_from_tensor = _reject_weight_update

    def _before_target_verify_forward(self) -> None:
        """Placement hook; ordinary colocated serving has nothing to acquire."""

    def _after_target_verify_forward(self) -> None:
        """Placement hook; ordinary colocated serving has nothing to release."""

    def _validate_batch_contract(self, batch) -> None:
        if batch.has_grammar:
            raise RuntimeError("NEXTLAT fixed tier does not support grammar decoding")
        if batch.return_logprob:
            raise RuntimeError("NEXTLAT fixed tier does not support return_logprob")
        if getattr(self, "_top2_tree_verify", False) and len(batch.reqs) != 1:
            raise RuntimeError("NEXTLAT top2 tree verification is fixed batch size one")
        sampling_info = batch.sampling_info
        if self._use_rejection_sampling:
            from sglang.srt.sampling.sampling_params import TOP_K_ALL

            if len(batch.reqs) != 1:
                raise RuntimeError(
                    "NEXTLAT rejection sampling currently supports fixed batch size one"
                )
            if sampling_info is None or sampling_info.is_all_greedy:
                raise RuntimeError(
                    "NEXTLAT rejection sampling requires non-greedy temperature sampling"
                )
            if (
                sampling_info.need_top_k_sampling
                or sampling_info.need_top_p_sampling
                or sampling_info.need_min_p_sampling
            ):
                raise RuntimeError(
                    "NEXTLAT rejection sampling currently supports temperature only"
                )
        else:
            TOP_K_ALL = None
            if sampling_info is not None and not sampling_info.is_all_greedy:
                raise RuntimeError("NEXTLAT fixed tier requires greedy top-k=1 decoding")

        if sampling_info is not None:
            if sampling_info.has_custom_logit_processor:
                raise RuntimeError("NEXTLAT does not support custom logit processors")
            if sampling_info.logit_bias is not None:
                raise RuntimeError("NEXTLAT does not support logit bias")
            penalizer = sampling_info.penalizer_orchestrator
            if penalizer is not None and penalizer.is_required:
                raise RuntimeError("NEXTLAT does not support token penalties")
        for request in batch.reqs:
            params = request.sampling_params
            if self._use_rejection_sampling:
                temperature_only = {
                    "temperature": not (
                        isinstance(params.temperature, (int, float))
                        and math.isfinite(float(params.temperature))
                        and float(params.temperature) > 0.0
                    ),
                    "top_k": params.top_k != TOP_K_ALL,
                    "top_p": params.top_p != 1.0,
                    "min_p": params.min_p != 0.0,
                }
            else:
                temperature_only = {
                    "top_k": params.top_k != 1,
                    "top_p": params.top_p != 1.0,
                    "min_p": params.min_p != 0.0,
                }
            unsupported = {
                **temperature_only,
                "frequency_penalty": params.frequency_penalty != 0.0,
                "presence_penalty": params.presence_penalty != 0.0,
                "repetition_penalty": params.repetition_penalty != 1.0,
                "logit_bias": params.logit_bias is not None,
                "custom_params": params.custom_params is not None,
                "parallel_or_beam": params.n != 1 or params.beam_width is not None,
            }
            enabled = sorted(name for name, value in unsupported.items() if value)
            if enabled:
                raise RuntimeError(
                    f"NEXTLAT request {request.rid!r} enables unsupported sampling "
                    f"features: {enabled}"
                )

    @staticmethod
    def _select_last_extend_hidden(batch, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size = len(batch.reqs)
        if batch.extend_lens is None or len(batch.extend_lens) != batch_size:
            raise RuntimeError("NEXTLAT prefill requires per-request extend_lens")
        extend_lens = torch.tensor(
            batch.extend_lens, dtype=torch.int64, device=hidden_states.device
        )
        if any(int(length) <= 0 for length in batch.extend_lens):
            raise RuntimeError("NEXTLAT cannot seed from an empty extend row")
        if hidden_states.shape[0] != int(sum(batch.extend_lens)):
            raise RuntimeError(
                "NEXTLAT full prefill hidden rows differ from sum(extend_lens)"
            )
        last_indices = torch.cumsum(extend_lens, dim=0) - 1
        return hidden_states[last_indices].contiguous()

    def _project_logits(self, states: torch.Tensor) -> torch.Tensor:
        if self.hot_lm_head_weight is None:
            raise RuntimeError("BF16 head is unavailable in B1 FP8 slim serving")
        return F.linear(states, self.hot_lm_head_weight)

    def _proposal_lens_cpu(self, batch) -> list[int]:
        lengths: list[int] = []
        for request in batch.reqs:
            maximum = request.sampling_params.max_new_tokens
            if maximum is None:
                raise RuntimeError("NEXTLAT requires a finite max_new_tokens")
            remaining_outputs = int(maximum) - len(request.output_ids)
            if remaining_outputs <= 0:
                raise RuntimeError("NEXTLAT received a request with no output budget")
            # The round must reserve one output slot for the exact target bonus.
            lengths.append(
                max(0, min(self.max_draft_tokens, remaining_outputs - 1))
            )
        return lengths

    def _verify_sampled_b1(
        self,
        *,
        proposal: NextLatProposal,
        target_logits: torch.Tensor,
        sampling_info,
        prefix_seq_lens: torch.Tensor,
        active_drafts: int,
    ):
        """Run fixed-B1 chain rejection sampling with SGLang's pinned kernel."""

        from sglang.kernels.ops.speculative.dspark.dspark_accept import SoftmaxTemp
        from sglang.kernels.ops.speculative.reject_sampling import (
            chain_speculative_sampling_triton,
        )
        from sglang.srt.speculative.eagle_utils import _verify_coins

        if proposal.q_hot is None:
            raise RuntimeError("sampled NEXTLAT proposal lost its q distribution")
        if proposal.anchor_tokens.shape[0] != 1:
            raise RuntimeError("sampled NEXTLAT verification supports B1 only")
        active_width = active_drafts + 1
        candidates = proposal.candidates[:, :active_width].contiguous()
        logits = mask_embedding_only_logits(
            target_logits[:active_width, : self.contract.vocab_size],
            tokenizer_vocab_size=QWEN3_1_7B_TOKENIZER_VOCAB_SIZE,
        )
        target_probs = SoftmaxTemp.execute(
            logits=logits,
            temperatures=sampling_info.temperatures,
            rows_per_request=active_width,
        ).view(1, active_width, self.contract.vocab_size)
        draft_probs = scatter_hot_probabilities(
            proposal.q_hot[:, :active_drafts],
            token_id_map=self.hot_token_id,
            target_vocab_size=self.contract.vocab_size,
        )
        # With no remaining draft budget the chain samples directly from p and
        # never dereferences q. Keep one harmless row so the Triton wrapper has
        # ordinary non-zero strides on every tensor argument.
        draft_probs_kernel = (
            draft_probs
            if active_drafts > 0
            else torch.zeros(
                (1, 1, self.contract.vocab_size),
                dtype=target_probs.dtype,
                device=self.device,
            )
        )

        retrieve_index = torch.arange(
            active_width, dtype=torch.int64, device=self.device
        ).view(1, active_width)
        retrieve_next_token = torch.arange(
            1, active_width + 1, dtype=torch.int64, device=self.device
        ).view(1, active_width)
        retrieve_next_token[:, -1] = -1
        retrieve_next_sibling = torch.full(
            (1, active_width), -1, dtype=torch.int64, device=self.device
        )
        predicts = torch.empty(
            (active_width,), dtype=torch.int32, device=self.device
        )
        kernel_accept_index = torch.full(
            (1, active_width), -1, dtype=torch.int32, device=self.device
        )
        num_correct_drafts = torch.empty(
            (1,), dtype=torch.int32, device=self.device
        )
        coins, final_coins = _verify_coins(
            sampling_info=sampling_info,
            seq_lens=prefix_seq_lens,
            draft_token_num=active_width,
            candidates=candidates,
            device=self.device,
        )
        chain_speculative_sampling_triton(
            predicts=predicts,
            accept_index=kernel_accept_index,
            accept_token_num=num_correct_drafts,
            candidates=candidates,
            retrive_index=retrieve_index,
            retrive_next_token=retrieve_next_token,
            retrive_next_sibling=retrieve_next_sibling,
            uniform_samples=coins,
            uniform_samples_for_final_sampling=final_coins,
            target_probs=target_probs,
            draft_probs=draft_probs_kernel,
            threshold_single=1.0,
            threshold_acc=1.0,
            deterministic=True,
        )
        return finish_sampled_verification(
            proposal=proposal,
            predicts=predicts,
            kernel_accept_index=kernel_accept_index,
            num_correct_drafts=num_correct_drafts,
            prefix_seq_lens=prefix_seq_lens,
            active_drafts=active_drafts,
        )

    def _trace_top2_tree_round(
        self,
        *,
        batch,
        seed: NextLatDraftInput,
        horizon: int,
        tree,
        positions: torch.Tensor,
        retrieve_index: torch.Tensor,
        retrieve_next_token: torch.Tensor,
        retrieve_next_sibling: torch.Tensor,
        custom_mask: torch.Tensor,
        verify_forward_batch,
        target_logits: torch.Tensor,
        predict: torch.Tensor,
        accept_lens: torch.Tensor,
        accept_index: torch.Tensor,
        num_correct_drafts: torch.Tensor,
        accepted_predictions: torch.Tensor,
        selected_hidden_indices: torch.Tensor,
        bonus_tokens: torch.Tensor,
        new_seq_lens: torch.Tensor,
        graph_admitted: bool,
        graph_executed: bool,
        graph_replay: dict[str, Any] | None,
    ) -> None:
        """Publish one request-scoped tree round without changing decode results."""

        if not self._trace_this_batch(batch):
            return
        request_ids = [request.rid for request in batch.reqs]
        if len(request_ids) != 1:
            raise RuntimeError("NEXTLAT tree trace requires one request")
        accept_lens_cpu = accept_lens.detach().cpu().tolist()
        if (
            len(accept_lens_cpu) != 1
            or not 1 <= accept_lens_cpu[0] <= horizon + 1
        ):
            raise RuntimeError("NEXTLAT tree trace received an invalid accept length")
        accepted_count = int(accept_lens_cpu[0])
        accepted_rows = accept_index[0, :accepted_count].to(torch.int64)
        if bool(torch.any(accepted_rows < 0).item()):
            raise RuntimeError("NEXTLAT tree trace received a negative accepted row")

        verify_out_cache_locs = verify_forward_batch.out_cache_loc
        verify_req_pool_indices = verify_forward_batch.req_pool_indices
        if (
            not isinstance(verify_out_cache_locs, torch.Tensor)
            or verify_out_cache_locs.numel() != tree.node_budget
            or not isinstance(verify_req_pool_indices, torch.Tensor)
        ):
            raise RuntimeError("NEXTLAT tree trace lost target verify KV geometry")
        accepted_source_kv_locs = verify_out_cache_locs.index_select(
            0, accepted_rows
        )

        batch_req_pool_indices = batch.req_pool_indices
        req_to_token = self.req_to_token_pool.req_to_token
        if (
            not isinstance(batch_req_pool_indices, torch.Tensor)
            or batch_req_pool_indices.numel() != 1
            or not isinstance(req_to_token, torch.Tensor)
        ):
            raise RuntimeError("NEXTLAT tree trace lost canonical target KV state")
        request_pool_index = int(batch_req_pool_indices.flatten()[0].item())
        prefix_seq_len = int(batch.seq_lens.flatten()[0].item())
        canonical_kv_locs = req_to_token[
            request_pool_index,
            prefix_seq_len : prefix_seq_len + accepted_count,
        ]
        if canonical_kv_locs.numel() != accepted_count:
            raise RuntimeError("NEXTLAT tree trace canonical KV slice is incomplete")

        target_predictions = torch.argmax(
            target_logits[:, : self.contract.vocab_size], dim=-1
        ).to(torch.int64)
        reference = reference_greedy_tree_acceptance(tree, target_predictions)
        self._write_diagnostic_trace(
            {
                "event": "tree_verify",
                "forward_iter": int(batch.forward_iter),
                "request_ids": request_ids,
                "output_start_indices": [
                    len(request.output_ids) for request in batch.reqs
                ],
                "prefix_seq_lens": [prefix_seq_len],
                "horizon": horizon,
                "anchor_tokens": seed.bonus_tokens.detach().cpu().tolist(),
                "tree": {
                    "node_budget": tree.node_budget,
                    "actual_node_count": int(tree.actual_node_count.item()),
                    "shared_prefix_length": int(
                        tree.shared_prefix_length.item()
                    ),
                    "selected_lines": tree.selected_lines.detach().cpu().tolist(),
                    "selected_log_q": tree.selected_log_q.detach().cpu().tolist(),
                    "node_tokens": tree.node_tokens.detach().cpu().tolist(),
                    "ancestor_mask": tree.ancestor_mask.detach().cpu().tolist(),
                },
                "target_verify": {
                    "input_ids": verify_forward_batch.input_ids.detach()
                    .cpu()
                    .tolist(),
                    "positions": positions.detach().cpu().tolist(),
                    "custom_mask_shape": [
                        tree.node_budget,
                        prefix_seq_len + tree.node_budget,
                    ],
                    "custom_mask": custom_mask.reshape(
                        tree.node_budget, prefix_seq_len + tree.node_budget
                    )
                    .detach()
                    .cpu()
                    .tolist(),
                    "retrieve_index": retrieve_index.detach().cpu().tolist(),
                    "retrieve_next_token": retrieve_next_token.detach()
                    .cpu()
                    .tolist(),
                    "retrieve_next_sibling": retrieve_next_sibling.detach()
                    .cpu()
                    .tolist(),
                    "target_predictions": target_predictions.detach()
                    .cpu()
                    .tolist(),
                    "kernel_predict": predict.detach().cpu().tolist(),
                    "target_rows": self._diagnostic_top_logits(target_logits),
                    "graph_admitted": graph_admitted,
                    "graph_executed": graph_executed,
                    "graph_replay": graph_replay,
                },
                "acceptance": {
                    "num_correct_drafts": num_correct_drafts.detach()
                    .cpu()
                    .tolist(),
                    "accept_lens": accept_lens_cpu,
                    "accept_index": accept_index.detach().cpu().tolist(),
                    "accepted_node_indices": accepted_rows.detach().cpu().tolist(),
                    "committed_output_tokens": accepted_predictions[
                        0, :accepted_count
                    ]
                    .detach()
                    .cpu()
                    .tolist(),
                    "selected_hidden_indices": selected_hidden_indices.detach()
                    .cpu()
                    .tolist(),
                    "bonus_tokens": bonus_tokens.detach().cpu().tolist(),
                    "new_seq_lens": new_seq_lens.detach().cpu().tolist(),
                    "reference_greedy_tree": reference,
                },
                "kv": {
                    "request_pool_index": request_pool_index,
                    "verify_req_pool_indices": verify_req_pool_indices.detach()
                    .cpu()
                    .tolist(),
                    "verify_out_cache_locs": verify_out_cache_locs.detach()
                    .cpu()
                    .tolist(),
                    "accepted_source_kv_locs": accepted_source_kv_locs.detach()
                    .cpu()
                    .tolist(),
                    "canonical_sequence_positions": list(
                        range(prefix_seq_len, prefix_seq_len + accepted_count)
                    ),
                    "canonical_kv_locs_after_move": canonical_kv_locs.detach()
                    .cpu()
                    .tolist(),
                },
            }
        )

    def _decode_top2_tree(self, batch, on_publish):
        """Borrow canonical request-table rows only for this tree forward."""

        if not self._top2_tree_diagonal_verify_graph:
            return self._decode_top2_tree_impl(batch, on_publish)
        pool = self.req_to_token_pool
        if pool is None:
            raise RuntimeError("tree diagonal verifier has no request-to-token pool")
        if self._tree_diagonal_req_pool_rows is not None:
            raise RuntimeError("tree diagonal verifier already holds request rows")
        rows = pool.alloc_rows(TREE_NODE_BUDGET)
        if rows is None:
            raise RuntimeError("tree diagonal verifier could not borrow fourteen rows")
        try:
            if len(rows) != TREE_NODE_BUDGET:
                raise RuntimeError("tree diagonal verifier received the wrong row count")
            self._tree_diagonal_req_pool_rows = rows
            self._tree_diagonal_req_pool_indices = torch.tensor(
                rows, dtype=torch.int32, device=self.device
            )
            return self._decode_top2_tree_impl(batch, on_publish)
        finally:
            # Tree scheduling is serial. Later writes to reused table rows are
            # enqueued after this forward on the same CUDA stream.
            try:
                pool.free_rows(rows)
            finally:
                self._tree_diagonal_req_pool_rows = None
                self._tree_diagonal_req_pool_indices = None

    def _decode_top2_tree_impl(self, batch, on_publish):
        """Verify two recurrent beam directions with SGLang's target tree path.

        The recurrent beam may use the fixed H8 draft graph; short completion
        tails keep their actual horizon in eager mode. The target sees one
        fixed 14-node irregular tree through either the explicit eager
        rail or the native one-shape target CUDA graph, and SGLang's EAGLE
        greedy kernel selects the target-consistent route.  No q score decides
        what is committed.
        """

        from sgl_kernel.speculative import reconstruct_indices_from_tree_mask
        from sglang.srt.managers.utils import GenerationBatchResult
        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
        from sglang.srt.speculative.eagle_info import EagleVerifyInput
        from sglang.srt.speculative.eagle_utils import (
            eagle_prepare_for_verify,
            eagle_sample,
        )
        from sglang.srt.speculative.spec_utils import (
            commit_mamba_states_after_verify,
            move_accept_tokens_to_target_kvcache,
            record_stream_each,
            record_stream_for_v2_verify,
        )

        phase_events: dict[str, Any] | None = (
            {} if self._phase_trace_path is not None else None
        )
        phase_host_started_ns = time.perf_counter_ns() if phase_events is not None else 0

        def mark_phase(name: str) -> None:
            if phase_events is not None:
                event = torch.cuda.Event(enable_timing=True)
                event.record(torch.cuda.current_stream(self.device))
                phase_events[name] = event

        mark_phase("decode_start")
        if len(batch.reqs) != 1:
            raise RuntimeError("NEXTLAT top2 tree verification is fixed B1")
        if self.token_to_kv_pool_allocator is None or self.req_to_token_pool is None:
            raise RuntimeError("NEXTLAT target memory pools were not injected")
        seed = batch.spec_info
        if not isinstance(seed, NextLatDraftInput):
            raise RuntimeError("NEXTLAT top2 tree lost its exact recurrent seed")
        if seed.bonus_tokens.shape != (1,) or seed.hidden_states.shape != (
            1,
            self.contract.hidden_size,
        ):
            raise RuntimeError("NEXTLAT top2 tree requires one exact anchor state")
        if batch.seq_lens_cpu is None or len(batch.seq_lens_cpu) != 1:
            raise RuntimeError(
                "NEXTLAT top2 tree verifier requires one host sequence length"
            )

        horizon = self._proposal_lens_cpu(batch)[0]
        mark_phase("draft_start")
        if self._draft_cuda_graph_enabled and horizon == TREE_HORIZON:
            selected_lines = self._replay_dense_draft_cuda_graph(
                seed.hidden_states, seed.bonus_tokens
            )
            selected_log_q = self._draft_graph_log_q
            if selected_log_q is None:
                raise RuntimeError("NEXTLAT tree draft graph lost its beam scores")
        else:
            selected_lines, selected_log_q = self._top2_draft_lines(
                seed.hidden_states, seed.bonus_tokens, horizon
            )
        tree = pack_top2_prefix_tree(
            anchor_token=seed.bonus_tokens,
            selected_lines=selected_lines,
            selected_log_q=selected_log_q,
            node_budget=self.verify_width,
        )
        mark_phase("draft_end")

        device = self.device
        node_budget = self.verify_width
        tree_mask = tree.ancestor_mask.contiguous().flatten()
        positions = torch.empty((node_budget,), dtype=torch.int64, device=device)
        retrieve = torch.full(
            (3, 1, node_budget), -1, dtype=torch.int64, device=device
        )
        retrieve_index, retrieve_next_token, retrieve_next_sibling = retrieve
        reconstruct_indices_from_tree_mask(
            tree_mask,
            batch.seq_lens,
            positions,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            1,
            node_budget,
        )
        prefix_len = int(batch.seq_lens_cpu[0])
        custom_mask = torch.cat(
            (
                torch.ones(
                    (node_budget, prefix_len), dtype=torch.bool, device=device
                ),
                tree.ancestor_mask,
            ),
            dim=1,
        ).flatten()
        verify_input = EagleVerifyInput(
            draft_token=tree.node_tokens,
            custom_mask=custom_mask,
            positions=positions,
            retrieve_index=retrieve_index,
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            retrieve_cum_len=None,
            # The actual topology is irregular; -1 selects SGLang's generic
            # linked-tree traversal while server topk=2 enables target masking.
            topk=-1,
            spec_steps=horizon,
            draft_token_num=node_budget,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            seq_lens_sum=None,
            seq_lens_cpu=None,
        )
        batch.spec_info = verify_input
        stream = torch.cuda.current_stream(device)
        record_stream_for_v2_verify(batch, verify_input, stream)
        verify_forward_batch, can_run_cuda_graph = eagle_prepare_for_verify(
            verify_input,
            self.req_to_token_pool,
            batch,
            self.target_worker,
        )
        native_graph_geometry = bool(
            self._top2_tree_verify_graph and horizon == TREE_HORIZON
        )
        diagonal_graph_geometry = self._top2_tree_diagonal_verify_graph
        exact_graph_geometry = native_graph_geometry or diagonal_graph_geometry
        diagonal_geometry: dict[str, Any] | None = None
        if diagonal_graph_geometry:
            if can_run_cuda_graph:
                raise RuntimeError(
                    "tree diagonal verifier unexpectedly admitted TARGET_VERIFY"
                )
            virtual_rows = self._tree_diagonal_req_pool_indices
            if virtual_rows is None:
                raise RuntimeError("tree diagonal verifier lost its virtual request rows")
            diagonal_geometry = diagonalize_tree_verify_forward_batch(
                verify_forward_batch,
                ancestor_mask=tree.ancestor_mask,
                canonical_req_pool_indices=batch.req_pool_indices,
                virtual_req_pool_indices=virtual_rows,
                req_to_token=self.req_to_token_pool.req_to_token,
                prefix_seq_len=prefix_len,
                page_size=self.token_to_kv_pool_allocator.page_size,
            )
            if not self.model_runner.decode_cuda_graph_runner.can_run_graph(
                verify_forward_batch
            ):
                raise RuntimeError("tree diagonal DECODE verifier missed its B14 graph")
        elif bool(can_run_cuda_graph) != native_graph_geometry:
            raise RuntimeError(
                "NEXTLAT top2 tree graph admission differs from fixed B1/H8/N14"
            )
        record_stream_each(
            (
                batch.input_ids,
                batch.out_cache_loc,
                positions,
                custom_mask,
                verify_forward_batch.req_pool_indices,
            ),
            stream,
        )
        mark_phase("target_start")
        target_output = self.target_worker.forward_batch_generation(
            batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
            # TARGET_VERIFY metadata is stale after tree diagonalization.  The
            # ordinary-DECODE graph initializes its own fixed B14 metadata.
            skip_attn_backend_init=None,
        )
        mark_phase("target_end")
        target_graph_executed = bool(target_output.can_run_cuda_graph)
        if target_graph_executed != exact_graph_geometry:
            raise RuntimeError(
                "NEXTLAT top2 tree target graph execution differs from admission"
            )
        logits_output = target_output.logits_output
        if logits_output is None or logits_output.hidden_states is None:
            raise RuntimeError("NEXTLAT top2 tree target returned no hidden states")
        if logits_output.next_token_logits.shape[0] != node_budget:
            raise RuntimeError("NEXTLAT top2 tree target returned the wrong row count")
        graph_replay: dict[str, Any] | None = None
        if exact_graph_geometry:
            if diagonal_graph_geometry:
                graph_replay = diagonal_replay_snapshot(
                    self.model_runner.decode_cuda_graph_runner
                )
                graph_replay["tree_diagonal_geometry"] = diagonal_geometry
            else:
                graph_replay = tree_replay_snapshot(
                    self.model_runner.decode_cuda_graph_runner, verify_forward_batch
                )

        predict, accept_lens, accept_index = eagle_sample(
            verify_input, batch, logits_output
        )
        if accept_index.shape != (1, horizon + 1):
            raise RuntimeError("NEXTLAT top2 tree acceptance width drifted")
        num_correct_drafts = accept_lens - 1
        new_seq_lens = batch.seq_lens + accept_lens

        move_accept_tokens_to_target_kvcache(
            batch,
            accept_index,
            num_correct_drafts,
            self.token_to_kv_pool_allocator,
        )
        commit_mamba_states_after_verify(
            self.target_worker,
            batch,
            accept_lens,
            accept_index,
            node_budget,
        )

        safe_accept_index = accept_index.to(torch.int64).clamp(min=0)
        accepted_predictions = predict[safe_accept_index]
        output_tokens = torch.zeros(
            (1, node_budget), dtype=predict.dtype, device=device
        )
        output_tokens[:, : horizon + 1] = accepted_predictions
        selected_columns = (accept_lens - 1).to(torch.int64)[:, None]
        selected_hidden_indices = accept_index.gather(
            1, selected_columns
        ).to(torch.int64).flatten()
        bonus_tokens = predict[selected_hidden_indices].to(torch.int64)
        next_hidden = logits_output.hidden_states.index_select(
            0, selected_hidden_indices
        ).contiguous()
        self._trace_top2_tree_round(
            batch=batch,
            seed=seed,
            horizon=horizon,
            tree=tree,
            positions=positions,
            retrieve_index=retrieve_index,
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            custom_mask=custom_mask,
            verify_forward_batch=verify_forward_batch,
            target_logits=logits_output.next_token_logits,
            predict=predict,
            accept_lens=accept_lens,
            accept_index=accept_index,
            num_correct_drafts=num_correct_drafts,
            accepted_predictions=accepted_predictions,
            selected_hidden_indices=selected_hidden_indices,
            bonus_tokens=bonus_tokens,
            new_seq_lens=new_seq_lens,
            graph_admitted=exact_graph_geometry,
            graph_executed=target_graph_executed,
            graph_replay=graph_replay,
        )
        logits_output.hidden_states = None
        if on_publish is not None:
            on_publish(new_seq_lens)
        next_seed = NextLatDraftInput(
            bonus_tokens=bonus_tokens,
            hidden_states=next_hidden,
            new_seq_lens=new_seq_lens,
        )
        result = GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=output_tokens.flatten(),
            can_run_cuda_graph=target_graph_executed,
            speculative_num_draft_tokens=node_budget,
            next_draft_input=next_seed,
            accept_lens=accept_lens,
            new_seq_lens=new_seq_lens,
            extra_keep_alive_refs=[verify_forward_batch, tree],
        )
        mark_phase("decode_end")
        if phase_events is not None:
            phase_events["decode_end"].synchronize()
            host_ms = (time.perf_counter_ns() - phase_host_started_ns) / 1_000_000.0

            def elapsed_ms(start: str, end: str) -> float:
                return float(phase_events[start].elapsed_time(phase_events[end]))

            self._write_phase_trace(
                {
                    "event": "decode_phase_timing",
                    "proposal_shape": "top2_prefix_tree_n14",
                    "forward_iter": int(batch.forward_iter),
                    "request_ids": [request.rid for request in batch.reqs],
                    "batch_size": 1,
                    "proposal_lens": [horizon],
                    "num_correct_drafts": num_correct_drafts.detach().cpu().tolist(),
                    "accept_lens": accept_lens.detach().cpu().tolist(),
                    "total_verify_rows": node_budget,
                    "draft_cuda_graph": self._draft_cuda_graph_enabled and horizon == TREE_HORIZON,
                    "draft_dtype": "fp8" if self._draft_fp8_enabled else "bf16",
                    "host_critical_path_ms": host_ms,
                    "cuda_ms": {
                        "decode_total": elapsed_ms("decode_start", "decode_end"),
                        "combined_draft_loop": elapsed_ms("draft_start", "draft_end"),
                        "exact_target_verify": elapsed_ms("target_start", "target_end"),
                        "pre_draft_control": elapsed_ms("decode_start", "draft_start"),
                        "verify_setup_control": elapsed_ms("draft_end", "target_start"),
                        "post_verify_control": elapsed_ms("target_end", "decode_end"),
                    },
                }
            )
        return result

    def _prefill(self, batch, on_publish):
        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

        batch_output = self.target_worker.forward_batch_generation(
            batch, capture_hidden_mode=CaptureHiddenMode.FULL
        )
        logits_output = batch_output.logits_output
        if logits_output is None or logits_output.hidden_states is None:
            raise RuntimeError("NEXTLAT target prefill returned no post-norm hidden state")
        last_hidden = self._select_last_extend_hidden(
            batch, logits_output.hidden_states
        )
        next_token_ids = batch_output.next_token_ids.to(torch.int64)
        new_seq_lens = batch.seq_lens.clone()
        if self._trace_raw_rollout_batch(batch):
            started = torch.cuda.Event(enable_timing=True)
            finished = torch.cuda.Event(enable_timing=True)
            started.record()
            raw_tokens = self._raw_recurrent_chain(last_hidden, next_token_ids)
            finished.record()
            finished.synchronize()
            self._write_raw_rollout_trace(
                {
                    "event": "raw_rollout",
                    "request_id": batch.reqs[0].rid,
                    "exact_anchor_token": int(next_token_ids[0].item()),
                    "recurrent_token_ids": raw_tokens[0, 1:].detach().cpu().tolist(),
                    "output_token_ids": raw_tokens[0].detach().cpu().tolist(),
                    "recurrent_steps": _RAW_ROLLOUT_TOKENS - 1,
                    "cuda_ms": float(started.elapsed_time(finished)),
                }
            )
        if self._trace_this_batch(batch):
            self._write_diagnostic_trace(
                {
                    "event": "prefill",
                    "request_ids": [request.rid for request in batch.reqs],
                    "output_start_indices": [
                        len(request.output_ids) for request in batch.reqs
                    ],
                    "next_token_ids": next_token_ids.detach().cpu().tolist(),
                    "target_top_logits": self._diagnostic_top_logits(
                        logits_output.next_token_logits
                    ),
                }
            )
        batch_output.next_draft_input = NextLatDraftInput(
            bonus_tokens=next_token_ids,
            hidden_states=last_hidden,
            new_seq_lens=new_seq_lens,
        )
        batch_output.new_seq_lens = new_seq_lens
        # Only the compact recurrent seed must survive output processing.
        logits_output.hidden_states = None
        if on_publish is not None:
            on_publish(new_seq_lens)
        return batch_output

    def _idle_result(self, on_publish):
        from sglang.srt.managers.utils import GenerationBatchResult

        seed = NextLatDraftInput.empty(
            device=self.device, hidden_size=self.contract.hidden_size
        )
        accept_lens = torch.empty((0,), dtype=torch.int32, device=self.device)
        if on_publish is not None:
            on_publish(seed.new_seq_lens)
        return GenerationBatchResult(
            logits_output=None,
            next_token_ids=seed.bonus_tokens,
            accept_lens=accept_lens,
            next_draft_input=seed,
            can_run_cuda_graph=False,
            speculative_num_draft_tokens=self.verify_width,
            new_seq_lens=seed.new_seq_lens,
        )

    def _decode(self, batch, on_publish):
        if self._top2_tree_verify:
            return self._decode_top2_tree(batch, on_publish)

        from sglang.kernels.ops.speculative.dspark.dspark_verify_window import (
            BuildRaggedVerifyWindow,
        )
        from sglang.srt.managers.utils import GenerationBatchResult
        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
        from sglang.srt.speculative.dflash_info import DFlashVerifyInput
        from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout
        from sglang.srt.speculative.spec_utils import (
            commit_mamba_states_after_verify,
            fast_sample,
            move_accept_tokens_to_target_kvcache,
            record_stream_each,
            record_stream_for_v2_verify,
        )

        phase_events: dict[str, Any] | None = (
            {} if self._phase_trace_path is not None else None
        )
        phase_host_started_ns = time.perf_counter_ns()

        def mark_phase(name: str) -> None:
            if phase_events is None:
                return
            if name in phase_events:
                raise RuntimeError(f"duplicate NEXTLAT phase marker: {name}")
            event = torch.cuda.Event(enable_timing=True)
            event.record(torch.cuda.current_stream(self.device))
            phase_events[name] = event

        mark_phase("decode_start")

        if self.token_to_kv_pool_allocator is None or self.req_to_token_pool is None:
            raise RuntimeError("NEXTLAT target memory pools were not injected")
        seed = batch.spec_info
        if not isinstance(seed, NextLatDraftInput):
            raise RuntimeError(
                "NEXTLAT decode expected the exact recurrent seed from the prior target step"
            )
        batch_size = len(batch.reqs)
        if self._b1_fp8_slim and batch_size != 1:
            raise RuntimeError("NEXTLAT_B1_FP8_SLIM admits one live decode request")
        if seed.bonus_tokens.shape != (batch_size,) or seed.hidden_states.shape != (
            batch_size,
            self.contract.hidden_size,
        ):
            raise RuntimeError("NEXTLAT recurrent seed shape differs from the batch")

        proposal_lens_cpu = self._proposal_lens_cpu(batch)
        proposal_lens = torch.tensor(
            proposal_lens_cpu, dtype=torch.int32, device=self.device
        )
        mark_phase("draft_loop_start")

        confidence_this_batch = self._trace_confidence_batch(batch)
        draft_top1_logprob: list[torch.Tensor] = []
        cumulative_draft_logprob: list[torch.Tensor] = []
        running_draft_logprob: torch.Tensor | None = None

        if confidence_this_batch:
            def project_logits(states: torch.Tensor) -> torch.Tensor:
                nonlocal running_draft_logprob
                logits = self._project_logits(states)
                step_logprob = top1_log_probability(
                    logits, vocab_size=self.draft_vocab_size
                )
                running_draft_logprob = (
                    step_logprob
                    if running_draft_logprob is None
                    else running_draft_logprob + step_logprob
                )
                draft_top1_logprob.append(step_logprob)
                cumulative_draft_logprob.append(running_draft_logprob)
                return logits
        else:
            project_logits = self._project_logits

        def observe_proposal_phase(name: str, step: int) -> None:
            mark_phase(f"step_{step}_{name}")

        dense_graph_replay = bool(
            self._draft_cuda_graph_enabled
            and batch_size == self._draft_graph_batch_size
            and (
                self._b1_fp8_slim
                or proposal_lens_cpu
                == [self.max_draft_tokens] * self._draft_graph_batch_size
            )
            and not confidence_this_batch
        )
        if dense_graph_replay:
            proposal = NextLatProposal(
                anchor_tokens=seed.bonus_tokens.to(torch.int64),
                draft_tokens=self._replay_dense_draft_cuda_graph(
                    seed.hidden_states, seed.bonus_tokens
                ),
                proposal_lens=proposal_lens,
            )
            proposal.validate()
        elif self._use_rejection_sampling:
            if batch.sampling_info is None:
                raise RuntimeError("sampled NEXTLAT decode lost sampling metadata")

            def sample_hot(probabilities: torch.Tensor) -> torch.Tensor:
                return fast_sample(probabilities, num_samples=1)[1].squeeze(1)

            proposal = propose_temperature_sampled(
                current_states=seed.hidden_states,
                anchor_tokens=seed.bonus_tokens,
                proposal_lens=proposal_lens,
                temperatures=batch.sampling_info.temperatures.view(-1)[:batch_size],
                dynamics=self.dynamics,
                embed_tokens=self.target_model.get_input_embeddings(),
                project_logits=project_logits,
                vocab_size=self.draft_vocab_size,
                max_draft_tokens=self.max_draft_tokens,
                token_id_map=self.hot_token_id,
                sample_probabilities=sample_hot,
                phase_observer=(
                    observe_proposal_phase if phase_events is not None else None
                ),
            )
        else:
            proposal = propose_greedy(
                current_states=seed.hidden_states,
                anchor_tokens=seed.bonus_tokens,
                proposal_lens=proposal_lens,
                dynamics=self.dynamics,
                embed_tokens=self.target_model.get_input_embeddings(),
                project_logits=project_logits,
                vocab_size=self.draft_vocab_size,
                max_draft_tokens=self.max_draft_tokens,
                token_id_map=self.hot_token_id,
                phase_observer=(
                    observe_proposal_phase if phase_events is not None else None
                ),
            )
        mark_phase("draft_loop_end")

        fixed_verify_geometry = self._fixed_dense_m8_verify
        use_fixed_verify_geometry = bool(
            fixed_verify_geometry is not None
            and batch_size == fixed_verify_geometry.batch_size
            and proposal_lens_cpu
            == [fixed_verify_geometry.draft_tokens]
            * fixed_verify_geometry.batch_size
        )
        if use_fixed_verify_geometry:
            verify_lens_cpu = fixed_verify_geometry.verify_lens_cpu
            total_verify_tokens = fixed_verify_geometry.total_verify_tokens
        else:
            verify_lens_cpu = [length + 1 for length in proposal_lens_cpu]
            total_verify_tokens = sum(verify_lens_cpu)
        if self._native_ragged_verify_graph:
            exact_graph_geometry = bool(
                batch_size == NATIVE_RAGGED_BATCH_SIZE
                and proposal_lens_cpu
                == [NATIVE_RAGGED_DRAFT_TOKENS] * NATIVE_RAGGED_BATCH_SIZE
                and total_verify_tokens == NATIVE_RAGGED_VERIFY_ROWS
            )
        else:
            exact_graph_geometry = bool(self._diagonal_verify_graph)
        if self._diagonal_verify_graph and exact_graph_geometry:
            graph_max_rows = diagonal_decode_max_rows()
            max_active_requests = graph_max_rows // self.verify_width
            if not 1 <= batch_size <= max_active_requests:
                raise RuntimeError(
                    "diagonal graph active request count exceeds its lane"
                )
            if total_verify_tokens > graph_max_rows:
                raise RuntimeError("diagonal verify rows exceed the graph cap")
        graph_replay_record: dict[str, Any] | None = None
        graph_returned_shapes: dict[str, list[int]] | None = None
        diagonal_geometry: dict[str, Any] | None = None
        stream = torch.cuda.current_stream(self.device)
        record_stream_for_v2_verify(batch, seed, stream)
        if use_fixed_verify_geometry and fixed_verify_geometry.is_bound:
            verify_forward_batch = fixed_verify_geometry.prepare(
                batch=batch,
                proposal=proposal,
                req_to_token=self.req_to_token_pool.req_to_token,
            )
            verify_positions = verify_forward_batch.positions
            diagonal_geometry = fixed_verify_geometry.trace_geometry()
        else:
            layout = RaggedVerifyLayout.from_verify_lens(
                verify_lens_cpu=verify_lens_cpu,
                device=self.device,
                # Eager only: an exact one-element grid forbids graph padding.
                grid=[total_verify_tokens],
            )
            verify_window = BuildRaggedVerifyWindow.execute(
                batch=batch,
                layout=layout,
                draft_block_ids=proposal.dspark_draft_block_ids,
                draft_tokens=proposal.draft_tokens,
                bs=batch_size,
                device=str(self.device),
                verify_num_draft_tokens=self.verify_width,
                model_runner=self.model_runner,
            )
            seq_lens_cpu_backup = batch.seq_lens_cpu
            seq_lens_cpu_live = seq_lens_cpu_backup
            if seq_lens_cpu_live is None:
                # The overlap scheduler keeps the canonical sequence lengths on
                # device and omits the synchronous host mirror.  The first
                # fixed-geometry decode still needs the host values while it
                # binds the target verifier graph; later dense decodes reuse the
                # bound buffers and do not take this copy.
                seq_lens_cpu_live = batch.seq_lens.detach().cpu()
            verify_input = DFlashVerifyInput(
                draft_token=verify_window.verify_ids,
                positions=verify_window.positions,
                draft_token_num=self.verify_width,
                custom_mask=None,
                capture_hidden_mode=CaptureHiddenMode.FULL,
                ragged_verify_layout=layout,
                live_seq_lens_cpu=seq_lens_cpu_live,
            )
            batch.out_cache_loc = verify_window.verify_cache_loc

            seq_lens_sum_backup = batch.seq_lens_sum
            verify_lens_host = torch.tensor(
                verify_lens_cpu, dtype=seq_lens_cpu_live.dtype
            )
            batch.seq_lens_cpu = seq_lens_cpu_live + verify_lens_host
            batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
            verify_forward_batch, can_run_cuda_graph = verify_input.prepare_for_verify(
                batch, self.target_worker
            )
            batch.seq_lens_cpu = seq_lens_cpu_backup
            batch.seq_lens_sum = seq_lens_sum_backup
            if self._native_ragged_verify_graph:
                if bool(can_run_cuda_graph) != exact_graph_geometry:
                    raise RuntimeError(
                        "native ragged graph admission differs from B32/M4"
                    )
            elif exact_graph_geometry:
                if can_run_cuda_graph:
                    raise RuntimeError(
                        "diagonal verifier unexpectedly admitted TARGET_VERIFY"
                    )
                diagonal_geometry = diagonalize_verify_forward_batch(
                    verify_forward_batch,
                    verify_lens_cpu=verify_lens_cpu,
                    prefix_seq_lens=batch.seq_lens,
                    prefix_seq_lens_cpu=seq_lens_cpu_live.tolist(),
                )
                if use_fixed_verify_geometry:
                    fixed_verify_geometry.bind(
                        runner=self.model_runner.decode_cuda_graph_runner,
                        forward_batch=verify_forward_batch,
                        verify_input=verify_input,
                        layout=layout,
                    )
                    verify_forward_batch = fixed_verify_geometry.prepare(
                        batch=batch,
                        proposal=proposal,
                        req_to_token=self.req_to_token_pool.req_to_token,
                    )
                    diagonal_geometry = fixed_verify_geometry.trace_geometry()
                if not self.model_runner.decode_cuda_graph_runner.can_run_graph(
                    verify_forward_batch
                ):
                    raise RuntimeError("diagonal DECODE verifier missed its graph")
            elif can_run_cuda_graph:
                raise RuntimeError(
                    "NEXTLAT eager tier unexpectedly selected a CUDA graph"
                )
            verify_positions = verify_forward_batch.positions
        record_stream_each(
            (batch.input_ids, batch.out_cache_loc, verify_positions), stream
        )
        mark_phase("exact_target_verify_start")
        self._before_target_verify_forward()
        target_output = self.target_worker.forward_batch_generation(
            batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
            # TARGET_VERIFY metadata is stale after diagonalization.  The
            # ordinary-DECODE graph owns its own static input preparation.
            skip_attn_backend_init=(
                None
                if self._diagonal_verify_graph and exact_graph_geometry
                else True
            ),
        )
        self._after_target_verify_forward()
        target_graph_executed = bool(target_output.can_run_cuda_graph)
        if (
            self._target_verify_graph_diagnostic
            and target_graph_executed != exact_graph_geometry
        ):
            raise RuntimeError(
                "NEXTLAT target graph execution differs from admission"
            )
        if exact_graph_geometry:
            graph_replay_record = {
                **(
                    native_replay_snapshot(
                        self.model_runner.decode_cuda_graph_runner
                    )
                    if self._native_ragged_verify_graph
                    else diagonal_replay_snapshot(
                        self.model_runner.decode_cuda_graph_runner
                    )
                ),
                "request_ids": [request.rid for request in batch.reqs],
                "proposal_lens": proposal_lens_cpu,
                "verify_lens": verify_lens_cpu,
            }
            if diagonal_geometry is not None:
                graph_replay_record["diagonal_geometry"] = diagonal_geometry
        logits_output = target_output.logits_output
        if logits_output is None or logits_output.hidden_states is None:
            raise RuntimeError("NEXTLAT target verify returned no full hidden states")
        if logits_output.next_token_logits.shape[0] != total_verify_tokens:
            raise RuntimeError("NEXTLAT target verify returned a non-compact logits layout")
        if exact_graph_geometry:
            if tuple(logits_output.hidden_states.shape) != (
                total_verify_tokens,
                self.contract.hidden_size,
            ):
                raise RuntimeError(
                    "NEXTLAT target graph returned a non-compact hidden layout"
                )
            if (
                logits_output.next_token_logits.ndim != 2
                or logits_output.next_token_logits.shape[0] != total_verify_tokens
                or logits_output.next_token_logits.shape[1]
                < self.contract.vocab_size
            ):
                raise RuntimeError(
                    "NEXTLAT target graph returned an invalid logits buffer"
                )
            if graph_replay_record is None:
                raise RuntimeError("NEXTLAT target graph lost its replay record")
            graph_returned_shapes = {
                "logits": list(logits_output.next_token_logits.shape),
                "hidden_states": list(logits_output.hidden_states.shape),
            }
        mark_phase("exact_target_verify_end")
        trace_this_batch = self._trace_this_batch(batch)
        if self._use_rejection_sampling:
            verified = self._verify_sampled_b1(
                proposal=proposal,
                target_logits=logits_output.next_token_logits,
                sampling_info=batch.sampling_info,
                prefix_seq_lens=batch.seq_lens,
                active_drafts=proposal_lens_cpu[0],
            )
            target_token_ids = (
                torch.argmax(
                    logits_output.next_token_logits[
                        :, : self.contract.vocab_size
                    ],
                    dim=-1,
                ).to(torch.int64)
                if trace_this_batch
                else None
            )
        else:
            target_token_ids = torch.argmax(
                logits_output.next_token_logits[
                    :, : self.contract.vocab_size
                ],
                dim=-1,
            ).to(torch.int64)
            verified = verify_greedy(
                proposal=proposal,
                compact_target_token_ids=target_token_ids,
                prefix_seq_lens=batch.seq_lens,
            )
        if trace_this_batch:
            if target_token_ids is None:
                raise RuntimeError("NEXTLAT diagnostic trace lost target token ids")
            top_rows = self._diagnostic_top_logits(
                logits_output.next_token_logits
            )
            near_tie_projection_probe = near_tie_lm_head_probe(
                raw_logits=logits_output.next_token_logits,
                hidden_states=logits_output.hidden_states,
                lm_head_weight=self.target_model.lm_head.weight,
                vocab_size=self.contract.vocab_size,
            )
            row_starts: list[int] = []
            cursor = 0
            for length in verify_lens_cpu:
                row_starts.append(cursor)
                cursor += length
            accept_lens_cpu = verified.accept_lens.detach().cpu().tolist()
            output_rows = verified.output_tokens.detach().cpu().tolist()
            self._write_diagnostic_trace(
                {
                    "event": "verify",
                    "forward_iter": int(batch.forward_iter),
                    "request_ids": [request.rid for request in batch.reqs],
                    "output_start_indices": [
                        len(request.output_ids) for request in batch.reqs
                    ],
                    "prefix_seq_lens": batch.seq_lens.detach().cpu().tolist(),
                    "proposal_lens": proposal_lens_cpu,
                    "anchor_tokens": proposal.anchor_tokens.detach().cpu().tolist(),
                    "draft_tokens": proposal.draft_tokens.detach().cpu().tolist(),
                    "compact_verify_tokens": proposal.compact_verify_tokens()
                    .detach()
                    .cpu()
                    .tolist(),
                    "target_token_ids": target_token_ids.detach().cpu().tolist(),
                    "target_rows": [
                        top_rows[start : start + length]
                        for start, length in zip(row_starts, verify_lens_cpu)
                    ],
                    "near_tie_lm_head_probe": near_tie_projection_probe,
                    "num_correct_drafts": verified.num_correct_drafts.detach()
                    .cpu()
                    .tolist(),
                    "accept_lens": accept_lens_cpu,
                    "accept_index": verified.accept_index.detach().cpu().tolist(),
                    "output_tokens": [
                        row[:length]
                        for row, length in zip(output_rows, accept_lens_cpu)
                    ],
                    "bonus_tokens": verified.bonus_tokens.detach().cpu().tolist(),
                    "selected_hidden_indices": verified.selected_hidden_indices.detach()
                    .cpu()
                    .tolist(),
                    "new_seq_lens": verified.new_seq_lens.detach().cpu().tolist(),
                }
            )

        if self._zero_copy_linear_kv_commit and use_fixed_verify_geometry:
            # The fixed dense linear verifier writes row j directly to the slot
            # already stored at req_to_token[request, prefix_seq_len + j].
            # Greedy and chain-rejection acceptance can only select the
            # contiguous row prefix [0, k], so those accepted locations are
            # already canonical. Skipping SGLang's mover here avoids a physical
            # self-copy across every target layer.
            #
            # Unaccepted rows remain allocator-owned lookahead slots: a later
            # verify overwrites them, or request teardown frees the reserved
            # suffix.  No req_to_token rewrite or allocator free is valid here
            # because overlap scheduling may already rely on that suffix.
            pass
        else:
            # Canonical fallback for tails and every non-fixed verifier geometry.
            move_accept_tokens_to_target_kvcache(
                batch,
                verified.accept_index,
                verified.num_correct_drafts,
                self.token_to_kv_pool_allocator,
            )
        commit_mamba_states_after_verify(
            self.target_worker,
            batch,
            verified.accept_lens,
            verified.accept_index,
            self.verify_width,
        )
        if on_publish is not None:
            # KV and sequence-length state are now canonical.  Publish before
            # gathering the next recurrent seed so the overlap scheduler can
            # begin its CPU-side work while that GPU work drains.
            on_publish(verified.new_seq_lens)

        next_hidden = logits_output.hidden_states[
            verified.selected_hidden_indices
        ].contiguous()
        next_seed = NextLatDraftInput(
            bonus_tokens=verified.bonus_tokens,
            hidden_states=next_hidden,
            new_seq_lens=verified.new_seq_lens,
        )
        logits_output.hidden_states = None
        result = GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=verified.output_tokens.flatten(),
            # SGLang uses this field for its decode-graph log line and graph
            # pass metric.  Preserve the target worker's executed-path truth;
            # hard-coding False makes a successful diagonal DECODE replay look
            # eager even though the invariant above has already proved it ran.
            can_run_cuda_graph=target_graph_executed,
            speculative_num_draft_tokens=self.verify_width,
            next_draft_input=next_seed,
            accept_lens=verified.accept_lens,
            new_seq_lens=verified.new_seq_lens,
            extra_keep_alive_refs=[verify_forward_batch],
        )
        mark_phase("decode_end")
        if phase_events is not None:
            phase_events["decode_end"].synchronize()
            host_critical_path_ms = (
                time.perf_counter_ns() - phase_host_started_ns
            ) / 1_000_000.0

            # This diagnostic runs strictly after the decode-end event, so its
            # global top-2 read cannot inflate any production phase below.  It
            # classifies every selected exact-target bonus row, independent of
            # whether all drafts were accepted.  It does not call ordinary
            # draft rejection a canonical correction, and does not pretend to
            # time the not-yet-implemented canonical-correction path.
            margin_start = torch.cuda.Event(enable_timing=True)
            margin_end = torch.cuda.Event(enable_timing=True)
            margin_start.record(torch.cuda.current_stream(self.device))
            selected_target_logits = logits_output.next_token_logits[
                verified.selected_hidden_indices,
                : self.contract.vocab_size,
            ]
            target_top2 = torch.topk(selected_target_logits.float(), k=2, dim=-1)
            target_margins = target_top2.values[:, 0] - target_top2.values[:, 1]
            margin_end.record(torch.cuda.current_stream(self.device))
            margin_end.synchronize()
            margin_diagnostic_ms = float(margin_start.elapsed_time(margin_end))
            target_margins_cpu = target_margins.detach().cpu().tolist()
            num_correct_cpu = verified.num_correct_drafts.detach().cpu().tolist()
            accept_lens_cpu = verified.accept_lens.detach().cpu().tolist()
            selected_bonus_low_margin = [
                float(margin) <= _LOW_MARGIN_LOGIT_GAP
                for margin in target_margins_cpu
            ]

            def elapsed_ms(start: str, end: str) -> float:
                return float(phase_events[start].elapsed_time(phase_events[end]))

            step_rows: list[dict[str, Any]] = []
            dynamics_ms = 0.0
            vocab_head_ms = 0.0
            embedding_ms = 0.0
            if not dense_graph_replay:
                for step in range(self.max_draft_tokens):
                    embedding = elapsed_ms(
                        f"step_{step}_embedding_start",
                        f"step_{step}_embedding_end",
                    )
                    dynamics = elapsed_ms(
                        f"step_{step}_dynamics_start",
                        f"step_{step}_dynamics_end",
                    )
                    vocab_head = elapsed_ms(
                        f"step_{step}_vocab_head_start",
                        f"step_{step}_vocab_head_end",
                    )
                    embedding_ms += embedding
                    dynamics_ms += dynamics
                    vocab_head_ms += vocab_head
                    step_rows.append(
                        {
                            "step": step,
                            "embedding_ms": embedding,
                            "recurrent_dynamics_ms": dynamics,
                            "vocab_head_ms": vocab_head,
                        }
                    )

            decode_ms = elapsed_ms("decode_start", "decode_end")
            draft_loop_ms = elapsed_ms("draft_loop_start", "draft_loop_end")
            exact_target_verify_ms = elapsed_ms(
                "exact_target_verify_start", "exact_target_verify_end"
            )
            pre_draft_control_ms = elapsed_ms("decode_start", "draft_loop_start")
            draft_loop_control_ms = (
                0.0
                if dense_graph_replay
                else draft_loop_ms - dynamics_ms - vocab_head_ms
            )
            verify_setup_control_ms = elapsed_ms(
                "draft_loop_end", "exact_target_verify_start"
            )
            post_verify_control_ms = elapsed_ms(
                "exact_target_verify_end", "decode_end"
            )
            draft_accounted_ms = (
                draft_loop_ms
                if dense_graph_replay
                else dynamics_ms + vocab_head_ms
            )
            scheduler_control_ms = (
                decode_ms - draft_accounted_ms - exact_target_verify_ms
            )
            low_margin_selected_bonus_count = sum(selected_bonus_low_margin)
            self._write_phase_trace(
                {
                    "event": "decode_phase_timing",
                    "forward_iter": int(batch.forward_iter),
                    "request_ids": [request.rid for request in batch.reqs],
                    "batch_size": batch_size,
                    "configured_max_drafts": self.max_draft_tokens,
                    "proposal_lens": proposal_lens_cpu,
                    "num_correct_drafts": num_correct_cpu,
                    "accept_lens": accept_lens_cpu,
                    "total_verify_rows": total_verify_tokens,
                    "executed_recurrent_steps": self.max_draft_tokens,
                    "host_critical_path_ms": host_critical_path_ms,
                    "cuda_ms": {
                        "decode_total": decode_ms,
                        "combined_draft_loop": draft_loop_ms,
                        "dense_draft_graph": (
                            draft_loop_ms if dense_graph_replay else None
                        ),
                        "embedding_sum": embedding_ms,
                        "recurrent_dynamics_sum": dynamics_ms,
                        "vocab_head_sum": vocab_head_ms,
                        "exact_target_verify": exact_target_verify_ms,
                        "scheduler_control_residual": scheduler_control_ms,
                        "pre_draft_control": pre_draft_control_ms,
                        "draft_loop_control": draft_loop_control_ms,
                        "verify_setup_control": verify_setup_control_ms,
                        "post_verify_control": post_verify_control_ms,
                    },
                    "draft_breakdown_available": not dense_graph_replay,
                    "per_step_cuda_ms": step_rows,
                    "selected_bonus_margin": {
                        "margin_authority": "exact target selected bonus row",
                        "low_margin_logit_gap_threshold": _LOW_MARGIN_LOGIT_GAP,
                        "selected_target_top1_minus_top2_logit": target_margins_cpu,
                        "request_selected_bonus_low_margin": selected_bonus_low_margin,
                        "low_margin_selected_bonus_count": low_margin_selected_bonus_count,
                        "verify_request_count": batch_size,
                        "excluded_margin_diagnostic_cuda_ms": margin_diagnostic_ms,
                    },
                }
            )
        # Replay evidence materializes CUDA tensors on the CPU.  Skip those
        # copies entirely when no graph-trace sink was requested.
        if (
            self._target_verify_graph_diagnostic
            and self._target_verify_graph_trace_path is not None
        ):
            if exact_graph_geometry:
                if graph_replay_record is None:
                    raise RuntimeError("NEXTLAT graph replay evidence is missing")
                if graph_returned_shapes is None:
                    raise RuntimeError("NEXTLAT graph returned-shape evidence is missing")
                self._target_verify_graph_replay_count += 1
                self._write_target_verify_graph_trace(
                    {
                        "event": "graph_replay",
                        "graph_replay_count": self._target_verify_graph_replay_count,
                        "forward_iter": int(batch.forward_iter),
                        **graph_replay_record,
                        "graph_admitted": True,
                        "graph_executed": True,
                        "returned_logits_shape": graph_returned_shapes["logits"],
                        "returned_hidden_states_shape": graph_returned_shapes[
                            "hidden_states"
                        ],
                        "num_correct_drafts": (
                            verified.num_correct_drafts.detach().cpu().tolist()
                        ),
                        "accept_lens": verified.accept_lens.detach().cpu().tolist(),
                    }
                )
            else:
                self._target_verify_graph_tail_eager_count += 1
                self._write_target_verify_graph_trace(
                    {
                        "event": "tail_eager",
                        "tail_eager_count": self._target_verify_graph_tail_eager_count,
                        "forward_iter": int(batch.forward_iter),
                        "request_ids": [request.rid for request in batch.reqs],
                        "proposal_lens": proposal_lens_cpu,
                        "verify_lens": verify_lens_cpu,
                        "total_verify_rows": total_verify_tokens,
                        "graph_admitted": False,
                        "graph_executed": False,
                        "excluded_from_timing": True,
                    }
                )
        if confidence_this_batch:
            if (
                len(draft_top1_logprob) != self.max_draft_tokens
                or len(cumulative_draft_logprob) != self.max_draft_tokens
            ):
                raise RuntimeError("NEXTLAT confidence observer missed a recurrent step")
            trace_copy_started_ns = time.perf_counter_ns()
            raw_scores = torch.stack(draft_top1_logprob, dim=1)
            cumulative_scores = torch.stack(cumulative_draft_logprob, dim=1)
            if raw_scores.shape != proposal.draft_tokens.shape:
                raise RuntimeError("NEXTLAT confidence score shape differs from drafts")
            confidence_payload = {
                "event": "draft_confidence",
                "forward_iter": int(batch.forward_iter),
                "request_ids": [request.rid for request in batch.reqs],
                "output_start_indices": [
                    len(request.output_ids) for request in batch.reqs
                ],
                "proposal_lens": proposal_lens_cpu,
                "draft_tokens": proposal.draft_tokens.detach().cpu().tolist(),
                "draft_top1_logprob": raw_scores.detach().cpu().tolist(),
                "cumulative_draft_logprob": cumulative_scores.detach()
                .cpu()
                .tolist(),
                "num_correct_drafts": verified.num_correct_drafts.detach()
                .cpu()
                .tolist(),
                "accept_lens": verified.accept_lens.detach().cpu().tolist(),
            }
            trace_copy_serialize_host_ms = (
                time.perf_counter_ns() - trace_copy_started_ns
            ) / 1_000_000.0
            primary_event_write_host_ms = self._write_confidence_trace(
                confidence_payload
            )
            self._write_confidence_trace(
                {
                    "event": "trace_host_overhead",
                    "forward_iter": int(batch.forward_iter),
                    "request_ids": [request.rid for request in batch.reqs],
                    "copy_and_python_materialization_host_ms": trace_copy_serialize_host_ms,
                    "primary_event_json_and_write_host_ms": primary_event_write_host_ms,
                    "scope": (
                        "diagnostic-only D2H/Python/JSONL publication after decode_end; "
                        "the small overhead-record append itself is excluded"
                    ),
                }
            )
        return result

    def forward_batch_generation(
        self, batch, on_publish=None, grammar_barrier=None
    ):
        if grammar_barrier is not None:
            raise RuntimeError("NEXTLAT fixed tier does not admit a grammar barrier")
        self._validate_batch_contract(batch)
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            return self._prefill(batch, on_publish)
        if batch.forward_mode.is_idle():
            return self._idle_result(on_publish)
        return self._decode(batch, on_publish)
