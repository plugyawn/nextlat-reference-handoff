"""Atomically derive a strict NextLat sidecar from an admitted training source.

The qualified Megatron TP1 export and the verified PyTorch joint checkpoint are
distinct authorities.  Neither path can impersonate the other.  Both rehash
their complete source payload, bind the executable HF target manifest, validate
the exact FP32 recurrent tensor roster, and atomically publish the same narrow
four-file serving sidecar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from safetensors import safe_open

from .admission import (
    admit_target_checkpoint,
    canonical_json_bytes,
    target_checkpoint_manifest_sha256,
)
from .checkpoint import (
    PYTORCH_CHECKPOINT_AUTHORITY,
    PYTORCH_SIDECAR_SCHEMA_VERSION,
    SIDECAR_SCHEMA_VERSION,
    read_sidecar_metadata,
)
from .constants import NextLatContract, QWEN3_1_7B_CONTRACT
from .dynamics import expected_state_shapes

EXPORT_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "result",
        "distributed_checkpoint",
        "output",
        "topology",
        "qualification",
        "provenance_paths",
        "runtime",
        "continuation",
        "fp32_parameter_extraction",
        "hf_serialization",
        "hf_collateral_sha256",
        "payload_sha256_before_receipt",
    }
)
TP1_TOPOLOGY = {
    "world_size": 1,
    "tensor_parallel_size": 1,
    "pipeline_parallel_size": 1,
    "data_parallel_size": 1,
    "context_parallel_size": 1,
    "sequence_parallel": False,
}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_RUNTIME_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_WANDB_RUN_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
_WANDB_ARTIFACT_KEYS = frozenset(
    {"artifact_digest", "artifact_name", "artifact_url", "artifact_version"}
)


@dataclass(frozen=True)
class AdmittedExport:
    root: Path
    model_root: Path
    dynamics_path: Path
    export_manifest_sha256: str
    export_receipt_sha256: str
    target_checkpoint_sha256: str
    dynamics_sha256: str
    lineage: dict[str, Any]

    @property
    def sidecar_schema_version(self) -> int:
        return SIDECAR_SCHEMA_VERSION

    @property
    def lineage_key(self) -> str:
        return "export_lineage"

    @property
    def authority_label(self) -> str:
        return "qualified Megatron TP1 export"


@dataclass(frozen=True)
class AdmittedPytorchCheckpoint:
    root: Path
    model_root: Path
    dynamics_path: Path
    checkpoint_manifest_sha256: str
    checkpoint_integrity_sha256: str
    target_checkpoint_sha256: str
    dynamics_sha256: str
    lineage: dict[str, Any]

    @property
    def sidecar_schema_version(self) -> int:
        return PYTORCH_SIDECAR_SCHEMA_VERSION

    @property
    def lineage_key(self) -> str:
        return "source_lineage"

    @property
    def authority_label(self) -> str:
        return "verified PyTorch joint checkpoint"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    def exact_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=exact_object
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"cannot read exact {label}: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise RuntimeError(f"{label} must be a lowercase SHA256")
    return value


def _regular_files(root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in directory_names:
            path = directory_path / name
            if path.is_symlink():
                raise RuntimeError(f"export contains a symlinked directory: {path}")
        for name in file_names:
            path = directory_path / name
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f"export payload is not a regular file: {path}")
            files[path.relative_to(root).as_posix()] = path
    return files


def _validate_hash_manifest(
    value: Any, actual: Mapping[str, Path]
) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise RuntimeError("export sha256.json must be a non-empty object")
    normalized: dict[str, str] = {}
    for relative, digest in value.items():
        if not isinstance(relative, str):
            raise RuntimeError("export manifest paths must be strings")
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != relative:
            raise RuntimeError(f"unsafe export manifest path: {relative!r}")
        normalized[relative] = _require_sha256(
            digest, label=f"export payload {relative} digest"
        )
    if set(normalized) != set(actual):
        raise RuntimeError(
            "export manifest file roster differs: "
            f"missing={sorted(set(normalized) - set(actual))}, "
            f"extra={sorted(set(actual) - set(normalized))}"
        )
    for relative, path in actual.items():
        if _sha256(path) != normalized[relative]:
            raise RuntimeError(f"export payload SHA256 differs: {relative}")
    return normalized


def _nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"{label} must be a non-negative integer")
    return value


def _validate_dynamics_metadata(path: Path, contract: NextLatContract) -> None:
    expected = expected_state_shapes(
        contract.hidden_size, projection_factor=contract.projection_factor
    )
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            if set(handle.keys()) != set(expected):
                raise RuntimeError("export dynamics tensor roster differs")
            for name, shape in expected.items():
                value = handle.get_slice(name)
                if value.get_dtype() != "F32" or tuple(value.get_shape()) != shape:
                    raise RuntimeError(
                        f"export dynamics tensor metadata differs for {name}"
                    )
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"cannot inspect export dynamics: {path}") from exc


def _infer_pytorch_branch(train_state: Mapping[str, Any]) -> str | None:
    explicit = train_state.get("branch")
    if explicit is not None:
        if explicit not in {"d4", "d8", "d16"}:
            raise RuntimeError("PyTorch checkpoint branch must be d4, d8, or d16")
        return explicit

    resume_kind_branch = (
        "d4"
        if train_state.get("resume_kind") in {"joint_d4_systems_fork", "joint_d4_fresh_data_pilot"}
        else None
    )
    continuation = train_state.get("continuation")
    if continuation is None:
        return resume_kind_branch
    if not isinstance(continuation, dict):
        raise RuntimeError("PyTorch checkpoint continuation must be an object")
    schedules = []
    for name in ("last_update_schedule", "next_update_schedule"):
        schedule = continuation.get(name)
        if not isinstance(schedule, dict):
            raise RuntimeError(
                f"PyTorch checkpoint continuation {name} must be an object"
            )
        horizon = schedule.get("active_horizon")
        if isinstance(horizon, bool) or not isinstance(horizon, int):
            raise RuntimeError(
                f"PyTorch checkpoint continuation {name} active_horizon is invalid"
            )
        schedules.append(horizon)
    # The checkpoint contains weights from ``last_update_schedule``.  At an
    # anneal boundary, ``next_update_schedule`` intentionally names the next
    # objective (for example d16 immediately after the final uniform-d8
    # update), so requiring the two horizons to match rejects the exact
    # boundary checkpoint we need to serve.
    completed_horizon = schedules[0]
    if completed_horizon not in {4, 8, 16}:
        raise RuntimeError(
            "PyTorch checkpoint completed-update horizon cannot identify "
            "d4, d8, or d16"
        )
    completed_branch = f"d{completed_horizon}"
    if resume_kind_branch is not None and completed_branch != resume_kind_branch:
        raise RuntimeError(
            "PyTorch checkpoint resume_kind branch differs from "
            "completed-update horizon"
        )
    return completed_branch


def _validate_wandb_checkpoint_artifact(value: Any) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != _WANDB_ARTIFACT_KEYS:
        raise RuntimeError("PyTorch W&B checkpoint artifact key roster differs")
    normalized: dict[str, str] = {}
    for name in sorted(_WANDB_ARTIFACT_KEYS):
        observed = value[name]
        if (
            not isinstance(observed, str)
            or not observed
            or observed != observed.strip()
            or len(observed) > 2048
        ):
            raise RuntimeError(f"PyTorch W&B checkpoint artifact {name} is invalid")
        normalized[name] = observed
    return normalized


def _checkpoint_integrity_sha256(
    hashes: Mapping[str, str], files: Mapping[str, Path]
) -> str:
    manifest = {
        "schema_version": 1,
        "files": [
            {
                "path": relative,
                "size": files[relative].stat().st_size,
                "sha256": hashes[relative],
            }
            for relative in sorted(hashes)
        ],
    }
    return hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()


def admit_pytorch_checkpoint(
    root: str | Path,
    *,
    expected_checkpoint_manifest_sha256: str,
    contract: NextLatContract = QWEN3_1_7B_CONTRACT,
) -> AdmittedPytorchCheckpoint:
    """Admit one exact ``nextlat.run_utils`` joint checkpoint.

    This is deliberately not a TP1 qualification.  It verifies the atomic
    marker, the exact payload roster and every source hash before extracting a
    separate PyTorch authority record for serving.
    """

    expected_manifest_sha256 = _require_sha256(
        expected_checkpoint_manifest_sha256,
        label="expected PyTorch checkpoint manifest digest",
    )
    raw_root = Path(root).expanduser()
    if raw_root.is_symlink():
        raise RuntimeError(f"PyTorch checkpoint root must not be a symlink: {raw_root}")
    checkpoint_root = raw_root.resolve()
    if not checkpoint_root.is_dir():
        raise FileNotFoundError(
            f"PyTorch checkpoint root is missing: {checkpoint_root}"
        )
    complete = checkpoint_root / "COMPLETE"
    if not complete.is_file() or complete.is_symlink() or complete.read_text() != "ok\n":
        raise RuntimeError("PyTorch checkpoint COMPLETE marker differs")
    if (checkpoint_root / "INCOMPLETE").exists():
        raise RuntimeError("PyTorch checkpoint is marked incomplete")

    manifest_path = checkpoint_root / "sha256.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise RuntimeError(
            "PyTorch checkpoint manifest must be a regular file: "
            f"{manifest_path}"
        )
    manifest_sha256 = _sha256(manifest_path)
    if manifest_sha256 != expected_manifest_sha256:
        raise RuntimeError(
            "PyTorch checkpoint manifest SHA256 differs: "
            f"{manifest_sha256} != {expected_manifest_sha256}"
        )

    all_files = _regular_files(checkpoint_root)
    payload_files = {
        name: path
        for name, path in all_files.items()
        if name not in {"COMPLETE", "sha256.json"}
    }
    hashes = _validate_hash_manifest(
        _read_json(manifest_path, label="PyTorch checkpoint manifest"), payload_files
    )
    train_state_path = checkpoint_root / "train_state.json"
    train_state = _read_json(train_state_path, label="PyTorch checkpoint train state")

    optimizer_field = train_state.get("optimizer_included")
    if optimizer_field is None:
        optimizer_included = "optimizer.pt" in payload_files
    elif isinstance(optimizer_field, bool):
        optimizer_included = optimizer_field
    else:
        raise RuntimeError("PyTorch optimizer_included must be a boolean when present")
    if optimizer_included != ("optimizer.pt" in payload_files):
        raise RuntimeError("PyTorch optimizer presence differs from train_state")

    rng_field = train_state.get("rng_state_included")
    if rng_field is None:
        rng_included = False
    elif isinstance(rng_field, bool):
        rng_included = rng_field
    else:
        raise RuntimeError("PyTorch rng_state_included must be a boolean when present")
    if rng_included != ("rng_state.pt" in payload_files):
        raise RuntimeError("PyTorch RNG-state presence differs from train_state")

    expected_top_level = {
        "COMPLETE",
        "sha256.json",
        "model",
        "dynamics.safetensors",
        "train_state.json",
    }
    if optimizer_included:
        expected_top_level.add("optimizer.pt")
    if rng_included:
        expected_top_level.add("rng_state.pt")
    observed_top_level = {path.name for path in checkpoint_root.iterdir()}
    if observed_top_level != expected_top_level:
        raise RuntimeError(
            "PyTorch checkpoint top-level roster differs: "
            f"missing={sorted(expected_top_level - observed_top_level)}, "
            f"extra={sorted(observed_top_level - expected_top_level)}"
        )
    model_root = checkpoint_root / "model"
    if not model_root.is_dir() or model_root.is_symlink():
        raise RuntimeError("PyTorch checkpoint model must be a regular directory")
    for entry in model_root.iterdir():
        if not entry.is_file() or entry.is_symlink():
            raise RuntimeError(
                f"PyTorch checkpoint model payload must be flat regular files: {entry}"
            )

    tokens_seen = _nonnegative_int(
        train_state.get("tokens_seen"), label="PyTorch tokens_seen"
    )
    update = _nonnegative_int(train_state.get("update"), label="PyTorch update")
    source_code_commit = train_state.get("source_code_commit")
    if not isinstance(source_code_commit, str) or _GIT_COMMIT.fullmatch(
        source_code_commit
    ) is None:
        raise RuntimeError("PyTorch source_code_commit must be a lowercase Git SHA1")
    branch = _infer_pytorch_branch(train_state)
    wandb_run_id = train_state.get("wandb_run_id")
    if wandb_run_id is not None and (
        not isinstance(wandb_run_id, str)
        or _WANDB_RUN_ID.fullmatch(wandb_run_id) is None
    ):
        raise RuntimeError("PyTorch wandb_run_id is invalid")
    wandb_artifact = _validate_wandb_checkpoint_artifact(
        train_state.get("wandb_checkpoint_artifact")
    )

    target_manifest = target_checkpoint_manifest_sha256(model_root, contract=contract)
    target = admit_target_checkpoint(
        model_root,
        expected_manifest_sha256=target_manifest,
        contract=contract,
    )
    for name, _size, digest in target.payloads:
        if hashes.get(f"model/{name}") != digest:
            raise RuntimeError(f"target model payload escaped checkpoint manifest: {name}")
    if target.manifest_sha256 != target_manifest:
        raise RuntimeError("target checkpoint manifest changed during PyTorch admission")

    dynamics_path = checkpoint_root / "dynamics.safetensors"
    if hashes.get("dynamics.safetensors") != _sha256(dynamics_path):
        raise RuntimeError("PyTorch dynamics escaped the checkpoint manifest")
    _validate_dynamics_metadata(dynamics_path, contract)

    if _sha256(manifest_path) != expected_manifest_sha256:
        raise RuntimeError("PyTorch checkpoint manifest changed during admission")
    integrity_sha256 = _checkpoint_integrity_sha256(hashes, payload_files)
    lineage = {
        "authority": PYTORCH_CHECKPOINT_AUTHORITY,
        "branch": branch,
        "checkpoint_integrity_sha256": integrity_sha256,
        "checkpoint_manifest_sha256": manifest_sha256,
        "checkpoint_name": checkpoint_root.name,
        "optimizer_included": optimizer_included,
        "schema_version": 1,
        "source_code_commit": source_code_commit,
        "tokens_seen": tokens_seen,
        "train_state_sha256": hashes["train_state.json"],
        "update": update,
        "wandb": {
            "checkpoint_artifact": wandb_artifact,
            "run_id": wandb_run_id,
        },
    }
    return AdmittedPytorchCheckpoint(
        root=checkpoint_root,
        model_root=model_root,
        dynamics_path=dynamics_path,
        checkpoint_manifest_sha256=manifest_sha256,
        checkpoint_integrity_sha256=integrity_sha256,
        target_checkpoint_sha256=target_manifest,
        dynamics_sha256=hashes["dynamics.safetensors"],
        lineage=lineage,
    )


def admit_tp1_export(
    root: str | Path,
    *,
    contract: NextLatContract = QWEN3_1_7B_CONTRACT,
) -> AdmittedExport:
    raw_root = Path(root).expanduser()
    if raw_root.is_symlink():
        raise RuntimeError(f"export root must not be a symlink: {raw_root}")
    export_root = raw_root.resolve()
    if not export_root.is_dir():
        raise FileNotFoundError(f"TP1 export root is missing: {export_root}")
    complete = export_root / "COMPLETE"
    if not complete.is_file() or complete.is_symlink() or complete.read_text() != "ok\n":
        raise RuntimeError("TP1 export COMPLETE marker differs")
    if (export_root / "INCOMPLETE").exists():
        raise RuntimeError("TP1 export is marked incomplete")

    all_files = _regular_files(export_root)
    payload_files = {
        name: path
        for name, path in all_files.items()
        if name not in {"COMPLETE", "sha256.json"}
    }
    manifest_path = export_root / "sha256.json"
    hashes = _validate_hash_manifest(
        _read_json(manifest_path, label="export manifest"), payload_files
    )
    receipt_path = export_root / "export_receipt.json"
    receipt = _read_json(receipt_path, label="export receipt")
    if set(receipt) != EXPORT_RECEIPT_KEYS:
        raise RuntimeError("export receipt key roster differs from export_tp1 schema v2")
    if receipt["schema_version"] != 2 or receipt["result"] != "pass":
        raise RuntimeError("export receipt is not FP32 schema v2 with result=pass")
    if receipt["topology"] != TP1_TOPOLOGY:
        raise RuntimeError("export topology is not strict TP1")
    output = receipt["output"]
    if (
        not isinstance(output, str)
        or not Path(output).is_absolute()
        or Path(output).resolve() != export_root
    ):
        raise RuntimeError("export receipt names a different output root")
    expected_pre_receipt = {
        name: digest for name, digest in hashes.items() if name != "export_receipt.json"
    }
    if receipt["payload_sha256_before_receipt"] != expected_pre_receipt:
        raise RuntimeError("export pre-receipt payload hashes differ")
    extraction = receipt["fp32_parameter_extraction"]
    if (
        not isinstance(extraction, dict)
        or extraction.get("value_authority")
        != "model.fp32_parameters"
        or extraction.get("model_parameter_value_reads")
        != extraction.get("destination_parameters")
        or extraction.get("all_output_cpu_fp32") is not True
        or extraction.get("tied_embedding") is not True
    ):
        raise RuntimeError("export did not preserve strict live FP32 parameter authority")
    continuation = receipt["continuation"]
    if not isinstance(continuation, dict) or _read_json(
        export_root / "train_state.json", label="export train state"
    ) != continuation:
        raise RuntimeError("export train_state differs from its receipt")
    tokens_seen = _nonnegative_int(
        continuation.get("tokens_seen"), label="continuation tokens_seen"
    )
    update = _nonnegative_int(continuation.get("update"), label="continuation update")

    qualification = receipt["qualification"]
    if not isinstance(qualification, dict):
        raise RuntimeError("export qualification provenance must be an object")
    qualification_receipt_sha256 = _require_sha256(
        qualification.get("qualification_receipt_sha256"),
        label="qualification receipt digest",
    )
    parent_manifest_sha256 = _require_sha256(
        qualification.get("qualified_parent_root_manifest_sha256"),
        label="qualified parent manifest digest",
    )
    checkpoint_manifest_sha256 = _require_sha256(
        qualification.get("checkpoint_root_manifest_sha256"),
        label="checkpoint root manifest digest",
    )
    checkpoint_integrity_sha256 = _require_sha256(
        qualification.get("checkpoint_integrity_sha256"),
        label="checkpoint integrity digest",
    )
    source_receipt_sha256 = _require_sha256(
        qualification.get("source_checkpoint_receipt_sha256"),
        label="source checkpoint receipt digest",
    )
    source_lock_sha256 = _require_sha256(
        qualification.get("source_lock_sha256"), label="source lock digest"
    )
    runtime_digest = qualification.get("runtime_image_digest")
    if not isinstance(runtime_digest, str) or _RUNTIME_DIGEST.fullmatch(runtime_digest) is None:
        raise RuntimeError("qualification runtime image digest differs")

    bindings = {
        "qualification_receipt_sha256": qualification_receipt_sha256,
        "qualified_parent_root_manifest_sha256": parent_manifest_sha256,
        "source_checkpoint_receipt_sha256": source_receipt_sha256,
        "source_lock_sha256": source_lock_sha256,
        "runtime_image_digest": runtime_digest,
    }
    mismatches = {
        name: {"observed": continuation[name], "expected": expected}
        for name, expected in bindings.items()
        if name in continuation and continuation[name] != expected
    }
    if mismatches:
        raise RuntimeError(f"target/export continuation lineage differs: {mismatches}")
    runtime = receipt["runtime"]
    if (
        not isinstance(runtime, dict)
        or runtime.get("runtime_image_digest") != runtime_digest
    ):
        raise RuntimeError("TP1 export runtime differs from qualification lineage")
    branch = continuation.get("branch")
    if branch not in (None, "d4", "d8", "d16"):
        raise RuntimeError("continuation branch must be absent, d4, d8, or d16")

    model_root = export_root / "model"
    target_manifest = target_checkpoint_manifest_sha256(
        model_root, contract=contract
    )
    # Every executable target byte must be covered by export sha256.json.
    target = admit_target_checkpoint(
        model_root,
        expected_manifest_sha256=target_manifest,
        contract=contract,
    )
    for name, _size, digest in target.payloads:
        if hashes.get(f"model/{name}") != digest:
            raise RuntimeError(f"target model payload escaped export manifest: {name}")
    if target.manifest_sha256 != target_manifest:
        raise RuntimeError("target checkpoint manifest changed during export admission")

    serialization = receipt["hf_serialization"]
    if not isinstance(serialization, dict):
        raise RuntimeError("export hf_serialization must be an object")
    serialized_files = set(serialization.get("shards", []))
    if serialization.get("index") is not None:
        serialized_files.add(serialization["index"])
    target_weight_files = {
        name for name, _size, _digest in target.payloads if name != "config.json"
    }
    if serialized_files != target_weight_files:
        raise RuntimeError("target weight files differ from HF serialization receipt")

    dynamics_path = export_root / "dynamics.safetensors"
    if hashes.get("dynamics.safetensors") != _sha256(dynamics_path):
        raise RuntimeError("export dynamics escaped the root manifest")
    _validate_dynamics_metadata(dynamics_path, contract)
    return AdmittedExport(
        root=export_root,
        model_root=model_root,
        dynamics_path=dynamics_path,
        export_manifest_sha256=_sha256(manifest_path),
        export_receipt_sha256=hashes["export_receipt.json"],
        target_checkpoint_sha256=target_manifest,
        dynamics_sha256=hashes["dynamics.safetensors"],
        lineage={
            "branch": branch,
            "checkpoint_integrity_sha256": checkpoint_integrity_sha256,
            "checkpoint_root_manifest_sha256": checkpoint_manifest_sha256,
            "export_manifest_sha256": _sha256(manifest_path),
            "export_receipt_sha256": hashes["export_receipt.json"],
            "qualification_receipt_sha256": qualification_receipt_sha256,
            "runtime_image_digest": runtime_digest,
            "source_checkpoint_receipt_sha256": source_receipt_sha256,
            "source_lock_sha256": source_lock_sha256,
            "tokens_seen": tokens_seen,
            "update": update,
        },
    )


def _write_exclusive(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _build_admitted_sidecar(
    admitted: AdmittedExport | AdmittedPytorchCheckpoint,
    output: str | Path,
    *,
    contract: NextLatContract = QWEN3_1_7B_CONTRACT,
) -> Path:
    output_arg = Path(output).expanduser()
    parent = output_arg.parent.resolve()
    if not parent.is_dir() or not output_arg.name:
        raise ValueError("sidecar output must have an existing parent and a name")
    final = parent / output_arg.name
    partial = parent / f".{output_arg.name}.partial"
    if final.exists():
        raise FileExistsError(f"refusing to overwrite sidecar: {final}")
    if partial.exists():
        raise FileExistsError(f"stale sidecar partial must be inspected: {partial}")
    partial.mkdir(mode=0o700, parents=False, exist_ok=False)
    _write_exclusive(partial / "INCOMPLETE", b"sidecar build in progress\n")

    dynamics_destination = partial / "dynamics.safetensors"
    if _sha256(admitted.dynamics_path) != admitted.dynamics_sha256:
        raise RuntimeError(
            f"admitted {admitted.authority_label} dynamics changed before copy"
        )
    shutil.copyfile(admitted.dynamics_path, dynamics_destination)
    with dynamics_destination.open("rb") as handle:
        os.fsync(handle.fileno())
    dynamics_sha256 = _sha256(dynamics_destination)
    if dynamics_sha256 != admitted.dynamics_sha256:
        raise RuntimeError(
            f"copied dynamics differs from admitted {admitted.authority_label}"
        )
    target_sha256 = target_checkpoint_manifest_sha256(
        admitted.model_root, contract=contract
    )
    if target_sha256 != admitted.target_checkpoint_sha256:
        raise RuntimeError(
            f"admitted {admitted.authority_label} target changed before publication"
        )
    config = {
        "architectures": ["Qwen3ForCausalLM"],
        "hidden_size": contract.hidden_size,
        "model_type": "qwen3",
        "tie_word_embeddings": True,
        "torch_dtype": contract.target_runtime_dtype,
        "vocab_size": contract.vocab_size,
        "nextlat": {
            "dropout": contract.dropout,
            "dynamics_dtype": contract.dynamics_dtype,
            "dynamics_file": "dynamics.safetensors",
            "dynamics_sha256": dynamics_sha256,
            "hidden_state_convention": contract.hidden_state_convention,
            "projection_factor": contract.projection_factor,
            "schema_version": admitted.sidecar_schema_version,
            "sglang_commit": contract.sglang_commit,
            "target_base_revision": contract.target_base_revision,
            "target_checkpoint_sha256": target_sha256,
            "target_model_id": contract.target_model_id,
        },
    }
    config["nextlat"][admitted.lineage_key] = admitted.lineage
    _write_exclusive(partial / "config.json", canonical_json_bytes(config, pretty=True))
    hashes = {
        "config.json": _sha256(partial / "config.json"),
        "dynamics.safetensors": dynamics_sha256,
    }
    _write_exclusive(
        partial / "sha256.json", canonical_json_bytes(hashes, pretty=True)
    )
    _write_exclusive(partial / "COMPLETE", b"ok\n")
    (partial / "INCOMPLETE").unlink()
    read_sidecar_metadata(partial, _contract=contract)
    os.rename(partial, final)
    directory_fd = os.open(parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return final


def build_sidecar(
    export_root: str | Path,
    output: str | Path,
    *,
    contract: NextLatContract = QWEN3_1_7B_CONTRACT,
) -> Path:
    """Build from the existing qualified Megatron TP1 export authority."""

    admitted = admit_tp1_export(export_root, contract=contract)
    return _build_admitted_sidecar(admitted, output, contract=contract)


def build_sidecar_from_pytorch_checkpoint(
    checkpoint_root: str | Path,
    output: str | Path,
    *,
    expected_checkpoint_manifest_sha256: str,
    contract: NextLatContract = QWEN3_1_7B_CONTRACT,
) -> Path:
    """Build from a verified joint PyTorch checkpoint without TP1 claims."""

    admitted = admit_pytorch_checkpoint(
        checkpoint_root,
        expected_checkpoint_manifest_sha256=expected_checkpoint_manifest_sha256,
        contract=contract,
    )
    return _build_admitted_sidecar(admitted, output, contract=contract)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--export",
        type=Path,
        help="completed qualified Megatron TP1 export from systems/export_tp1.py",
    )
    source.add_argument(
        "--pytorch-checkpoint",
        type=Path,
        help="completed verified joint PyTorch checkpoint from nextlat.run_utils",
    )
    parser.add_argument(
        "--expected-checkpoint-manifest-sha256",
        help="trusted lowercase SHA256 of the source checkpoint's literal sha256.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.export is not None:
        if args.expected_checkpoint_manifest_sha256 is not None:
            parser.error(
                "--expected-checkpoint-manifest-sha256 is only valid with "
                "--pytorch-checkpoint"
            )
        result = build_sidecar(args.export, args.output)
    else:
        if args.expected_checkpoint_manifest_sha256 is None:
            parser.error(
                "--expected-checkpoint-manifest-sha256 is required with "
                "--pytorch-checkpoint"
            )
        result = build_sidecar_from_pytorch_checkpoint(
            args.pytorch_checkpoint,
            args.output,
            expected_checkpoint_manifest_sha256=(
                args.expected_checkpoint_manifest_sha256
            ),
        )
    print(result)


if __name__ == "__main__":
    main()
