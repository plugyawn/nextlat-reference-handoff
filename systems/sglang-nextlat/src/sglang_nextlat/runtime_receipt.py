"""Deterministic admission and smoke receipts for the fixed serving tier."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Mapping

from .admission import (
    SGLangSourceAdmission,
    TargetCheckpointAdmission,
    canonical_json_bytes,
)
from .checkpoint import (
    NextLatPytorchCheckpointLineage,
    NextLatSidecarMetadata,
)
from .constants import (
    FIXED_TOPK,
    MAX_DRAFT_TOKENS,
)

RUNTIME_RECEIPT_SCHEMA_VERSION = 2


def build_runtime_receipt(
    *,
    sidecar: NextLatSidecarMetadata,
    target: TargetCheckpointAdmission,
    source: SGLangSourceAdmission,
    evidence: Mapping[str, Any],
    max_draft_tokens: int,
) -> dict[str, Any]:
    """Build a time-free receipt; identical evidence produces identical bytes."""

    if not 1 <= max_draft_tokens <= MAX_DRAFT_TOKENS:
        raise ValueError(
            f"max_draft_tokens must be in [1, {MAX_DRAFT_TOKENS}]"
        )

    lineage = sidecar.source_lineage
    pytorch_source = isinstance(lineage, NextLatPytorchCheckpointLineage)
    payload: dict[str, Any] = {
        "schema_version": 3 if pytorch_source else RUNTIME_RECEIPT_SCHEMA_VERSION,
        "algorithm": "NEXTLAT",
        "fixed_tier": {
            "topk": FIXED_TOPK,
            "configured_max_draft_tokens": max_draft_tokens,
            "verify_stride": max_draft_tokens + 1,
            "per_request_draft_lengths": {
                "minimum": 0,
                "maximum": max_draft_tokens,
                "layout": "compact_ragged",
            },
            "adaptive": False,
            "overlap": False,
            "cuda_graphs": False,
        },
        "runtime_dtypes": {
            "dynamics_source": sidecar.contract.dynamics_dtype,
            "dynamics_runtime": sidecar.contract.target_runtime_dtype,
            "target_source": sidecar.contract.target_source_dtype,
            "target": sidecar.contract.target_runtime_dtype,
        },
        "target": {
            "model_id": sidecar.contract.target_model_id,
            "base_revision": sidecar.contract.target_base_revision,
            "manifest_sha256": target.manifest_sha256,
            "payloads": [
                {"path": name, "size": size, "sha256": digest}
                for name, size, digest in target.payloads
            ],
        },
        "dynamics": {
            "sha256": sidecar.dynamics_sha256,
            "expected_parameter_count": sidecar.expected_parameter_count,
            "hidden_state_convention": sidecar.contract.hidden_state_convention,
        },
        "sglang": {
            "commit": sidecar.contract.sglang_commit,
            "source_receipt_sha256": source.receipt_sha256,
            "audited_file_count": len(source.audited_files),
        },
        "evidence": dict(evidence),
    }
    if pytorch_source:
        payload["source_lineage"] = {
            "authority": lineage.authority,
            "branch": lineage.branch,
            "checkpoint_integrity_sha256": lineage.checkpoint_integrity_sha256,
            "checkpoint_manifest_sha256": lineage.checkpoint_manifest_sha256,
            "checkpoint_name": lineage.checkpoint_name,
            "optimizer_included": lineage.optimizer_included,
            "schema_version": 1,
            "source_code_commit": lineage.source_code_commit,
            "tokens_seen": lineage.tokens_seen,
            "train_state_sha256": lineage.train_state_sha256,
            "update": lineage.update,
            "wandb": {
                "checkpoint_artifact": lineage.wandb_checkpoint_artifact,
                "run_id": lineage.wandb_run_id,
            },
        }
    else:
        payload["export_lineage"] = {
            "branch": lineage.branch,
            "checkpoint_integrity_sha256": lineage.checkpoint_integrity_sha256,
            "checkpoint_root_manifest_sha256": (
                lineage.checkpoint_root_manifest_sha256
            ),
            "export_manifest_sha256": lineage.export_manifest_sha256,
            "export_receipt_sha256": lineage.export_receipt_sha256,
            "qualification_receipt_sha256": lineage.qualification_receipt_sha256,
            "runtime_image_digest": lineage.runtime_image_digest,
            "source_checkpoint_receipt_sha256": (
                lineage.source_checkpoint_receipt_sha256
            ),
            "source_lock_sha256": lineage.source_lock_sha256,
            "tokens_seen": lineage.tokens_seen,
            "update": lineage.update,
        }
    receipt_sha256 = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return {**payload, "receipt_sha256": receipt_sha256}


def write_runtime_receipt(path: str | Path, receipt: Mapping[str, Any]) -> Path:
    """Create a receipt atomically; only an identical rerun may reuse the path."""

    destination = Path(path).expanduser()
    if destination.is_symlink():
        raise ValueError(f"runtime receipt path must not be a symlink: {destination}")
    parent = destination.parent.resolve()
    if not parent.is_dir():
        raise ValueError(f"runtime receipt parent does not exist: {parent}")
    destination = parent / destination.name
    encoded = canonical_json_bytes(dict(receipt), pretty=True)
    if destination.exists():
        if not destination.is_file() or destination.is_symlink():
            raise ValueError(f"runtime receipt destination is not a regular file: {destination}")
        if destination.read_bytes() != encoded:
            raise FileExistsError(
                f"refusing to overwrite a different runtime receipt: {destination}"
            )
        return destination

    temporary = parent / f".{destination.name}.{os.getpid()}.partial"
    if temporary.exists():
        raise FileExistsError(f"stale runtime receipt partial exists: {temporary}")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination
