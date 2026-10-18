"""Receipt-bound loader for the exact NextLat dynamics sidecar."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from .constants import NextLatContract, scaling_contract
from .dynamics import NextLatDynamics, projection_size, validate_state_dict

SIDECAR_SCHEMA_VERSION = 1
PYTORCH_SIDECAR_SCHEMA_VERSION = 2
MEGATRON_TP1_AUTHORITY = "qualified_megatron_tp1_export_v2"
PYTORCH_CHECKPOINT_AUTHORITY = "verified_pytorch_joint_checkpoint_v1"
PYTORCH_SERVING_PAYLOAD_AUTHORITY = "verified_pytorch_joint_serving_payload_v1"
_EXPECTED_FILES = frozenset(
    {"COMPLETE", "config.json", "dynamics.safetensors", "sha256.json"}
)
_HASHED_PAYLOADS = frozenset({"config.json", "dynamics.safetensors"})
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}")
_NEXTLAT_KEYS = frozenset(
    {
        "dropout",
        "dynamics_dtype",
        "dynamics_file",
        "dynamics_sha256",
        "export_lineage",
        "hidden_state_convention",
        "projection_factor",
        "schema_version",
        "sglang_commit",
        "target_base_revision",
        "target_checkpoint_sha256",
        "target_model_id",
    }
)
_PYTORCH_NEXTLAT_KEYS = (_NEXTLAT_KEYS - {"export_lineage"}) | {"source_lineage"}
_EXPORT_LINEAGE_KEYS = frozenset(
    {
        "branch",
        "checkpoint_integrity_sha256",
        "checkpoint_root_manifest_sha256",
        "export_manifest_sha256",
        "export_receipt_sha256",
        "qualification_receipt_sha256",
        "runtime_image_digest",
        "source_checkpoint_receipt_sha256",
        "source_lock_sha256",
        "tokens_seen",
        "update",
    }
)
_PYTORCH_LINEAGE_KEYS = frozenset(
    {
        "authority",
        "branch",
        "checkpoint_integrity_sha256",
        "checkpoint_manifest_sha256",
        "checkpoint_name",
        "optimizer_included",
        "schema_version",
        "source_code_commit",
        "tokens_seen",
        "train_state_sha256",
        "update",
        "wandb",
    }
)
_PYTORCH_WANDB_KEYS = frozenset({"checkpoint_artifact", "run_id"})
_WANDB_ARTIFACT_KEYS = frozenset(
    {"artifact_digest", "artifact_name", "artifact_url", "artifact_version"}
)
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_WANDB_RUN_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")


@dataclass(frozen=True)
class NextLatExportLineage:
    export_manifest_sha256: str
    export_receipt_sha256: str
    qualification_receipt_sha256: str
    checkpoint_root_manifest_sha256: str
    checkpoint_integrity_sha256: str
    source_checkpoint_receipt_sha256: str
    source_lock_sha256: str
    runtime_image_digest: str
    tokens_seen: int
    update: int
    branch: str | None

    @property
    def authority(self) -> str:
        return MEGATRON_TP1_AUTHORITY


@dataclass(frozen=True)
class NextLatPytorchCheckpointLineage:
    checkpoint_manifest_sha256: str
    checkpoint_integrity_sha256: str
    train_state_sha256: str
    checkpoint_name: str
    source_code_commit: str
    tokens_seen: int
    update: int
    branch: str | None
    optimizer_included: bool
    wandb_run_id: str | None
    wandb_checkpoint_artifact: dict[str, str] | None
    source_authority: str = PYTORCH_CHECKPOINT_AUTHORITY

    @property
    def authority(self) -> str:
        return self.source_authority


@dataclass(frozen=True)
class NextLatSidecarMetadata:
    root: Path
    target_checkpoint_sha256: str
    dynamics_sha256: str
    expected_parameter_count: int
    contract: NextLatContract
    export_lineage: NextLatExportLineage | NextLatPytorchCheckpointLineage

    @property
    def source_lineage(
        self,
    ) -> NextLatExportLineage | NextLatPytorchCheckpointLineage:
        """Return the admitted source authority under a topology-neutral name."""

        return self.export_lineage


@dataclass(frozen=True)
class LoadedNextLatSidecar:
    metadata: NextLatSidecarMetadata
    dynamics: NextLatDynamics

    @property
    def parameter_bytes(self) -> int:
        return sum(
            parameter.numel() * parameter.element_size()
            for parameter in self.dynamics.parameters()
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
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
        raise ValueError(f"cannot read valid JSON object from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _require_equal(name: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        raise ValueError(f"{name}={observed!r}, expected {expected!r}")


def _validate_hash(name: str, value: Any) -> str:
    if not isinstance(value, str) or _HEX_SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 64-character SHA256 digest")
    return value


def _validate_export_lineage(value: Any) -> NextLatExportLineage:
    if not isinstance(value, dict) or set(value) != _EXPORT_LINEAGE_KEYS:
        observed = set(value) if isinstance(value, dict) else set()
        raise ValueError(
            "config.json export_lineage keys do not match: "
            f"missing={sorted(_EXPORT_LINEAGE_KEYS - observed)}, "
            f"extra={sorted(observed - _EXPORT_LINEAGE_KEYS)}"
        )
    hashes = {
        name: _validate_hash(f"nextlat.export_lineage.{name}", value[name])
        for name in _EXPORT_LINEAGE_KEYS
        if name.endswith("_sha256")
    }
    runtime_image_digest = value["runtime_image_digest"]
    if (
        not isinstance(runtime_image_digest, str)
        or not runtime_image_digest.startswith("sha256:")
        or len(runtime_image_digest) != 71
        or _HEX_SHA256.fullmatch(runtime_image_digest.removeprefix("sha256:")) is None
    ):
        raise ValueError(
            "nextlat.export_lineage.runtime_image_digest must be sha256: plus "
            "64 lowercase hex characters"
        )
    for name in ("tokens_seen", "update"):
        observed = value[name]
        if isinstance(observed, bool) or not isinstance(observed, int) or observed < 0:
            raise ValueError(
                f"nextlat.export_lineage.{name} must be a non-negative integer"
            )
    branch = value["branch"]
    if branch not in (None, "d4", "d8", "d16"):
        raise ValueError(
            "nextlat.export_lineage.branch must be null, d4, d8, or d16"
        )
    return NextLatExportLineage(
        export_manifest_sha256=hashes["export_manifest_sha256"],
        export_receipt_sha256=hashes["export_receipt_sha256"],
        qualification_receipt_sha256=hashes["qualification_receipt_sha256"],
        checkpoint_root_manifest_sha256=hashes[
            "checkpoint_root_manifest_sha256"
        ],
        checkpoint_integrity_sha256=hashes["checkpoint_integrity_sha256"],
        source_checkpoint_receipt_sha256=hashes[
            "source_checkpoint_receipt_sha256"
        ],
        source_lock_sha256=hashes["source_lock_sha256"],
        runtime_image_digest=runtime_image_digest,
        tokens_seen=value["tokens_seen"],
        update=value["update"],
        branch=branch,
    )


def _validate_pytorch_lineage(value: Any) -> NextLatPytorchCheckpointLineage:
    if not isinstance(value, dict) or set(value) != _PYTORCH_LINEAGE_KEYS:
        observed = set(value) if isinstance(value, dict) else set()
        raise ValueError(
            "config.json source_lineage keys do not match: "
            f"missing={sorted(_PYTORCH_LINEAGE_KEYS - observed)}, "
            f"extra={sorted(observed - _PYTORCH_LINEAGE_KEYS)}"
        )
    if value["authority"] not in {
        PYTORCH_CHECKPOINT_AUTHORITY, PYTORCH_SERVING_PAYLOAD_AUTHORITY,
    }:
        raise ValueError("unrecognized nextlat.source_lineage.authority")
    _require_equal("nextlat.source_lineage.schema_version", value["schema_version"], 1)
    hashes = {
        name: _validate_hash(f"nextlat.source_lineage.{name}", value[name])
        for name in (
            "checkpoint_integrity_sha256",
            "checkpoint_manifest_sha256",
            "train_state_sha256",
        )
    }
    checkpoint_name = value["checkpoint_name"]
    if (
        not isinstance(checkpoint_name, str)
        or not checkpoint_name
        or len(checkpoint_name) > 255
        or Path(checkpoint_name).name != checkpoint_name
    ):
        raise ValueError(
            "nextlat.source_lineage.checkpoint_name must be one flat non-empty name"
        )
    source_code_commit = value["source_code_commit"]
    if not isinstance(source_code_commit, str) or _GIT_COMMIT.fullmatch(
        source_code_commit
    ) is None:
        raise ValueError(
            "nextlat.source_lineage.source_code_commit must be a lowercase Git SHA1"
        )
    for name in ("tokens_seen", "update"):
        observed = value[name]
        if isinstance(observed, bool) or not isinstance(observed, int) or observed < 0:
            raise ValueError(
                f"nextlat.source_lineage.{name} must be a non-negative integer"
            )
    branch = value["branch"]
    if branch not in (None, "d4", "d8", "d16"):
        raise ValueError(
            "nextlat.source_lineage.branch must be null, d4, d8, or d16"
        )
    optimizer_included = value["optimizer_included"]
    if not isinstance(optimizer_included, bool):
        raise ValueError(
            "nextlat.source_lineage.optimizer_included must be a boolean"
        )
    wandb = value["wandb"]
    if not isinstance(wandb, dict) or set(wandb) != _PYTORCH_WANDB_KEYS:
        raise ValueError("nextlat.source_lineage.wandb key roster differs")
    run_id = wandb["run_id"]
    if run_id is not None and (
        not isinstance(run_id, str) or _WANDB_RUN_ID.fullmatch(run_id) is None
    ):
        raise ValueError("nextlat.source_lineage.wandb.run_id is invalid")
    artifact = wandb["checkpoint_artifact"]
    normalized_artifact: dict[str, str] | None = None
    if artifact is not None:
        if not isinstance(artifact, dict) or set(artifact) != _WANDB_ARTIFACT_KEYS:
            raise ValueError(
                "nextlat.source_lineage.wandb.checkpoint_artifact key roster differs"
            )
        normalized_artifact = {}
        for name in sorted(_WANDB_ARTIFACT_KEYS):
            observed = artifact[name]
            if (
                not isinstance(observed, str)
                or not observed
                or observed != observed.strip()
                or len(observed) > 2048
            ):
                raise ValueError(
                    "nextlat.source_lineage.wandb.checkpoint_artifact."
                    f"{name} is invalid"
                )
            normalized_artifact[name] = observed
    return NextLatPytorchCheckpointLineage(
        checkpoint_manifest_sha256=hashes["checkpoint_manifest_sha256"],
        checkpoint_integrity_sha256=hashes["checkpoint_integrity_sha256"],
        train_state_sha256=hashes["train_state_sha256"],
        checkpoint_name=checkpoint_name,
        source_code_commit=source_code_commit,
        tokens_seen=value["tokens_seen"],
        update=value["update"],
        branch=branch,
        optimizer_included=optimizer_included,
        wandb_run_id=run_id,
        wandb_checkpoint_artifact=normalized_artifact,
        source_authority=value["authority"],
    )


def read_sidecar_metadata(
    root: str | Path,
    *,
    _contract: NextLatContract | None = None,
) -> NextLatSidecarMetadata:
    """Verify completeness, file receipts, target provenance, and fixed shape."""

    sidecar_root = Path(root).expanduser().resolve()
    if not sidecar_root.is_dir():
        raise ValueError(f"NextLat sidecar is not a directory: {sidecar_root}")
    if (sidecar_root / "INCOMPLETE").exists():
        raise ValueError(f"NextLat sidecar is marked incomplete: {sidecar_root}")

    observed_files = frozenset(path.name for path in sidecar_root.iterdir())
    if observed_files != _EXPECTED_FILES:
        missing = sorted(_EXPECTED_FILES - observed_files)
        extra = sorted(observed_files - _EXPECTED_FILES)
        raise ValueError(
            f"NextLat sidecar file set does not match: missing={missing}, extra={extra}"
        )
    for name in _EXPECTED_FILES:
        path = sidecar_root / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"NextLat sidecar entry must be a regular file: {path}")
    if (sidecar_root / "COMPLETE").read_text(encoding="utf-8") != "ok\n":
        raise ValueError("NextLat sidecar COMPLETE marker differs")

    receipt = _read_json_object(sidecar_root / "sha256.json")
    if set(receipt) != _HASHED_PAYLOADS:
        missing = sorted(_HASHED_PAYLOADS - set(receipt))
        extra = sorted(set(receipt) - _HASHED_PAYLOADS)
        raise ValueError(
            f"sidecar SHA256 receipt does not match payloads: "
            f"missing={missing}, extra={extra}"
        )
    for relative_path, expected_digest in receipt.items():
        expected_digest = _validate_hash(
            f"sha256.json[{relative_path!r}]", expected_digest
        )
        observed_digest = _sha256(sidecar_root / relative_path)
        if observed_digest != expected_digest:
            raise ValueError(
                f"sidecar SHA256 mismatch for {relative_path}: "
                f"{observed_digest} != {expected_digest}"
            )

    config = _read_json_object(sidecar_root / "config.json")
    if _contract is None:
        source = config.get("nextlat")
        if not isinstance(source, dict):
            raise ValueError("config.json must contain a nextlat JSON object")
        _contract = scaling_contract(
            source.get("target_model_id"), source.get("target_base_revision"),
            config.get("hidden_size"), config.get("vocab_size"),
            source.get("projection_factor"),
        )
    _require_equal("model_type", config.get("model_type"), "qwen3")
    _require_equal(
        "architectures", config.get("architectures"), ["Qwen3ForCausalLM"]
    )
    _require_equal("hidden_size", config.get("hidden_size"), _contract.hidden_size)
    _require_equal("vocab_size", config.get("vocab_size"), _contract.vocab_size)
    _require_equal("tie_word_embeddings", config.get("tie_word_embeddings"), True)
    _require_equal(
        "torch_dtype", config.get("torch_dtype"), _contract.target_runtime_dtype
    )

    nextlat = config.get("nextlat")
    if not isinstance(nextlat, dict):
        raise ValueError("config.json must contain a nextlat JSON object")
    schema_version = nextlat.get("schema_version")
    expected_keys = (
        _NEXTLAT_KEYS
        if schema_version == SIDECAR_SCHEMA_VERSION
        else _PYTORCH_NEXTLAT_KEYS
        if schema_version == PYTORCH_SIDECAR_SCHEMA_VERSION
        else frozenset()
    )
    if not expected_keys:
        raise ValueError(f"unsupported nextlat.schema_version: {schema_version!r}")
    if set(nextlat) != expected_keys:
        missing = sorted(expected_keys - set(nextlat))
        extra = sorted(set(nextlat) - expected_keys)
        raise ValueError(
            f"config.json nextlat keys do not match: missing={missing}, extra={extra}"
        )

    _require_equal("nextlat.schema_version", nextlat["schema_version"], schema_version)
    _require_equal(
        "nextlat.dynamics_file", nextlat["dynamics_file"], "dynamics.safetensors"
    )
    _require_equal(
        "nextlat.target_model_id", nextlat["target_model_id"], _contract.target_model_id
    )
    _require_equal(
        "nextlat.target_base_revision",
        nextlat["target_base_revision"],
        _contract.target_base_revision,
    )
    _require_equal(
        "nextlat.hidden_state_convention",
        nextlat["hidden_state_convention"],
        _contract.hidden_state_convention,
    )
    _require_equal(
        "nextlat.projection_factor",
        nextlat["projection_factor"],
        _contract.projection_factor,
    )
    _require_equal("nextlat.dropout", nextlat["dropout"], _contract.dropout)
    _require_equal(
        "nextlat.dynamics_dtype",
        nextlat["dynamics_dtype"],
        _contract.dynamics_dtype,
    )
    _require_equal(
        "nextlat.sglang_commit", nextlat["sglang_commit"], _contract.sglang_commit
    )

    dynamics_sha256 = _validate_hash(
        "nextlat.dynamics_sha256", nextlat["dynamics_sha256"]
    )
    _require_equal(
        "nextlat.dynamics_sha256",
        dynamics_sha256,
        receipt["dynamics.safetensors"],
    )
    target_checkpoint_sha256 = _validate_hash(
        "nextlat.target_checkpoint_sha256", nextlat["target_checkpoint_sha256"]
    )
    export_lineage = (
        _validate_export_lineage(nextlat["export_lineage"])
        if schema_version == SIDECAR_SCHEMA_VERSION
        else _validate_pytorch_lineage(nextlat["source_lineage"])
    )

    projected = projection_size(
        _contract.hidden_size, projection_factor=_contract.projection_factor
    )
    expected_parameter_count = (
        2 * _contract.hidden_size
        + projected * 2 * _contract.hidden_size
        + projected * projected
        + _contract.hidden_size * projected
    )
    return NextLatSidecarMetadata(
        root=sidecar_root,
        target_checkpoint_sha256=target_checkpoint_sha256,
        dynamics_sha256=dynamics_sha256,
        expected_parameter_count=expected_parameter_count,
        contract=_contract,
        export_lineage=export_lineage,
    )


def load_dynamics_sidecar(
    root: str | Path,
    *,
    device: str | torch.device = "cpu",
    _contract: NextLatContract | None = None,
) -> LoadedNextLatSidecar:
    """Load the source FP32 dynamics only after all receipts pass."""

    metadata = read_sidecar_metadata(root, _contract=_contract)
    _contract = metadata.contract
    state = load_file(
        metadata.root / "dynamics.safetensors", device=str(torch.device(device))
    )
    if _contract.dynamics_dtype != "float32":
        raise ValueError(
            "the narrow loader only admits exact FP32 source dynamics; "
            f"got {_contract.dynamics_dtype}"
        )
    validate_state_dict(
        state,
        hidden_size=_contract.hidden_size,
        projection_factor=_contract.projection_factor,
        expected_dtype=torch.float32,
    )
    dynamics = NextLatDynamics(
        hidden_size=_contract.hidden_size,
        projection_factor=_contract.projection_factor,
        dropout=_contract.dropout,
    ).to(device=device, dtype=torch.float32)
    dynamics.load_state_dict(state, strict=True)
    dynamics.eval()
    return LoadedNextLatSidecar(metadata=metadata, dynamics=dynamics)
