"""Strict target-checkpoint and pinned-SGLang admission checks."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .constants import AUDITED_SGLANG_COMMIT, NextLatContract, QWEN3_1_7B_CONTRACT

TARGET_MANIFEST_SCHEMA_VERSION = 1
SOURCE_RECEIPT_NAME = "audited_sglang.json"
_SOURCE_PATH_PREFIX = "python/sglang/"


@dataclass(frozen=True)
class TargetCheckpointAdmission:
    root: Path
    manifest_sha256: str
    payloads: tuple[tuple[str, int, str], ...]


@dataclass(frozen=True)
class SGLangSourceAdmission:
    package_root: Path
    receipt_sha256: str
    audited_files: tuple[tuple[str, str], ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    if pretty:
        encoded = json.dumps(value, sort_keys=True, indent=2)
    else:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return (encoded + "\n").encode("utf-8")


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON object from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _regular_file(root: Path, relative: str) -> Path:
    path = root / relative
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"target payload must be a regular non-symlink file: {path}")
    return path


def _validate_target_config(
    config: dict[str, Any], contract: NextLatContract
) -> None:
    expected = {
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
        "hidden_size": contract.hidden_size,
        "vocab_size": contract.vocab_size,
        "tie_word_embeddings": True,
    }
    for name, wanted in expected.items():
        observed = config.get(name)
        if observed != wanted:
            raise ValueError(
                f"target config {name}={observed!r}, expected {wanted!r}"
            )
    observed_dtype = config.get("torch_dtype", config.get("dtype"))
    if observed_dtype != contract.target_source_dtype:
        raise ValueError(
            "target config source dtype must be "
            f"{contract.target_source_dtype}, got {observed_dtype!r}"
        )


def _inspect_target_checkpoint(
    root: str | Path,
    *,
    contract: NextLatContract = QWEN3_1_7B_CONTRACT,
) -> TargetCheckpointAdmission:
    raw_root = Path(root).expanduser()
    if raw_root.is_symlink():
        raise ValueError(f"target checkpoint root must not be a symlink: {raw_root}")
    target_root = raw_root.resolve()
    if not target_root.is_dir():
        raise ValueError(f"target checkpoint is not a local directory: {target_root}")
    config_path = _regular_file(target_root, "config.json")
    _validate_target_config(_json_object(config_path), contract)

    index_path = target_root / "model.safetensors.index.json"
    single_path = target_root / "model.safetensors"
    payload_names: list[str]
    if index_path.exists():
        if single_path.exists():
            raise ValueError("target checkpoint mixes indexed and single-file weights")
        _regular_file(target_root, index_path.name)
        index = _json_object(index_path)
        if set(index) != {"metadata", "weight_map"}:
            raise ValueError("safetensors index must contain exactly metadata and weight_map")
        weight_map = index["weight_map"]
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("safetensors index weight_map must be a non-empty object")
        shard_names = set()
        for tensor_name, shard_name in weight_map.items():
            if not isinstance(tensor_name, str) or not tensor_name:
                raise ValueError("safetensors index contains an invalid tensor name")
            if not isinstance(shard_name, str) or Path(shard_name).name != shard_name:
                raise ValueError("safetensors shard names must be flat relative filenames")
            if not shard_name.endswith(".safetensors"):
                raise ValueError(f"indexed weight is not safetensors: {shard_name}")
            shard_names.add(shard_name)
        payload_names = ["config.json", index_path.name, *sorted(shard_names)]
    else:
        _regular_file(target_root, single_path.name)
        payload_names = ["config.json", single_path.name]

    referenced_weights = {name for name in payload_names if name.endswith(".safetensors")}
    observed_weights = {
        path.name for path in target_root.glob("*.safetensors") if path.is_file()
    }
    if observed_weights != referenced_weights:
        raise ValueError(
            "target safetensors file set differs from the executable manifest: "
            f"missing={sorted(referenced_weights - observed_weights)}, "
            f"extra={sorted(observed_weights - referenced_weights)}"
        )
    forbidden = sorted(
        path.name
        for pattern in ("*.bin", "*.pt", "*.pth")
        for path in target_root.glob(pattern)
        if path.is_file()
    )
    if forbidden:
        raise ValueError(f"target checkpoint contains alternate weight payloads: {forbidden}")

    payloads = tuple(
        (
            name,
            _regular_file(target_root, name).stat().st_size,
            sha256_file(_regular_file(target_root, name)),
        )
        for name in payload_names
    )
    manifest = {
        "schema_version": TARGET_MANIFEST_SCHEMA_VERSION,
        "payloads": [
            {"path": name, "size": size, "sha256": digest}
            for name, size, digest in payloads
        ],
    }
    observed_manifest_sha256 = hashlib.sha256(
        canonical_json_bytes(manifest)
    ).hexdigest()
    return TargetCheckpointAdmission(
        root=target_root,
        manifest_sha256=observed_manifest_sha256,
        payloads=payloads,
    )


def admit_target_checkpoint(
    root: str | Path,
    *,
    expected_manifest_sha256: str,
    contract: NextLatContract = QWEN3_1_7B_CONTRACT,
) -> TargetCheckpointAdmission:
    """Hash the exact executable HF model payload and bind it to the sidecar.

    Tokenizer collateral is intentionally outside this digest.  The executable
    target is exactly ``config.json`` plus either one safetensors file or an
    index and every shard named by that index.
    """

    if len(expected_manifest_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in expected_manifest_sha256
    ):
        raise ValueError("expected target manifest must be a lowercase SHA256")
    admission = _inspect_target_checkpoint(root, contract=contract)
    if admission.manifest_sha256 != expected_manifest_sha256:
        raise ValueError(
            "target checkpoint manifest SHA256 differs from the NextLat sidecar: "
            f"{admission.manifest_sha256} != {expected_manifest_sha256}"
        )
    return admission


def target_checkpoint_manifest_sha256(
    root: str | Path,
    *,
    contract: NextLatContract = QWEN3_1_7B_CONTRACT,
) -> str:
    """Compute the deterministic digest used when constructing a sidecar."""

    return _inspect_target_checkpoint(root, contract=contract).manifest_sha256


def _bundled_source_receipt() -> Path:
    return Path(__file__).with_name(SOURCE_RECEIPT_NAME)


def admit_sglang_source(
    *,
    package_root: str | Path | None = None,
    receipt_path: str | Path | None = None,
) -> SGLangSourceAdmission:
    """Verify every serving API anchor against the audited source receipt."""

    receipt_file = (
        Path(receipt_path).resolve()
        if receipt_path is not None
        else _bundled_source_receipt()
    )
    receipt = _json_object(receipt_file)
    if receipt.get("schema_version") != 1:
        raise ValueError("unsupported audited SGLang receipt schema")
    if receipt.get("commit") != AUDITED_SGLANG_COMMIT:
        raise ValueError("audited SGLang receipt commit differs from the contract")
    audited_files = receipt.get("audited_files")
    if not isinstance(audited_files, dict) or not audited_files:
        raise ValueError("audited SGLang receipt has no source files")

    if package_root is None:
        spec = importlib.util.find_spec("sglang")
        locations = None if spec is None else spec.submodule_search_locations
        if not locations:
            raise ValueError("cannot locate an installed SGLang package")
        sglang_root = Path(next(iter(locations))).resolve()
    else:
        sglang_root = Path(package_root).expanduser().resolve()
    if not sglang_root.is_dir():
        raise ValueError(f"SGLang package root is not a directory: {sglang_root}")

    verified: list[tuple[str, str]] = []
    for receipt_name, expected_digest in sorted(audited_files.items()):
        if not isinstance(receipt_name, str) or not receipt_name.startswith(
            _SOURCE_PATH_PREFIX
        ):
            raise ValueError(f"invalid audited SGLang path: {receipt_name!r}")
        if not isinstance(expected_digest, str) or len(expected_digest) != 64:
            raise ValueError(f"invalid audited digest for {receipt_name}")
        relative = receipt_name.removeprefix(_SOURCE_PATH_PREFIX)
        source_path = sglang_root / relative
        if not source_path.is_file() or source_path.is_symlink():
            raise ValueError(f"audited SGLang source file is missing: {source_path}")
        observed = sha256_file(source_path)
        if observed != expected_digest:
            raise ValueError(
                f"audited SGLang source drift for {receipt_name}: "
                f"{observed} != {expected_digest}"
            )
        verified.append((receipt_name, observed))

    return SGLangSourceAdmission(
        package_root=sglang_root,
        receipt_sha256=sha256_file(receipt_file),
        audited_files=tuple(verified),
    )
