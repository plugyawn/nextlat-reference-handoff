"""Verify the exact plugin source set copied into the serving image."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

from .constants import AUDITED_SGLANG_COMMIT

PLUGIN_MANIFEST_SCHEMA_VERSION = 1
PLUGIN_MANIFEST_RELATIVE = Path("receipts/plugin-source-manifest.json")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protected_source_files(root: str | Path) -> dict[str, Path]:
    package_root = Path(root).resolve()
    files = {
        "Dockerfile.sm90": package_root / "Dockerfile.sm90",
        "pyproject.toml": package_root / "pyproject.toml",
    }
    source_root = package_root / "src" / "sglang_nextlat"
    for path in sorted(source_root.rglob("*")):
        if path.is_file() and not path.is_symlink() and path.suffix in {".py", ".json"}:
            relative = path.relative_to(package_root).as_posix()
            files[relative] = path
    return files


def plugin_manifest_sha256(root: str | Path) -> str:
    return sha256_file(Path(root).resolve() / PLUGIN_MANIFEST_RELATIVE)


def verify_plugin_source_manifest(
    root: str | Path,
    *,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    package_root = Path(root).resolve()
    manifest_path = package_root / PLUGIN_MANIFEST_RELATIVE
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise RuntimeError(f"plugin source manifest is missing: {manifest_path}")
    observed_manifest_sha256 = sha256_file(manifest_path)
    if (
        expected_manifest_sha256 is not None
        and observed_manifest_sha256 != expected_manifest_sha256
    ):
        raise RuntimeError("plugin source manifest digest differs from build argument")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("plugin source manifest is not valid JSON") from exc
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema_version",
        "sglang_commit",
        "cuda_architecture",
        "files",
    }:
        raise RuntimeError("plugin source manifest schema keys differ")
    if manifest["schema_version"] != PLUGIN_MANIFEST_SCHEMA_VERSION:
        raise RuntimeError("unsupported plugin source manifest schema")
    if manifest["sglang_commit"] != AUDITED_SGLANG_COMMIT:
        raise RuntimeError("plugin manifest SGLang commit differs")
    if manifest["cuda_architecture"] != "sm90":
        raise RuntimeError("plugin manifest CUDA architecture must be sm90")
    recorded = manifest["files"]
    if not isinstance(recorded, dict) or not recorded:
        raise RuntimeError("plugin manifest files must be a non-empty object")
    normalized: dict[str, str] = {}
    for relative, digest in recorded.items():
        if not isinstance(relative, str):
            raise RuntimeError("plugin manifest path must be a string")
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != relative:
            raise RuntimeError(f"unsafe plugin manifest path: {relative!r}")
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise RuntimeError(f"invalid plugin source digest: {relative}")
        normalized[relative] = digest
    actual = protected_source_files(package_root)
    if set(normalized) != set(actual):
        raise RuntimeError(
            "plugin protected source roster differs: "
            f"missing={sorted(set(normalized) - set(actual))}, "
            f"extra={sorted(set(actual) - set(normalized))}"
        )
    for relative, path in actual.items():
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"plugin protected source is not regular: {path}")
        if sha256_file(path) != normalized[relative]:
            raise RuntimeError(f"plugin protected source drift: {relative}")
    return {
        "manifest_sha256": observed_manifest_sha256,
        "file_count": len(normalized),
        "sglang_commit": manifest["sglang_commit"],
        "cuda_architecture": manifest["cuda_architecture"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256")
    args = parser.parse_args()
    result = verify_plugin_source_manifest(
        args.root, expected_manifest_sha256=args.expected_manifest_sha256
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
