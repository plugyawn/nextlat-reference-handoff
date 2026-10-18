"""Build and receipt the immutable SM90 NextLat serving image."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping

from .admission import canonical_json_bytes
from .constants import AUDITED_SGLANG_COMMIT
from .image_contract import (
    plugin_manifest_sha256,
    sha256_file,
    verify_plugin_source_manifest,
)

IMAGE_RECEIPT_SCHEMA_VERSION = 1
_LOCAL_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")
_REGISTRY_DIGEST = re.compile(r"[^@\s]+@sha256:[0-9a-f]{64}\Z")


def validate_immutable_base_reference(
    value: str, *, expected_local_image_id: str | None = None
) -> str:
    if _REGISTRY_DIGEST.fullmatch(value) is not None:
        if expected_local_image_id is not None:
            raise ValueError(
                "registry digest base must not also provide an expected local image ID"
            )
        return value
    if _LOCAL_IMAGE_ID.fullmatch(value) is not None:
        raise ValueError(
            "raw local image IDs are immutable but not valid BuildKit FROM "
            "references; pass a local tag plus expected_local_image_id"
        )
    if (
        not value
        or any(character.isspace() for character in value)
        or "@" in value
        or expected_local_image_id is None
        or _LOCAL_IMAGE_ID.fullmatch(expected_local_image_id) is None
    ):
        raise ValueError(
            "base image must be a registry reference ending in @sha256:<64 hex> "
            "or a local tag bound to a full expected local image ID"
        )
    return value


def docker_build_command(
    *,
    package_root: Path,
    base_image: str,
    expected_local_base_image_id: str | None,
    tag: str,
    manifest_sha256: str,
) -> list[str]:
    validate_immutable_base_reference(
        base_image, expected_local_image_id=expected_local_base_image_id
    )
    if not tag or any(character.isspace() for character in tag):
        raise ValueError("image tag must be non-empty and contain no whitespace")
    command = [
        "docker",
        "build",
    ]
    if _REGISTRY_DIGEST.fullmatch(base_image) is None:
        command.append("--pull=false")
    command.extend(
        [
        "--file",
        str(package_root / "Dockerfile.sm90"),
        "--target",
        "nextlat-serving",
        "--build-arg",
        f"SGLANG_BASE_IMAGE={base_image}",
        "--build-arg",
        f"SGLANG_COMMIT={AUDITED_SGLANG_COMMIT}",
        "--build-arg",
        f"PLUGIN_SOURCE_MANIFEST_SHA256={manifest_sha256}",
        "--tag",
        tag,
        str(package_root),
        ]
    )
    return command


def _run(command: list[str]) -> str:
    result = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed with status {result.returncode}: {command!r}\n"
            f"{result.stdout[-12000:]}"
        )
    return result.stdout


def _inspect_image(reference: str) -> dict[str, Any]:
    output = _run(["docker", "image", "inspect", reference])
    try:
        values = json.loads(output)
    except json.JSONDecodeError as exc:
        raise RuntimeError("docker image inspect did not return JSON") from exc
    if not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], dict):
        raise RuntimeError("docker image inspect returned an unexpected shape")
    image = values[0]
    if not isinstance(image.get("Id"), str) or _LOCAL_IMAGE_ID.fullmatch(image["Id"]) is None:
        raise RuntimeError("docker image inspect returned no content-addressed image id")
    return image


def _write_receipt(path: Path, receipt: Mapping[str, Any]) -> Path:
    destination = path.expanduser()
    if destination.is_symlink():
        raise ValueError("image receipt destination must not be a symlink")
    parent = destination.parent.resolve()
    if not parent.is_dir():
        raise FileNotFoundError(f"image receipt parent is missing: {parent}")
    destination = parent / destination.name
    encoded = canonical_json_bytes(dict(receipt), pretty=True)
    if destination.exists():
        if destination.read_bytes() != encoded:
            raise FileExistsError(f"refusing to overwrite image receipt: {destination}")
        return destination
    partial = parent / f".{destination.name}.{os.getpid()}.partial"
    if partial.exists():
        raise FileExistsError(f"stale image receipt partial exists: {partial}")
    try:
        with partial.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.rename(partial, destination)
    finally:
        if partial.exists():
            partial.unlink()
    return destination


def build_image(
    *,
    package_root: Path,
    base_image: str,
    expected_local_base_image_id: str | None,
    tag: str,
    receipt_path: Path,
) -> Path:
    package_root = package_root.resolve()
    base_image = validate_immutable_base_reference(
        base_image, expected_local_image_id=expected_local_base_image_id
    )
    manifest = verify_plugin_source_manifest(package_root)
    manifest_sha256 = plugin_manifest_sha256(package_root)
    base = _inspect_image(base_image)
    if (
        expected_local_base_image_id is not None
        and base["Id"] != expected_local_base_image_id
    ):
        raise RuntimeError(
            "local base tag image ID differs from expected_local_base_image_id"
        )
    command = docker_build_command(
        package_root=package_root,
        base_image=base_image,
        expected_local_base_image_id=expected_local_base_image_id,
        tag=tag,
        manifest_sha256=manifest_sha256,
    )
    _run(command)
    image = _inspect_image(tag)
    labels = image.get("Config", {}).get("Labels", {})
    expected_labels = {
        "ai.nextlat.sglang-commit": AUDITED_SGLANG_COMMIT,
        "ai.nextlat.plugin-source-manifest-sha256": manifest_sha256,
        "ai.nextlat.cuda-architecture": "sm90",
    }
    if not isinstance(labels, dict) or any(
        labels.get(name) != expected for name, expected in expected_labels.items()
    ):
        raise RuntimeError("derived image labels differ from the build contract")
    runtime_output = _run(
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "python3",
            image["Id"],
            "-m",
            "sglang_nextlat.image_selfcheck",
        ]
    )
    try:
        runtime_contract = json.loads(runtime_output.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError("derived image self-check returned no JSON receipt") from exc
    if runtime_contract.get("sglang_commit") != AUDITED_SGLANG_COMMIT:
        raise RuntimeError("derived image self-check reported a different SGLang commit")

    payload = {
        "schema_version": IMAGE_RECEIPT_SCHEMA_VERSION,
        "result": "pass",
        "image_id": image["Id"],
        "repo_digests": sorted(image.get("RepoDigests") or []),
        "base_reference": base_image,
        "base_image_id": base["Id"],
        "expected_local_base_image_id": expected_local_base_image_id,
        "sglang_commit": AUDITED_SGLANG_COMMIT,
        "plugin_source_manifest_sha256": manifest_sha256,
        "plugin_source_file_count": manifest["file_count"],
        "dockerfile_sha256": sha256_file(package_root / "Dockerfile.sm90"),
        "cuda_architecture": "sm90",
        "runtime_contract": runtime_contract,
        "build_command_sha256": hashlib.sha256(
            canonical_json_bytes(command)
        ).hexdigest(),
    }
    receipt_sha256 = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return _write_receipt(
        receipt_path, {**payload, "receipt_sha256": receipt_sha256}
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--base-image", required=True)
    parser.add_argument("--expected-local-base-image-id")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    print(
        build_image(
            package_root=args.package_root,
            base_image=args.base_image,
            expected_local_base_image_id=args.expected_local_base_image_id,
            tag=args.tag,
            receipt_path=args.receipt,
        )
    )


if __name__ == "__main__":
    main()
