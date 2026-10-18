"""Export a scaling checkpoint's verified serving subset through the existing builder.

No Adam/RNG payload is required or claimed verified. The original checkpoint
manifest remains unchanged; source_lineage explicitly identifies serving-only
authority and its integrity digest covers only the rehashed serving payload.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from safetensors import safe_open

from sglang_nextlat.admission import admit_target_checkpoint, target_checkpoint_manifest_sha256
from sglang_nextlat.build_sidecar import (
    AdmittedPytorchCheckpoint, _build_admitted_sidecar,
    _checkpoint_integrity_sha256, _infer_pytorch_branch, _read_json,
    _regular_files, _require_sha256, _sha256, _validate_dynamics_metadata,
    _validate_hash_manifest,
)
from sglang_nextlat.checkpoint import PYTORCH_SERVING_PAYLOAD_AUTHORITY, read_sidecar_metadata
from sglang_nextlat.constants import scaling_contract


@dataclass(frozen=True)
class AdmittedServingPayload(AdmittedPytorchCheckpoint):
    @property
    def authority_label(self) -> str:
        return "verified PyTorch joint serving payload (no optimizer/RNG admission)"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--models-json", type=Path, required=True)
    p.add_argument("--expected-checkpoint-manifest-sha256", required=True)
    p.add_argument("--expected-tokens", type=int, required=True)
    p.add_argument("--artifact-name", required=True)
    p.add_argument("--artifact-digest", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--receipt", type=Path, required=True)
    args = p.parse_args()
    root = args.checkpoint.resolve()
    if args.checkpoint.is_symlink() or not root.is_dir():
        raise ValueError("checkpoint must be a real directory")
    if (root / "COMPLETE").read_text() != "ok\n" or (root / "INCOMPLETE").exists():
        raise ValueError("source checkpoint is not complete")
    manifest_hash = _require_sha256(
        args.expected_checkpoint_manifest_sha256, label="expected checkpoint manifest")
    if _sha256(root / "sha256.json") != manifest_hash:
        raise ValueError("checkpoint manifest differs from downloaded immutable artifact")
    source_hashes = _read_json(root / "sha256.json", label="original checkpoint manifest")
    required = {name: digest for name, digest in source_hashes.items()
                if name.startswith("model/") or name in {"dynamics.safetensors", "train_state.json"}}
    files = _regular_files(root)
    observed = {name: path for name, path in files.items()
                if name not in {"COMPLETE", "sha256.json", "optimizer.pt", "rng_state.pt"}}
    verified = _validate_hash_manifest(required, observed)
    for name in ("model/config.json", "dynamics.safetensors", "train_state.json"):
        if name not in verified:
            raise ValueError(f"missing serving payload: {name}")

    config = _read_json(root / "model/config.json", label="target config")
    state = _read_json(root / "train_state.json", label="source training state")
    if state["tokens_seen"] != args.expected_tokens:
        raise ValueError("checkpoint exposure differs from immutable inventory")
    records = _read_json(args.models_json, label="pinned scaling targets")
    matches = [r for r in records.values()
               if r["config"]["hidden_size"] == config["hidden_size"]]
    if len(matches) != 1:
        raise ValueError("checkpoint does not identify exactly one pinned scaling target")
    record = matches[0]
    hidden = config["hidden_size"]
    with safe_open(root / "dynamics.safetensors", framework="pt", device="cpu") as f:
        projection, input_width = f.get_slice("mlp.0.weight").get_shape()
    if input_width != 2 * hidden:
        raise ValueError("sidecar input width differs from the exact target")
    contract = scaling_contract(record["hub_id"], record["revision"], hidden,
                                config["vocab_size"], projection / (2 * hidden))
    _validate_dynamics_metadata(root / "dynamics.safetensors", contract)
    parameter_count = 2 * hidden + 3 * hidden * projection + projection * projection
    if state["parameter_counts"]["psi"] != parameter_count:
        raise ValueError("sidecar tensor geometry differs from saved training state")
    target_hash = target_checkpoint_manifest_sha256(root / "model", contract=contract)
    target = admit_target_checkpoint(root / "model", expected_manifest_sha256=target_hash,
                                     contract=contract)
    if any(verified[f"model/{name}"] != digest for name, _, digest in target.payloads):
        raise ValueError("executable target escaped original source manifest")
    integrity_hash = _checkpoint_integrity_sha256(verified, observed)
    lineage = {
        "authority": PYTORCH_SERVING_PAYLOAD_AUTHORITY,
        "branch": _infer_pytorch_branch(state),
        "checkpoint_integrity_sha256": integrity_hash,
        "checkpoint_manifest_sha256": manifest_hash,
        "checkpoint_name": root.name,
        "optimizer_included": False,
        "schema_version": 1,
        "source_code_commit": state["source_code_commit"],
        "tokens_seen": state["tokens_seen"],
        "train_state_sha256": verified["train_state.json"],
        "update": state["update"],
        "wandb": {"run_id": state.get("wandb_run_id"), "checkpoint_artifact": {
            "artifact_name": args.artifact_name,
            "artifact_digest": args.artifact_digest,
            "artifact_version": args.artifact_name.rsplit(":", 1)[1],
            "artifact_url": "wandb-artifact:///" + args.artifact_name,
        }},
    }
    admitted = AdmittedServingPayload(
        root=root, model_root=root / "model", dynamics_path=root / "dynamics.safetensors",
        checkpoint_manifest_sha256=manifest_hash, checkpoint_integrity_sha256=integrity_hash,
        target_checkpoint_sha256=target_hash, dynamics_sha256=verified["dynamics.safetensors"],
        lineage=lineage,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = _build_admitted_sidecar(admitted, args.output, contract=contract)
    metadata = read_sidecar_metadata(output)
    if metadata.contract != contract or metadata.expected_parameter_count != parameter_count:
        raise ValueError("published sidecar failed derived-contract roundtrip")
    receipt = {
        "target": str(root / "model"), "sidecar": str(output),
        "tokens_seen": state["tokens_seen"], "sidecar_parameters": parameter_count,
        "contract": asdict(contract), "source_lineage": lineage,
        "verified_payload_sha256": verified,
        "explicitly_not_verified": [name for name in source_hashes if name not in verified],
        "target_checkpoint_sha256": target_hash,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open("x") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")
    print(json.dumps({key: receipt[key] for key in
                      ("target", "sidecar", "tokens_seen", "sidecar_parameters",
                       "target_checkpoint_sha256")}))


if __name__ == "__main__":
    main()
