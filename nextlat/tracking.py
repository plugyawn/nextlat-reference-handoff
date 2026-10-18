"""Required online Weights & Biases tracking for the registered long run."""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import wandb


RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _flatten_numeric(value: Any, prefix: str, output: Dict[str, float]) -> None:
    """Flatten scalar metrics while deliberately skipping strings and vectors."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            child_prefix = f"{prefix}/{key}" if prefix else str(key)
            _flatten_numeric(child, child_prefix, output)
        return
    if isinstance(value, (list, tuple)) or value is None or isinstance(value, str):
        return
    if isinstance(value, bool):
        output[prefix] = float(value)
        return
    if isinstance(value, (int, float)):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise RuntimeError(f"refusing to log non-finite W&B metric {prefix}={value}")
        output[prefix] = numeric


def numeric_payload(
    namespace: str,
    record: Mapping[str, Any],
    *,
    tokens_seen: Optional[int] = None,
    update: Optional[int] = None,
) -> Dict[str, float]:
    """Convert a local record to scalar W&B metrics with a shared token axis."""

    payload: Dict[str, float] = {}
    for key, value in record.items():
        if key in {"type", "tokens_seen", "update"}:
            continue
        if key in {"smooth_l1_by_horizon", "kl_by_horizon"} and isinstance(
            value, (list, tuple)
        ):
            for horizon, child in enumerate(value, start=1):
                _flatten_numeric(child, f"{namespace}/{key}/h{horizon}", payload)
            continue
        _flatten_numeric(value, f"{namespace}/{key}", payload)
    resolved_tokens = record.get("tokens_seen") if tokens_seen is None else tokens_seen
    resolved_update = record.get("update") if update is None else update
    if resolved_tokens is not None:
        payload["tokens_seen"] = float(resolved_tokens)
    if resolved_update is not None:
        payload["update"] = float(resolved_update)
    return payload


@dataclass
class WandbTracker:
    run: Any
    receipt_path: Path

    def log_record(
        self,
        namespace: str,
        record: Mapping[str, Any],
        *,
        tokens_seen: Optional[int] = None,
        update: Optional[int] = None,
    ) -> None:
        self.run.log(
            numeric_payload(
                namespace,
                record,
                tokens_seen=tokens_seen,
                update=update,
            )
        )

    def log_checkpoint_artifact(
        self,
        checkpoint_dir: Path,
        *,
        artifact_name: str,
        aliases: list[str],
        metadata: Mapping[str, Any],
    ) -> None:
        """Publish a complete, self-verifying optimizer checkpoint to W&B."""

        if not (checkpoint_dir / "COMPLETE").is_file():
            raise RuntimeError(f"refusing to upload incomplete checkpoint: {checkpoint_dir}")
        artifact = wandb.Artifact(
            name=artifact_name,
            type="optimizer-checkpoint",
            metadata=dict(metadata),
        )
        artifact.add_dir(str(checkpoint_dir), name=checkpoint_dir.name)
        self.run.log_artifact(artifact, aliases=aliases)

    def log_checkpoint_receipt_artifact(
        self,
        checkpoint_dir: Path,
        *,
        artifact_name: str,
        aliases: list[str],
        metadata: Mapping[str, Any],
    ) -> None:
        """Publish only binding receipts for a locally retained checkpoint."""

        required = ("COMPLETE", "sha256.json", "train_state.json")
        missing = [name for name in required if not (checkpoint_dir / name).is_file()]
        if missing:
            raise RuntimeError(
                f"checkpoint receipt artifact is incomplete: {sorted(missing)}"
            )
        artifact = wandb.Artifact(
            name=artifact_name,
            type="optimizer-checkpoint-receipt",
            metadata=dict(metadata),
        )
        for name in required:
            artifact.add_file(
                str(checkpoint_dir / name),
                name=f"{checkpoint_dir.name}/{name}",
            )
        self.run.log_artifact(artifact, aliases=aliases)

    def finish(
        self,
        *,
        summary_namespace: Optional[str] = None,
        summary: Optional[Mapping[str, Any]] = None,
        exit_code: int = 0,
    ) -> None:
        if summary_namespace is not None and summary is not None:
            flattened: Dict[str, float] = {}
            _flatten_numeric(summary, summary_namespace, flattened)
            for key, value in flattened.items():
                self.run.summary[key] = value
        self.run.finish(exit_code=exit_code)


def start_wandb_tracker(
    *,
    tracking_config: Mapping[str, Any],
    run_dir: Path,
    run_id: str,
    resume_mode: str,
    experiment_config: Mapping[str, Any],
    source_code_commit: str,
    phase: str,
) -> WandbTracker:
    """Start one required-online W&B session and write a secret-free receipt."""

    if tracking_config.get("backend") != "wandb":
        raise RuntimeError("the registered tracking backend must be wandb")
    if tracking_config.get("required") is not True:
        raise RuntimeError("the registered long run requires W&B tracking")
    if tracking_config.get("mode") != "online":
        raise RuntimeError("the registered W&B run must use online mode")
    if resume_mode not in {"never", "must"}:
        raise ValueError(f"unsupported W&B resume mode: {resume_mode}")
    if not run_id or not RUN_ID_RE.fullmatch(run_id):
        raise RuntimeError("--wandb-run-id must contain only letters, digits, _ or -")
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY is required for the registered long run")

    entity = str(tracking_config.get("entity") or "")
    project = str(tracking_config.get("project") or "")
    if not entity or not project:
        raise RuntimeError("W&B entity and project must be explicit")

    run = wandb.init(
        entity=entity,
        project=project,
        id=run_id,
        name=str(tracking_config["run_name"]),
        group=str(tracking_config["group"]),
        job_type=str(tracking_config["job_type"]),
        tags=[str(value) for value in tracking_config.get("tags", [])],
        notes=str(tracking_config.get("notes") or ""),
        dir=str(run_dir),
        mode="online",
        resume=resume_mode,
        config={
            "experiment": dict(experiment_config),
            "source_code_commit": source_code_commit,
        },
        settings=wandb.Settings(init_timeout=120),
        save_code=True,
    )
    if run is None:
        raise RuntimeError("wandb.init returned no run")
    if run.entity != entity or run.project != project or run.id != run_id:
        run.finish(exit_code=1)
        raise RuntimeError(
            "W&B initialized the wrong destination: "
            f"{run.entity}/{run.project}/{run.id} != {entity}/{project}/{run_id}"
        )
    if not run.url:
        run.finish(exit_code=1)
        raise RuntimeError("online W&B run has no dashboard URL")

    run.define_metric("tokens_seen", summary="max")
    run.define_metric("update", summary="max")
    for namespace in ("train", "recursive", "evaluation", "search_problem"):
        run.define_metric(f"{namespace}/*", step_metric="tokens_seen")

    receipt = {
        "schema_version": 1,
        "backend": "wandb",
        "mode": "online",
        "entity": run.entity,
        "project": run.project,
        "run_id": run.id,
        "run_name": run.name,
        "url": run.url,
        "phase": phase,
        "resume_mode": resume_mode,
        "source_code_commit": source_code_commit,
    }
    receipt_path = run_dir / "wandb.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return WandbTracker(run=run, receipt_path=receipt_path)
