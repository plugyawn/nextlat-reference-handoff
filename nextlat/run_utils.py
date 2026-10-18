"""Fail-closed run receipts and checkpoints."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch
from safetensors.torch import load_file, save_file


def _command_output(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
    except (OSError, subprocess.CalledProcessError) as error:
        return f"UNAVAILABLE: {error}\n"


def git_commit() -> str:
    """Return the exact source commit, failing outside a Git checkout."""

    commit = _command_output(["git", "rev-parse", "HEAD"]).strip()
    if not commit or commit.startswith("UNAVAILABLE:"):
        raise RuntimeError(f"cannot resolve source commit: {commit}")
    return commit


def assert_claim_source_committed(extra_paths: Iterable[Path | str] = ()) -> str:
    """Require the claim-bearing implementation and config to match ``HEAD``.

    Operational files such as the Prime ledger may remain dirty. The files that
    determine data, losses, training, and evaluation may not: otherwise the run
    receipt's commit would not identify the executable source.
    """

    root_text = _command_output(["git", "rev-parse", "--show-toplevel"]).strip()
    if not root_text or root_text.startswith("UNAVAILABLE:"):
        raise RuntimeError(f"cannot resolve repository root: {root_text}")
    root = Path(root_text).resolve()
    required = [
        root / "nextlat",
        root / "configs" / "nextlat_qwen3_1_7b_d1.json",
        root / "configs" / "nextlat_qwen3_1_7b_d4.json",
        root / "configs" / "nextlat_qwen3_1_7b_d4_long.json",
        root / "requirements-nextlat.txt",
        root / "PLAN.md",
        root / "tests" / "test_nextlat.py",
        root / "tests" / "test_nextlat_data.py",
        root / "tests" / "test_nextlat_official.py",
        root / "tests" / "test_nextlat_runtime.py",
        *(Path(path).resolve() for path in extra_paths),
    ]
    relative: list[str] = []
    for path in required:
        try:
            relative.append(str(path.relative_to(root)))
        except ValueError as error:
            raise RuntimeError(f"claim source is outside repository: {path}") from error

    status = _command_output(
        [
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            *relative,
        ]
    )
    if status.strip():
        raise RuntimeError(
            "claim-bearing source is not committed exactly:\n" + status.rstrip()
        )
    tracked = _command_output(["git", "ls-files", "--", *relative])
    if tracked.startswith("UNAVAILABLE:") or not tracked.strip():
        raise RuntimeError("claim-bearing source paths are not tracked")
    return git_commit()


def initialize_run_directory(
    run_dir: Path,
    config: Dict[str, Any],
    command: str,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "command.txt").write_text(command.rstrip() + "\n")
    (run_dir / "config.resolved.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )
    (run_dir / "git_commit.txt").write_text(
        _command_output(["git", "rev-parse", "HEAD"])
    )
    (run_dir / "git_status.txt").write_text(
        _command_output(["git", "status", "--short", "--branch"])
    )
    (run_dir / "git_diff.patch").write_text(_command_output(["git", "diff", "--binary"]))
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count(),
        "hostname": platform.node(),
    }
    (run_dir / "env.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n"
    )
    (run_dir / "metrics.jsonl").touch()
    resolved = sorted(
        f"{distribution.metadata['Name']}=={distribution.version}"
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    )
    (run_dir / "requirements.resolved.txt").write_text(
        "\n".join(resolved) + "\n"
    )
    (run_dir / "nvidia-smi.txt").write_text(
        _command_output(["nvidia-smi", "-q"])
    )
    write_status(
        run_dir,
        result_state="diagnostic",
        passed=[],
        failed_or_missing=["run not complete"],
        main_metric="pending",
        baseline="untouched Qwen3-1.7B",
        caveats=["active run"],
        next_target="complete the current rung",
    )


def write_jsonl(path: Path, record: Dict[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_status(
    run_dir: Path,
    *,
    result_state: str,
    passed: list[str],
    failed_or_missing: list[str],
    main_metric: str,
    baseline: str,
    caveats: list[str],
    next_target: str,
) -> None:
    lines = [
        f"Result state: {result_state}",
        f"Passed gates: {', '.join(passed) if passed else 'none'}",
        "Failed/missing gates: "
        + (", ".join(failed_or_missing) if failed_or_missing else "none"),
        f"Main metric: {main_metric}",
        f"Baseline comparison: {baseline}",
        f"Known caveats: {', '.join(caveats) if caveats else 'none'}",
        f"Next deletion/pruning target: {next_target}",
    ]
    (run_dir / "status.md").write_text("\n".join(lines) + "\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_rng_state() -> Dict[str, Any]:
    """Capture every RNG stream owned by the single-node trainer."""

    state: Dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    """Restore a checkpointed RNG state, failing closed on missing streams."""

    required = {"python", "torch_cpu"}
    missing = required - set(state)
    if missing:
        raise RuntimeError(f"RNG checkpoint is missing streams: {sorted(missing)}")
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available():
        cuda_state = state.get("torch_cuda")
        if cuda_state is None:
            raise RuntimeError("CUDA run requires checkpointed CUDA RNG state")
        if len(cuda_state) != torch.cuda.device_count():
            raise RuntimeError(
                "CUDA RNG device count differs: "
                f"{len(cuda_state)} != {torch.cuda.device_count()}"
            )
        torch.cuda.set_rng_state_all(cuda_state)


def save_checkpoint(
    *,
    checkpoint_dir: Path,
    model: torch.nn.Module,
    dynamics: torch.nn.Module,
    tokenizer: Any,
    train_state: Dict[str, Any],
    optimizer: Optional[torch.optim.Optimizer] = None,
    rng_state: Optional[Dict[str, Any]] = None,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    (checkpoint_dir / "INCOMPLETE").write_text("checkpoint write in progress\n")
    model_dir = checkpoint_dir / "model"
    model.save_pretrained(
        model_dir,
        safe_serialization=True,
        max_shard_size="4GB",
    )
    tokenizer.save_pretrained(model_dir)
    dynamics_state = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in dynamics.state_dict().items()
    }
    save_file(dynamics_state, checkpoint_dir / "dynamics.safetensors")
    if optimizer is not None:
        torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
    if rng_state is not None:
        torch.save(rng_state, checkpoint_dir / "rng_state.pt")
    (checkpoint_dir / "train_state.json").write_text(
        json.dumps(train_state, indent=2, sort_keys=True) + "\n"
    )
    hashes = {
        str(path.relative_to(checkpoint_dir)): _sha256_file(path)
        for path in sorted(checkpoint_dir.rglob("*"))
        if path.is_file() and path.name != "INCOMPLETE"
    }
    (checkpoint_dir / "sha256.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n"
    )
    (checkpoint_dir / "INCOMPLETE").unlink()
    (checkpoint_dir / "COMPLETE").write_text("ok\n")


def load_dynamics_checkpoint(
    dynamics: torch.nn.Module,
    checkpoint_dir: Path,
    device: torch.device,
    *,
    verify: bool = True,
) -> None:
    if verify:
        verify_checkpoint(checkpoint_dir)
    state = load_file(checkpoint_dir / "dynamics.safetensors", device=str(device))
    dynamics.load_state_dict(state)


def verify_checkpoint(checkpoint_dir: Path) -> None:
    if not (checkpoint_dir / "COMPLETE").exists():
        raise RuntimeError(f"checkpoint is not complete: {checkpoint_dir}")
    hash_path = checkpoint_dir / "sha256.json"
    if not hash_path.exists():
        raise RuntimeError(f"checkpoint has no hash receipt: {checkpoint_dir}")
    expected = json.loads(hash_path.read_text())
    if not isinstance(expected, dict) or not expected:
        raise RuntimeError(f"checkpoint hash receipt is empty: {checkpoint_dir}")
    for relative, digest in expected.items():
        path = checkpoint_dir / relative
        if not path.is_file():
            raise RuntimeError(f"checkpoint file is missing: {relative}")
        observed = _sha256_file(path)
        if observed != digest:
            raise RuntimeError(
                f"checkpoint hash mismatch for {relative}: {observed} != {digest}"
            )
