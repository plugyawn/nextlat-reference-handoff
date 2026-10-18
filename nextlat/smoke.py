"""Real-Qwen loss, gradient, and full-checkpoint parity smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .core import NextLatDynamics, NextLatLossConfig, NextLatQwen
from .run_utils import (
    assert_claim_source_committed,
    initialize_run_directory,
    load_dynamics_checkpoint,
    save_checkpoint,
    verify_checkpoint,
    write_jsonl,
    write_status,
)


FIXED_CONVERSATIONS = [
    (
        "If x + 7 = 19, solve for x and put the answer in a box.",
        "<think>Subtracting 7 from both sides gives x = 12.</think>\n\n"
        "Therefore, \\boxed{12}.",
    ),
    (
        "A bag has 3 red and 5 blue marbles. How many marbles are there?",
        "<think>The total is the sum of the two counts: 3 + 5 = 8.</think>\n\n"
        "There are \\boxed{8} marbles.",
    ),
]


def _fixed_batch(tokenizer: Any, device: torch.device) -> Dict[str, torch.Tensor]:
    encoded = []
    for prompt, answer in FIXED_CONVERSATIONS:
        prompt_messages = [{"role": "user", "content": prompt}]
        full_messages = prompt_messages + [{"role": "assistant", "content": answer}]
        prompt_ids = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        full_ids = tokenizer.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=True,
        )
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise RuntimeError("fixed-batch chat prefix mismatch")
        encoded.append((full_ids, len(prompt_ids)))
    length = max(len(ids) for ids, _ in encoded)
    input_ids = torch.full(
        (len(encoded), length), tokenizer.pad_token_id, dtype=torch.long
    )
    attention = torch.zeros_like(input_ids, dtype=torch.bool)
    assistant = torch.zeros_like(input_ids, dtype=torch.bool)
    for row, (ids, prompt_length) in enumerate(encoded):
        input_ids[row, : len(ids)] = torch.tensor(ids)
        attention[row, : len(ids)] = True
        assistant[row, prompt_length : len(ids)] = True
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention.to(device),
        "assistant_loss_mask": assistant.to(device),
    }


def _grad_norm(module: torch.nn.Module) -> float:
    pieces = [
        parameter.grad.detach().float().pow(2).sum()
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    return float(torch.stack(pieces).sum().sqrt().item()) if pieces else 0.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_recorded_reference_tests(run_dir: Path) -> Dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "unittest",
        "tests.test_nextlat",
        "tests.test_nextlat_data",
        "tests.test_nextlat_official",
        "tests.test_nextlat_runtime",
        "-v",
    ]
    started = time.time()
    test_environment = os.environ.copy()
    upstream_root = test_environment.get("NEXTLAT_OFFICIAL_ROOT")
    default_upstream = Path.home() / "nextlat-official"
    if upstream_root is None and default_upstream.is_dir():
        upstream_root = str(default_upstream)
        test_environment["NEXTLAT_OFFICIAL_ROOT"] = upstream_root
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        env=test_environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path = run_dir / "official_parity_tests.log"
    log_path.write_text(completed.stdout)
    match = re.search(r"Ran (\d+) tests?", completed.stdout)
    test_count = int(match.group(1)) if match else 0
    receipt = {
        "command": command,
        "returncode": completed.returncode,
        "tests_run": test_count,
        "elapsed_seconds": time.time() - started,
        "log_sha256": _sha256(log_path),
        "upstream_checkout": upstream_root,
    }
    (run_dir / "official_parity_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    if completed.returncode != 0 or test_count < 21 or "skipped=" in completed.stdout:
        raise RuntimeError(
            "recorded pinned-upstream parity suite failed or was incomplete; "
            f"see {log_path}"
        )
    return receipt


@torch.inference_mode()
def _snapshot(joint: NextLatQwen, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = joint(**batch)
    for name in ("loss", "ce", "smooth_l1", "kl"):
        if not bool(torch.isfinite(outputs[name])):
            raise RuntimeError(f"non-finite fixed-batch {name}: {outputs[name]}")
    return {
        "loss": outputs["loss"].detach().cpu(),
        "ce": outputs["ce"].detach().cpu(),
        "smooth_l1": outputs["smooth_l1"].detach().cpu(),
        "kl": outputs["kl"].detach().cpu(),
        "predicted_prefix": outputs["predicted_next_states"][:, :3, :16]
        .detach()
        .cpu(),
    }


def run_smoke(config_path: Path, run_dir: Path) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("real-model smoke requires CUDA")
    source_commit = assert_claim_source_committed([config_path])
    config = json.loads(config_path.read_text())
    horizon = int(config["nextlat"]["horizon"])
    if horizon not in (1, 4):
        raise RuntimeError("the claim-contract smoke permits only d=1 or d=4")
    torch.manual_seed(int(config["training"]["seed"]))
    torch.cuda.manual_seed_all(int(config["training"]["seed"]))
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    initialize_run_directory(
        run_dir,
        config,
        " ".join(shlex.quote(part) for part in sys.argv),
    )
    reference_receipt = _run_recorded_reference_tests(run_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["id"],
        revision=config["model"]["revision"],
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["id"],
        revision=config["model"]["revision"],
        torch_dtype=torch.float32,
        attn_implementation=config["model"]["attention"],
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to(device)
    model.config.use_cache = False
    if model.get_input_embeddings().weight.data_ptr() != model.lm_head.weight.data_ptr():
        raise RuntimeError("expected Qwen input and output weights to be tied")
    dynamics = NextLatDynamics(
        int(model.config.hidden_size),
        projection_factor=float(config["nextlat"]["projection_factor"]),
        dropout=float(config["nextlat"]["dropout"]),
    ).to(device=device, dtype=torch.float32)
    loss_config = NextLatLossConfig(
        lambda_h=float(config["nextlat"]["lambda_h"]),
        lambda_kl=float(config["nextlat"]["lambda_kl"]),
        vocab_chunk_tokens=int(config["nextlat"]["vocab_chunk_tokens"]),
        horizon=horizon,
    )
    joint = NextLatQwen(model, dynamics, loss_config).to(device)
    batch = _fixed_batch(tokenizer, device)

    joint.train()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = joint(**batch)
    auxiliary = (
        loss_config.lambda_h * outputs["smooth_l1"]
        + loss_config.lambda_kl * outputs["kl"]
    )
    auxiliary.backward()
    theta_aux_grad_norm = _grad_norm(model)
    psi_aux_grad_norm = _grad_norm(dynamics)
    if theta_aux_grad_norm <= 0 or psi_aux_grad_norm <= 0:
        raise RuntimeError(
            f"auxiliary gradient gate failed: theta={theta_aux_grad_norm}, "
            f"psi={psi_aux_grad_norm}"
        )
    model.zero_grad(set_to_none=True)
    dynamics.zero_grad(set_to_none=True)

    joint.eval()
    before = _snapshot(joint, batch)
    checkpoint_dir = run_dir / "checkpoint-roundtrip"
    save_checkpoint(
        checkpoint_dir=checkpoint_dir,
        model=model,
        dynamics=dynamics,
        tokenizer=tokenizer,
        train_state={"kind": "loss-parity-smoke", "optimizer_steps": 0},
    )

    del outputs, auxiliary, joint, dynamics, model
    torch.cuda.empty_cache()
    verify_checkpoint(checkpoint_dir)
    restored_model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir / "model",
        torch_dtype=torch.float32,
        attn_implementation=config["model"]["attention"],
        low_cpu_mem_usage=True,
    ).to(device)
    restored_model.config.use_cache = False
    if (
        restored_model.get_input_embeddings().weight.data_ptr()
        != restored_model.lm_head.weight.data_ptr()
    ):
        raise RuntimeError("restored Qwen checkpoint did not preserve tied weights")
    restored_dynamics = NextLatDynamics(
        int(restored_model.config.hidden_size),
        projection_factor=float(config["nextlat"]["projection_factor"]),
        dropout=float(config["nextlat"]["dropout"]),
    ).to(device=device, dtype=torch.float32)
    load_dynamics_checkpoint(restored_dynamics, checkpoint_dir, device)
    restored = NextLatQwen(restored_model, restored_dynamics, loss_config).to(device).eval()
    after = _snapshot(restored, batch)
    max_abs_difference = 0.0
    for name in before:
        difference = float((before[name].float() - after[name].float()).abs().max().item())
        max_abs_difference = max(max_abs_difference, difference)
        torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)

    result = {
        "type": "loss_parity_smoke",
        "result_state": "claim-grade",
        "source_code_commit": source_commit,
        "model": config["model"],
        "upstream_nextlat": config["upstream_nextlat"],
        "batch_shape": list(batch["input_ids"].shape),
        "loss": float(before["loss"].item()),
        "ce": float(before["ce"].item()),
        "smooth_l1": float(before["smooth_l1"].item()),
        "kl": float(before["kl"].item()),
        "theta_aux_grad_norm": theta_aux_grad_norm,
        "psi_aux_grad_norm": psi_aux_grad_norm,
        "checkpoint_max_abs_difference": max_abs_difference,
        "checkpoint_complete": (checkpoint_dir / "COMPLETE").exists(),
        "official_parity_receipt": reference_receipt,
    }
    write_jsonl(run_dir / "metrics.jsonl", result)
    (run_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    write_status(
        run_dir,
        result_state="claim-grade",
        passed=[
            "pinned official dynamics and loss methods match executable source",
            "CUDA count-weighted microbatch gradients match full-batch gradients",
            f"real Qwen d={horizon} loss finite",
            "auxiliary gradients reach theta and psi",
            "full checkpoint roundtrip is bit-exact",
        ],
        failed_or_missing=[],
        main_metric=f"checkpoint max abs difference {max_abs_difference}",
        baseline="pinned official NextLat commit 3770be6 plus real Qwen3-1.7B",
        caveats=[
            "pinned official executable drops terminal shifted KL position"
        ],
        next_target="tiny joint FFT engineering rung",
    )
    print(json.dumps(result, sort_keys=True), flush=True)


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    run_smoke(args.config, args.run_dir)


if __name__ == "__main__":
    main()
