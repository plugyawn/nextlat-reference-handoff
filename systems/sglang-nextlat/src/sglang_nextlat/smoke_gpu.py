"""Real end-to-end GPU smoke for the pinned NEXTLAT SGLang plugin.

This entrypoint intentionally has no CPU or exact-decoding fallback. It launches
the real SGLang HTTP server, submits one variable-budget greedy batch, and writes
a deterministic pass receipt only after successful generation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import torch

from .admission import admit_sglang_source, admit_target_checkpoint, canonical_json_bytes
from .checkpoint import read_sidecar_metadata
from .constants import DEFAULT_MAX_DRAFT_TOKENS, MAX_DRAFT_TOKENS
from .runtime_receipt import build_runtime_receipt, write_runtime_receipt


def _require_installed_plugin() -> None:
    matches = [
        entry
        for entry in importlib.metadata.entry_points(group="sglang.srt.plugins")
        if entry.name == "nextlat" and entry.value == "sglang_nextlat.plugin:register"
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "GPU smoke requires exactly one installed sglang.srt.plugins entry "
            "nextlat=sglang_nextlat.plugin:register; install this package first"
        )


def _request_json(url: str, payload: dict[str, Any] | None, timeout: float) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status} from {url}: {body[:500]!r}")
    if not body:
        return None
    return json.loads(body)


def _wait_ready(base_url: str, process: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"SGLang server exited before readiness: {process.returncode}")
        try:
            _request_json(f"{base_url}/health", None, timeout=5.0)
            return
        except (OSError, ValueError, urllib.error.HTTPError) as exc:
            last_error = exc
            time.sleep(1.0)
    raise TimeoutError(f"SGLang server did not become ready: {last_error}")


def _server_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(args.model),
        "--load-format",
        "safetensors",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--device",
        "cuda",
        "--dtype",
        "bfloat16",
        "--attention-backend",
        "flashinfer",
        "--cuda-graph-backend-decode",
        "disabled",
        "--cuda-graph-backend-prefill",
        "disabled",
        "--disable-overlap-schedule",
        "--chunked-prefill-size",
        "-1",
        "--num-continuous-decode-steps",
        "1",
        "--tp-size",
        "1",
        "--pp-size",
        "1",
        "--dp-size",
        "1",
        "--speculative-algorithm",
        "NEXTLAT",
        "--speculative-draft-model-path",
        str(args.sidecar),
        "--speculative-num-steps",
        str(args.max_draft_tokens),
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        str(args.max_draft_tokens + 1),
    ]


def _terminate_owned_process(process: subprocess.Popen) -> None:
    # launch_server owns child engine processes. start_new_session=True gives
    # this smoke an exact process-group boundary, so cleanup cannot orphan GPUs.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)


def run(args: argparse.Namespace) -> Path:
    _require_installed_plugin()
    if not DEFAULT_MAX_DRAFT_TOKENS <= args.max_draft_tokens <= MAX_DRAFT_TOKENS:
        raise ValueError(
            "GPU smoke max-draft-tokens must be in "
            f"[{DEFAULT_MAX_DRAFT_TOKENS}, {MAX_DRAFT_TOKENS}] so it exercises "
            "the required length-eight row"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("NEXTLAT GPU smoke has no CPU fallback")
    if args.gpu_index < 0 or args.gpu_index >= torch.cuda.device_count():
        raise ValueError(
            f"gpu-index {args.gpu_index} is outside {torch.cuda.device_count()} devices"
        )

    sidecar = read_sidecar_metadata(args.sidecar)
    source = admit_sglang_source()
    target = admit_target_checkpoint(
        args.model,
        expected_manifest_sha256=sidecar.target_checkpoint_sha256,
        contract=sidecar.contract,
    )
    command = _server_command(args)
    command_digest = hashlib.sha256(canonical_json_bytes(command)).hexdigest()

    prompts = [
        "Reply with only the number: 2 + 3 =",
        "Complete this sequence with only one number: 1, 1, 2, 3, 5,",
        "In one short sentence, explain why the sky looks blue.",
        "Continue briefly: The quickest route is",
    ]
    # After prefill emits the anchor, these budgets leave first-decode proposal
    # lengths [0, 2, 4, 8] while reserving one exact target bonus.
    budgets = [2, 4, 6, 10]
    payload = {
        "text": prompts,
        "sampling_params": [
            {
                "max_new_tokens": budget,
                "temperature": 0.0,
                "top_k": 1,
                "top_p": 1.0,
                "min_p": 0.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                "repetition_penalty": 1.0,
                "n": 1,
                "ignore_eos": True,
            }
            for budget in budgets
        ],
        "return_logprob": False,
    }
    base_url = f"http://127.0.0.1:{args.port}"
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    environment["SGLANG_PLUGINS"] = "nextlat"

    with tempfile.TemporaryDirectory(prefix="nextlat-gpu-smoke-") as directory:
        log_path = Path(directory) / "server.log"
        with log_path.open("w+b") as log:
            process = subprocess.Popen(
                command,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                _wait_ready(base_url, process, args.startup_timeout)
                response = _request_json(
                    f"{base_url}/generate", payload, timeout=args.request_timeout
                )
            except Exception as exc:
                _terminate_owned_process(process)
                log.flush()
                log.seek(0)
                tail = log.read().decode("utf-8", errors="replace")[-12000:]
                raise RuntimeError(f"NEXTLAT GPU smoke failed: {exc}\n{tail}") from exc
            finally:
                _terminate_owned_process(process)

    if not isinstance(response, list) or len(response) != len(prompts):
        raise RuntimeError("NEXTLAT batch response shape differs from the submitted batch")
    outputs: list[str] = []
    for index, item in enumerate(response):
        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            raise RuntimeError(f"NEXTLAT response {index} has no text output")
        outputs.append(item["text"])
    if any(not output for output in outputs):
        raise RuntimeError("NEXTLAT GPU smoke returned an empty generation")

    properties = torch.cuda.get_device_properties(args.gpu_index)
    receipt = build_runtime_receipt(
        sidecar=sidecar,
        target=target,
        source=source,
        evidence={
            "result": "pass",
            "scope": "end_to_end_sglang_http_smoke",
            "gpu": {
                "name": properties.name,
                "compute_capability": [properties.major, properties.minor],
            },
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "server_command_sha256": command_digest,
            "prompt_batch_sha256": hashlib.sha256(
                canonical_json_bytes(prompts)
            ).hexdigest(),
            "max_new_tokens": budgets,
            "first_decode_proposal_lengths": [0, 2, 4, 8],
            "output_sha256": [
                hashlib.sha256(output.encode("utf-8")).hexdigest()
                for output in outputs
            ],
        },
        max_draft_tokens=args.max_draft_tokens,
    )
    return write_runtime_receipt(args.receipt, receipt)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument(
        "--max-draft-tokens", type=int, default=DEFAULT_MAX_DRAFT_TOKENS
    )
    parser.add_argument("--port", type=int, default=31000)
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    receipt_path = run(args)
    print(receipt_path)


if __name__ == "__main__":
    main()
