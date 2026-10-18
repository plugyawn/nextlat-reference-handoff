"""Fail-closed DeepSeek/SGLang-path evaluation for the NEXTLAT worker.

The name refers to the DSpark ragged verify/control primitives used by NEXTLAT.
It does not claim a model-vs-model comparison with a DSpark drafter: no matched
Qwen3-1.7B DSpark drafter/confidence checkpoint is admitted by this package.

The harness launches one exact-greedy target server and then three NEXTLAT
servers sequentially.  It compares raw output token IDs at batch sizes 1, 2,
4, and 8 for maximum draft lengths 2, 4, and 8.  A receipt is published only
after every response and every SGLang speculative metric passes strict checks.
There is no alternate server, CPU path, or text-only parity fallback.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import importlib.metadata
import json
import math
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

from .admission import (
    admit_sglang_source,
    admit_target_checkpoint,
    canonical_json_bytes,
)
from .checkpoint import read_sidecar_metadata
from .constants import AUDITED_SGLANG_COMMIT, SERVING_ATTENTION_BACKENDS
from .smoke_gpu import (
    _request_json,
    _require_installed_plugin,
    _terminate_owned_process,
    _wait_ready,
)

EVAL_SCHEMA_VERSION = 2
EVAL_BATCH_SIZES = (1, 2, 4, 8)
EVAL_MAX_DRAFT_TOKENS = (2, 4, 8)
_SERVER_KINDS = frozenset({"exact_greedy", "nextlat_dspark_verify"})
_REGISTERED_INPUT_KEYS = frozenset(
    {
        "schema_version",
        "authority",
        "prepared_manifest_sha256",
        "recursive_holdout_record_digest",
        "selection",
        "inputs",
    }
)
_REGISTERED_INPUT_ITEM_KEYS = frozenset({"ordinal", "problem_hash", "input_ids"})
_SHA256_CHARS = frozenset("0123456789abcdef")
_REGISTERED_INPUT_AUTHORITY = "registered_prepared_recursive_holdout_v1"
_REGISTERED_INPUT_SELECTION = "first_8_in_registered_holdout_order"
_REGISTERED_PREPARED_MANIFEST_SHA256 = (
    "390df21664f57c1df0e44cb33cefcbd37d82d908458730a968aec4485e13466f"
)
_REGISTERED_HOLDOUT_RECORD_DIGEST = (
    "96e6d57b92ce1a4f5314665fa6dfb0b9f515a062f5cc111d3b61f3251843a3f1"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _finite_number(value: Any, *, label: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise RuntimeError(f"{label} must be finite and >= {minimum}")
    return result


def _nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"{label} must be a non-negative integer")
    return value


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"required runtime package is not installed: {name}") from exc


def _server_command(
    args: argparse.Namespace,
    *,
    kind: str,
    max_draft_tokens: int | None,
) -> list[str]:
    if kind not in _SERVER_KINDS:
        raise ValueError(f"unknown server kind: {kind}")
    if (kind == "exact_greedy") != (max_draft_tokens is None):
        raise ValueError("exact-greedy alone must omit max_draft_tokens")
    if max_draft_tokens is not None and max_draft_tokens not in EVAL_MAX_DRAFT_TOKENS:
        raise ValueError(f"unsupported evaluation draft maximum: {max_draft_tokens}")

    command = [
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
        "--disable-radix-cache",
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
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--enable-metrics",
    ]
    if kind == "nextlat_dspark_verify":
        assert max_draft_tokens is not None
        command.extend(
            [
                "--speculative-algorithm",
                "NEXTLAT",
                "--speculative-draft-model-path",
                str(args.sidecar),
                "--speculative-num-steps",
                str(max_draft_tokens),
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                str(max_draft_tokens + 1),
            ]
        )
    return command


def _server_environment(*, kind: str, gpu_index: int) -> dict[str, str]:
    if kind not in _SERVER_KINDS:
        raise ValueError(f"unknown server kind: {kind}")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    environment["TOKENIZERS_PARALLELISM"] = "false"
    if kind == "nextlat_dspark_verify":
        environment["SGLANG_PLUGINS"] = "nextlat"
    else:
        environment.pop("SGLANG_PLUGINS", None)
    return environment


def _assert_port_unused(port: int) -> None:
    if port < 1 or port > 65535:
        raise ValueError("server port must be in [1, 65535]")
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.settimeout(0.25)
        result = probe.connect_ex(("127.0.0.1", port))
        if result == 0:
            raise RuntimeError(f"server port 127.0.0.1:{port} is already in use")
        if result != errno.ECONNREFUSED:
            raise RuntimeError(
                f"cannot prove server port 127.0.0.1:{port} is unused: errno {result}"
            )
    finally:
        probe.close()


def _same_local_path(observed: Any, expected: Path) -> bool:
    if not isinstance(observed, str) or not observed:
        return False
    return Path(observed).expanduser().resolve() == expected.expanduser().resolve()


def _assert_server_identity(
    base_url: str,
    *,
    args: argparse.Namespace,
    kind: str,
    max_draft_tokens: int | None,
    timeout: float,
) -> dict[str, Any]:
    server_info = _request_json(f"{base_url}/server_info", None, timeout=timeout)
    model_info = _request_json(f"{base_url}/model_info", None, timeout=timeout)
    if not isinstance(server_info, dict) or not isinstance(model_info, dict):
        raise RuntimeError("SGLang identity endpoints did not return JSON objects")

    expected_attention_backend = getattr(args, "attention_backend", "flashinfer")
    if expected_attention_backend not in SERVING_ATTENTION_BACKENDS:
        raise ValueError(
            "identity check attention backend is outside the admitted set: "
            f"{expected_attention_backend!r}"
        )
    expected_server = {
        "load_format": "safetensors",
        "device": "cuda",
        "dtype": "bfloat16",
        "attention_backend": expected_attention_backend,
        "cuda_graph_backend_decode": "disabled",
        "cuda_graph_backend_prefill": "disabled",
        "disable_overlap_schedule": True,
        "disable_radix_cache": True,
        "chunked_prefill_size": -1,
        "num_continuous_decode_steps": 1,
        "tp_size": 1,
        "pp_size": 1,
        "dp_size": 1,
        "enable_metrics": True,
    }
    for name, expected in expected_server.items():
        if server_info.get(name) != expected:
            raise RuntimeError(
                f"SGLang /server_info {name}={server_info.get(name)!r}, expected {expected!r}"
            )
    graph_config = server_info.get("cuda_graph_config")
    if (
        not isinstance(graph_config, dict)
        or not all(
            isinstance(graph_config.get(phase), dict)
            and graph_config[phase].get("backend") == "disabled"
            for phase in ("decode", "prefill")
        )
    ):
        raise RuntimeError("SGLang resolved CUDA graph phases are not both disabled")
    if not math.isclose(
        _finite_number(server_info.get("mem_fraction_static"), label="mem_fraction_static"),
        args.mem_fraction_static,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("SGLang /server_info memory fraction differs from launch")
    if not _same_local_path(server_info.get("model_path"), args.model):
        raise RuntimeError("SGLang /server_info model path differs from launch")

    if kind == "exact_greedy":
        for name in (
            "speculative_algorithm",
            "speculative_draft_model_path",
            "speculative_num_steps",
            "speculative_eagle_topk",
            "speculative_num_draft_tokens",
        ):
            if server_info.get(name) is not None:
                raise RuntimeError(f"exact server unexpectedly enables {name}")
    else:
        assert max_draft_tokens is not None
        expected_spec = {
            "speculative_algorithm": "NEXTLAT",
            "speculative_num_steps": max_draft_tokens,
            "speculative_eagle_topk": 1,
            "speculative_num_draft_tokens": max_draft_tokens + 1,
        }
        for name, expected in expected_spec.items():
            if server_info.get(name) != expected:
                raise RuntimeError(
                    f"SGLang /server_info {name}={server_info.get(name)!r}, expected {expected!r}"
                )
        if not _same_local_path(server_info.get("speculative_draft_model_path"), args.sidecar):
            raise RuntimeError("SGLang /server_info draft path differs from sidecar")

    if not _same_local_path(model_info.get("model_path"), args.model):
        raise RuntimeError("SGLang /model_info model path differs from launch")
    expected_model = {
        "load_format": "safetensors",
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
    }
    for name, expected in expected_model.items():
        if model_info.get(name) != expected:
            raise RuntimeError(
                f"SGLang /model_info {name}={model_info.get(name)!r}, expected {expected!r}"
            )
    return {
        "admitted": True,
        "server_info_sha256": _json_sha256(server_info),
        "model_info_sha256": _json_sha256(model_info),
        "server": {
            name: server_info[name]
            for name in (
                "model_path",
                *expected_server,
                "cuda_graph_config",
                "mem_fraction_static",
                "speculative_algorithm",
                "speculative_draft_model_path",
                "speculative_num_steps",
                "speculative_eagle_topk",
                "speculative_num_draft_tokens",
                "version",
            )
        },
        "model": {
            name: model_info.get(name)
            for name in (
                "model_path",
                "served_model_name",
                "tokenizer_path",
                "load_format",
                "model_type",
                "architectures",
                "weight_version",
            )
        },
    }


def _generation_payload(
    input_ids: list[list[int]], *, max_new_tokens: int, request_prefix: str
) -> dict[str, Any]:
    sampling = {
        "max_new_tokens": max_new_tokens,
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
    return {
        "rid": [f"{request_prefix}-{index}" for index in range(len(input_ids))],
        "input_ids": input_ids,
        "sampling_params": [dict(sampling) for _ in input_ids],
        "return_logprob": False,
        "stream": False,
        "log_metrics": True,
    }


def _response_items(response: Any, *, expected_rids: list[str]) -> list[dict[str, Any]]:
    if len(expected_rids) == 1 and isinstance(response, dict):
        response = [response]
    if not isinstance(response, list) or len(response) != len(expected_rids):
        raise RuntimeError(
            "SGLang response batch shape differs: "
            f"expected {len(expected_rids)}, got {type(response).__name__}"
        )
    if any(not isinstance(item, dict) for item in response):
        raise RuntimeError("SGLang response batch contains a non-object item")
    by_rid: dict[str, dict[str, Any]] = {}
    for item in response:
        meta = item.get("meta_info")
        rid = None if not isinstance(meta, dict) else meta.get("id")
        if not isinstance(rid, str) or not rid or rid in by_rid:
            raise RuntimeError("SGLang response has a missing or duplicate request id")
        by_rid[rid] = item
    if set(by_rid) != set(expected_rids):
        raise RuntimeError("SGLang response request-id roster differs from the request")
    return [by_rid[rid] for rid in expected_rids]


def _nearest_rank(histogram: list[int], quantile: float) -> int:
    rank = max(1, math.ceil(quantile * sum(histogram)))
    cumulative = 0
    for value, count in enumerate(histogram):
        cumulative += count
        if cumulative >= rank:
            return value
    raise RuntimeError("acceptance histogram is empty")


def _acceptance_summary(histogram: list[int], *, max_draft_tokens: int) -> dict[str, Any]:
    normalized = histogram + [0] * (max_draft_tokens + 1 - len(histogram))
    verify_count = sum(normalized)
    if verify_count <= 0:
        raise RuntimeError("acceptance histogram has no verification rounds")
    accepted = sum(index * count for index, count in enumerate(normalized))
    mean = accepted / verify_count
    return {
        "verification_rounds": verify_count,
        "accepted_draft_prefix_length": {
            "histogram": normalized,
            "mean": mean,
            "median": _nearest_rank(normalized, 0.5),
            "p90": _nearest_rank(normalized, 0.9),
            "maximum": max(index for index, count in enumerate(normalized) if count),
            "survival_probability_at_least": [
                sum(normalized[length:]) / verify_count
                for length in range(1, max_draft_tokens + 1)
            ],
        },
        "emitted_output_tokens_per_verify": {
            "mean": 1.0 + mean,
            "median": 1 + _nearest_rank(normalized, 0.5),
            "p90": 1 + _nearest_rank(normalized, 0.9),
        },
    }


def _aggregate_acceptance(
    cases: list[dict[str, Any]], *, max_draft_tokens: int
) -> dict[str, Any]:
    histogram = [0] * (max_draft_tokens + 1)
    for case in cases:
        for request in case["requests"]:
            observed = request["speculative"]["accept_histogram"]
            histogram = [left + right for left, right in zip(histogram, observed)]
    return _acceptance_summary(histogram, max_draft_tokens=max_draft_tokens)


def _target_forward_observation(
    *,
    prompt_tokens: int,
    completion_tokens: int,
    spec: dict[str, Any] | None,
) -> dict[str, Any]:
    if spec is None:
        decode_positions = max(0, completion_tokens - 1)
        return {
            "authority": "exact_from_non_speculative_token_counts",
            "prefill_positions": prompt_tokens,
            "decode_positions": decode_positions,
            "total_positions": prompt_tokens + decode_positions,
        }

    # Pinned SGLang computes spec_num_proposed_drafts as verify_ct * configured
    # draft width. NEXTLAT has a compact ragged final row, so that counter can
    # exceed the positions actually sent through the target. Preserve it as an
    # explicit configured-width upper bound; never relabel it as exact work.
    configured_verify = spec["num_proposed_drafts"] + spec["verify_count"]
    return {
        "authority": "sglang_configured_width_upper_bound",
        "exact_compact_positions": None,
        "prefill_positions": prompt_tokens,
        "reported_verify_positions_upper_bound": configured_verify,
        "reported_total_positions_upper_bound": prompt_tokens + configured_verify,
        "unavailable_reason": (
            "pinned SGLang does not expose NEXTLAT per-round compact ragged "
            "proposal lengths in response meta_info"
        ),
    }


def _parse_request_result(
    item: dict[str, Any],
    *,
    request_id: str,
    registered_input: dict[str, Any],
    expected_completion_tokens: int,
    max_draft_tokens: int | None,
) -> dict[str, Any]:
    output_ids = item.get("output_ids")
    if (
        not isinstance(output_ids, list)
        or len(output_ids) != expected_completion_tokens
        or any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0
            for token in output_ids
        )
    ):
        raise RuntimeError("SGLang output_ids are missing or differ from the exact budget")
    meta = item.get("meta_info")
    if not isinstance(meta, dict):
        raise RuntimeError("SGLang response has no meta_info object")
    if meta.get("id") != request_id:
        raise RuntimeError("SGLang response request id differs from the submitted id")
    prompt_tokens = _nonnegative_int(meta.get("prompt_tokens"), label="prompt_tokens")
    input_ids = registered_input["input_ids"]
    if prompt_tokens != len(input_ids):
        raise RuntimeError("SGLang prompt_tokens differs from registered input_ids length")
    completion_tokens = _nonnegative_int(
        meta.get("completion_tokens"), label="completion_tokens"
    )
    if completion_tokens != expected_completion_tokens:
        raise RuntimeError(
            f"completion_tokens={completion_tokens}, expected {expected_completion_tokens}"
        )
    finish_reason = meta.get("finish_reason")
    if not isinstance(finish_reason, dict) or finish_reason.get("type") != "length":
        raise RuntimeError("ignore_eos evaluation request did not finish by length")

    request_received = _finite_number(
        meta.get("request_received_ts"), label="request_received_ts"
    )
    prefill_finished = _finite_number(
        meta.get("prefill_finished_time"), label="prefill_finished_time"
    )
    prefill_completion_latency = prefill_finished - request_received
    if prefill_completion_latency < 0:
        raise RuntimeError("prefill completion precedes request receipt")
    e2e_latency = _finite_number(meta.get("e2e_latency"), label="e2e_latency")
    decode_throughput = _finite_number(
        meta.get("decode_throughput"), label="decode_throughput", minimum=1e-300
    )
    decode_latency = (completion_tokens - 1) / decode_throughput
    time_to_first_token = e2e_latency - decode_latency
    if time_to_first_token < -1e-9:
        raise RuntimeError("derived server time-to-first-token is negative")
    time_to_first_token = max(0.0, time_to_first_token)

    spec: dict[str, Any] | None = None
    if max_draft_tokens is None:
        forbidden = sorted(name for name in meta if name.startswith("spec_"))
        if forbidden:
            raise RuntimeError(
                f"exact-greedy response unexpectedly contains speculative metrics: {forbidden}"
            )
    else:
        accept_length = _finite_number(
            meta.get("spec_accept_length"), label="spec_accept_length"
        )
        verify_count = _nonnegative_int(
            meta.get("spec_verify_ct"), label="spec_verify_ct"
        )
        if verify_count <= 0:
            raise RuntimeError("NEXTLAT request executed no target verify round")
        num_correct = _nonnegative_int(
            meta.get("spec_num_correct_drafts"), label="spec_num_correct_drafts"
        )
        num_proposed = _nonnegative_int(
            meta.get("spec_num_proposed_drafts"), label="spec_num_proposed_drafts"
        )
        expected_proposed = verify_count * max_draft_tokens
        if num_proposed != expected_proposed:
            raise RuntimeError(
                f"SGLang configured-width proposal count {num_proposed} != {expected_proposed}"
            )
        histogram = meta.get("spec_accept_histogram")
        canonical_histogram = meta.get("spec_correct_drafts_histogram")
        if histogram != canonical_histogram:
            raise RuntimeError("SGLang canonical and compatibility acceptance histograms differ")
        if (
            not isinstance(histogram, list)
            or not histogram
            or any(
                isinstance(count, bool) or not isinstance(count, int) or count < 0
                for count in histogram
            )
            or len(histogram) > max_draft_tokens + 1
        ):
            raise RuntimeError("spec_accept_histogram is not a valid bounded count vector")
        if sum(histogram) != verify_count:
            raise RuntimeError("spec_accept_histogram count differs from spec_verify_ct")
        histogram_correct = sum(index * count for index, count in enumerate(histogram))
        if histogram_correct != num_correct:
            raise RuntimeError(
                "spec_accept_histogram weighted count differs from spec_num_correct_drafts"
            )
        if completion_tokens != 1 + verify_count + num_correct:
            raise RuntimeError(
                "completion_tokens differs from initial token plus verifies plus accepted drafts"
            )
        expected_accept_length = completion_tokens / verify_count
        if not math.isclose(
            accept_length, expected_accept_length, rel_tol=0.0, abs_tol=1e-12
        ):
            raise RuntimeError(
                "spec_accept_length differs from completion_tokens/spec_verify_ct"
            )
        accept_rate = _finite_number(
            meta.get("spec_accept_rate"), label="spec_accept_rate"
        )
        expected_accept_rate = num_correct / num_proposed
        if not math.isclose(
            accept_rate, expected_accept_rate, rel_tol=0.0, abs_tol=1e-12
        ):
            raise RuntimeError(
                "spec_accept_rate differs from correct/configured-width proposed drafts"
            )
        summary = _acceptance_summary(
            histogram, max_draft_tokens=max_draft_tokens
        )
        spec = {
            "accept_histogram": summary["accepted_draft_prefix_length"]["histogram"],
            "num_correct_drafts": num_correct,
            "num_proposed_drafts": num_proposed,
            "verify_count": verify_count,
            "acceptance_summary": summary,
            "sglang_reported": {
                "completion_tokens_div_verify_count": accept_length,
                "correct_div_configured_width_proposals": accept_rate,
            },
        }

    return {
        "request_id": request_id,
        "registered_input": {
            "ordinal": registered_input["ordinal"],
            "problem_hash": registered_input["problem_hash"],
            "input_ids_sha256": _json_sha256(input_ids),
        },
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "output_ids": output_ids,
        "output_ids_sha256": _json_sha256(output_ids),
        "timing_seconds": {
            "server_time_to_first_token": time_to_first_token,
            "server_prefill_completion_latency": prefill_completion_latency,
            "server_e2e_latency": e2e_latency,
            "server_decode_latency": decode_latency,
            "server_decode_throughput_tokens_per_second": decode_throughput,
        },
        "speculative": spec,
        "target_forward_positions": _target_forward_observation(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            spec=spec,
        ),
    }


def _query_gpu_memory_mib(gpu_index: int) -> float:
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu_index}",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    first_line = completed.stdout.strip().splitlines()[0]
    try:
        value = float(first_line)
    except (IndexError, ValueError) as exc:
        raise RuntimeError(f"nvidia-smi memory.used is not numeric: {first_line!r}") from exc
    if not math.isfinite(value) or value < 0:
        raise RuntimeError("nvidia-smi memory.used is not finite and non-negative")
    return value


class _GpuMemorySampler:
    def __init__(self, *, gpu_index: int, interval: float) -> None:
        self.gpu_index = gpu_index
        self.interval = interval
        self.samples: list[float] = []
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("GPU memory sampler already started")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.samples.append(_query_gpu_memory_mib(self.gpu_index))
            except Exception as exc:  # evidence records absence; no inferred fallback
                self.error = f"{type(exc).__name__}: {exc}"
                return
            self._stop.wait(self.interval)

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(10.0, 2 * self.interval))
            if self._thread.is_alive():
                raise RuntimeError("GPU memory sampler did not stop")
        if not self.samples:
            return {
                "authority": "nvidia_smi_memory.used",
                "available": False,
                "error": self.error or "no sample collected",
                "sample_count": 0,
                "peak_mib": None,
            }
        return {
            "authority": "nvidia_smi_memory.used",
            "available": True,
            "error": self.error,
            "sample_count": len(self.samples),
            "peak_mib": max(self.samples),
        }


def _server_observation(base_url: str, *, timeout: float) -> dict[str, Any]:
    info = _request_json(f"{base_url}/server_info", None, timeout=timeout)
    if not isinstance(info, dict):
        raise RuntimeError("SGLang /server_info response is not an object")
    states = info.get("internal_states")
    if (
        not isinstance(states, list)
        or len(states) != 1
        or not isinstance(states[0], dict)
    ):
        raise RuntimeError("SGLang evaluation requires exactly one scheduler state")
    state = states[0]
    if state.get("world_size") != 1:
        raise RuntimeError("SGLang evaluation server world_size is not one")
    memory_usage = state.get("memory_usage")
    if not isinstance(memory_usage, dict):
        raise RuntimeError("SGLang /server_info exposes no memory_usage object")
    return {
        "world_size": 1,
        "memory_usage": memory_usage,
        "last_gen_throughput": state.get("last_gen_throughput"),
        "avg_spec_accept_length": state.get("avg_spec_accept_length"),
    }


def _measure_case(
    *,
    base_url: str,
    inputs: list[dict[str, Any]],
    batch_size: int,
    repetition: int,
    max_new_tokens: int,
    max_draft_tokens: int | None,
    gpu_index: int,
    memory_sample_interval: float,
    request_timeout: float,
) -> dict[str, Any]:
    selected = inputs[:batch_size]
    mode_label = "exact" if max_draft_tokens is None else f"nextlat-m{max_draft_tokens}"
    payload = _generation_payload(
        [item["input_ids"] for item in selected],
        max_new_tokens=max_new_tokens,
        request_prefix=f"{mode_label}-b{batch_size}-r{repetition}",
    )
    before = _server_observation(base_url, timeout=request_timeout)
    memory = _GpuMemorySampler(
        gpu_index=gpu_index, interval=memory_sample_interval
    )
    memory.start()
    started = time.monotonic()
    try:
        response = _request_json(
            f"{base_url}/generate", payload, timeout=request_timeout
        )
    finally:
        wall_seconds = time.monotonic() - started
        memory_observation = memory.stop()
    request_ids = payload["rid"]
    items = _response_items(response, expected_rids=request_ids)
    requests = [
        _parse_request_result(
            item,
            request_id=request_id,
            registered_input=registered_input,
            expected_completion_tokens=max_new_tokens,
            max_draft_tokens=max_draft_tokens,
        )
        for item, request_id, registered_input in zip(items, request_ids, selected)
    ]
    after = _server_observation(base_url, timeout=request_timeout)
    total_output_tokens = sum(item["completion_tokens"] for item in requests)
    return {
        "batch_size": batch_size,
        "repetition": repetition,
        "request_payload_sha256": _json_sha256(payload),
        "client_wall_seconds": wall_seconds,
        "output_tokens": total_output_tokens,
        "output_throughput_tokens_per_second": total_output_tokens / wall_seconds,
        "gpu_memory": memory_observation,
        "server_before": before,
        "server_after": after,
        "requests": requests,
    }


def _read_log_tail(log: Any, *, limit: int = 16000) -> str:
    log.flush()
    log.seek(0)
    return log.read().decode("utf-8", errors="replace")[-limit:]


@contextmanager
def _termination_guard() -> Any:
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = {name: signal.getsignal(name) for name in (signal.SIGINT, signal.SIGTERM)}

    def interrupt(signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt(f"received signal {signum}")

    for name in previous:
        signal.signal(name, interrupt)
    try:
        yield
    finally:
        for name, handler in previous.items():
            signal.signal(name, handler)


def _run_server(
    args: argparse.Namespace,
    *,
    kind: str,
    max_draft_tokens: int | None,
    inputs: list[dict[str, Any]],
) -> dict[str, Any]:
    _assert_port_unused(args.port)
    command = _server_command(
        args, kind=kind, max_draft_tokens=max_draft_tokens
    )
    environment = _server_environment(kind=kind, gpu_index=args.gpu_index)
    base_url = f"http://127.0.0.1:{args.port}"
    with tempfile.TemporaryDirectory(prefix="nextlat-dspark-eval-") as directory:
        log_path = Path(directory) / "server.log"
        with log_path.open("w+b") as log:
            process = subprocess.Popen(
                command,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            with _termination_guard():
                try:
                    _wait_ready(base_url, process, args.startup_timeout)
                    identity = _assert_server_identity(
                        base_url,
                        args=args,
                        kind=kind,
                        max_draft_tokens=max_draft_tokens,
                        timeout=args.request_timeout,
                    )
                    warmups: list[dict[str, Any]] = []
                    cases: list[dict[str, Any]] = []
                    for batch_size in EVAL_BATCH_SIZES:
                        warmups.append(
                            _measure_case(
                                base_url=base_url,
                                inputs=inputs,
                                batch_size=batch_size,
                                repetition=-1,
                                max_new_tokens=args.max_new_tokens,
                                max_draft_tokens=max_draft_tokens,
                                gpu_index=args.gpu_index,
                                memory_sample_interval=args.memory_sample_interval,
                                request_timeout=args.request_timeout,
                            )
                        )
                        for repetition in range(args.repetitions):
                            cases.append(
                                _measure_case(
                                    base_url=base_url,
                                    inputs=inputs,
                                    batch_size=batch_size,
                                    repetition=repetition,
                                    max_new_tokens=args.max_new_tokens,
                                    max_draft_tokens=max_draft_tokens,
                                    gpu_index=args.gpu_index,
                                    memory_sample_interval=args.memory_sample_interval,
                                    request_timeout=args.request_timeout,
                                )
                            )
                except Exception as exc:
                    tail = _read_log_tail(log)
                    raise RuntimeError(
                        f"{kind} server evaluation failed: {exc}\n{tail}"
                    ) from exc
                finally:
                    _terminate_owned_process(process)
    return {
        "kind": kind,
        "max_draft_tokens": max_draft_tokens,
        "server_command": command,
        "server_command_sha256": _json_sha256(command),
        "server_identity": identity,
        "shape_warmups": warmups,
        "cases": cases,
        "acceptance_summary": (
            None
            if max_draft_tokens is None
            else _aggregate_acceptance(cases, max_draft_tokens=max_draft_tokens)
        ),
    }


def _case_map(server: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    result: dict[tuple[int, int], dict[str, Any]] = {}
    for case in server["cases"]:
        key = (case["batch_size"], case["repetition"])
        if key in result:
            raise RuntimeError(f"duplicate evaluation case: {key}")
        result[key] = case
    return result


def _assert_exact_parity(
    baseline: dict[str, Any], nextlat_servers: list[dict[str, Any]]
) -> dict[str, Any]:
    expected = _case_map(baseline)
    comparisons = 0
    for server in nextlat_servers:
        observed = _case_map(server)
        if set(observed) != set(expected):
            raise RuntimeError("NEXTLAT case roster differs from exact-greedy baseline")
        for key in sorted(expected):
            baseline_requests = expected[key]["requests"]
            nextlat_requests = observed[key]["requests"]
            if len(nextlat_requests) != len(baseline_requests):
                raise RuntimeError(f"request count differs for case {key}")
            for index, (left, right) in enumerate(
                zip(baseline_requests, nextlat_requests)
            ):
                comparisons += 1
                if left["output_ids"] != right["output_ids"]:
                    mismatch = next(
                        (
                            position
                            for position, (a, b) in enumerate(
                                zip(left["output_ids"], right["output_ids"])
                            )
                            if a != b
                        ),
                        min(len(left["output_ids"]), len(right["output_ids"])),
                    )
                    raise RuntimeError(
                        "exact token parity failed for "
                        f"M={server['max_draft_tokens']} case={key} request={index} "
                        f"at output position {mismatch}"
                    )
    return {
        "all_output_token_ids_exact": True,
        "comparison_count": comparisons,
        "baseline_server_kind": "exact_greedy",
        "candidate_server_kind": "nextlat_dspark_verify",
    }


def _read_registered_inputs(path: Path, *, vocab_size: int) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("registered input JSON must be one regular non-symlink file")

    def exact_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in registered inputs")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=exact_object
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read registered input JSON: {path}") from exc
    if not isinstance(value, dict) or set(value) != _REGISTERED_INPUT_KEYS:
        raise ValueError("registered input JSON has unknown or missing top-level keys")
    expected_header = {
        "schema_version": 1,
        "authority": _REGISTERED_INPUT_AUTHORITY,
        "prepared_manifest_sha256": _REGISTERED_PREPARED_MANIFEST_SHA256,
        "recursive_holdout_record_digest": _REGISTERED_HOLDOUT_RECORD_DIGEST,
        "selection": _REGISTERED_INPUT_SELECTION,
    }
    for name, expected in expected_header.items():
        if value.get(name) != expected:
            raise ValueError(f"registered input {name} differs from the frozen contract")
    inputs = value["inputs"]
    if not isinstance(inputs, list) or len(inputs) != max(EVAL_BATCH_SIZES):
        raise ValueError("registered input JSON must contain exactly eight inputs")
    problem_hashes: set[str] = set()
    for ordinal, item in enumerate(inputs):
        if not isinstance(item, dict) or set(item) != _REGISTERED_INPUT_ITEM_KEYS:
            raise ValueError("registered input item has unknown or missing keys")
        problem_hash = item["problem_hash"]
        token_ids = item["input_ids"]
        if item["ordinal"] != ordinal:
            raise ValueError("registered input ordinals are not the exact holdout order")
        if (
            not isinstance(problem_hash, str)
            or len(problem_hash) != 64
            or any(char not in _SHA256_CHARS for char in problem_hash)
            or problem_hash in problem_hashes
        ):
            raise ValueError("registered input has an invalid or duplicate problem hash")
        if (
            not isinstance(token_ids, list)
            or not token_ids
            or any(
                isinstance(token, bool)
                or not isinstance(token, int)
                or token < 0
                or token >= vocab_size
                for token in token_ids
            )
        ):
            raise ValueError("registered input_ids are outside the target vocabulary")
        problem_hashes.add(problem_hash)
    return value


def _write_receipt(path: Path, payload: dict[str, Any]) -> Path:
    body = dict(payload)
    body["receipt_sha256"] = _json_sha256(payload)
    encoded = canonical_json_bytes(body, pretty=True)
    destination_arg = path.expanduser()
    parent = destination_arg.parent.resolve()
    if not parent.is_dir() or not destination_arg.name:
        raise ValueError("receipt must have an existing parent and a file name")
    destination = parent / destination_arg.name
    if destination.exists():
        if (
            destination.is_file()
            and not destination.is_symlink()
            and destination.read_bytes() == encoded
        ):
            return destination
        raise FileExistsError(f"refusing to overwrite a different receipt: {destination}")
    temporary = parent / f".{destination.name}.{os.getpid()}.partial"
    if temporary.exists():
        raise FileExistsError(f"stale receipt partial must be inspected: {temporary}")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def run(args: argparse.Namespace) -> Path:
    _require_installed_plugin()
    if args.max_new_tokens < max(EVAL_MAX_DRAFT_TOKENS) + 2:
        raise ValueError("max-new-tokens must be at least 10 to exercise M=8")
    if args.repetitions < 1:
        raise ValueError("repetitions must be positive")
    if not 0.0 < args.mem_fraction_static < 1.0:
        raise ValueError("mem-fraction-static must be strictly between zero and one")
    if args.memory_sample_interval <= 0:
        raise ValueError("memory-sample-interval must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("DSpark-path evaluation has no CPU fallback")
    if args.gpu_index < 0 or args.gpu_index >= torch.cuda.device_count():
        raise ValueError(
            f"gpu-index {args.gpu_index} is outside {torch.cuda.device_count()} devices"
        )

    sidecar = read_sidecar_metadata(args.sidecar)
    registered_inputs_arg = args.registered_inputs.expanduser()
    registered_inputs_file_sha256 = _sha256_file(registered_inputs_arg)
    registered_inputs = _read_registered_inputs(
        registered_inputs_arg, vocab_size=sidecar.contract.vocab_size
    )
    if _sha256_file(registered_inputs_arg) != registered_inputs_file_sha256:
        raise RuntimeError("registered input file changed while it was admitted")
    registered_inputs_path = registered_inputs_arg.resolve()
    source = admit_sglang_source()
    target = admit_target_checkpoint(
        args.model,
        expected_manifest_sha256=sidecar.target_checkpoint_sha256,
        contract=sidecar.contract,
    )

    baseline = _run_server(
        args,
        kind="exact_greedy",
        max_draft_tokens=None,
        inputs=registered_inputs["inputs"],
    )
    nextlat_servers = [
        _run_server(
            args,
            kind="nextlat_dspark_verify",
            max_draft_tokens=max_draft_tokens,
            inputs=registered_inputs["inputs"],
        )
        for max_draft_tokens in EVAL_MAX_DRAFT_TOKENS
    ]
    parity = _assert_exact_parity(baseline, nextlat_servers)
    if _sha256_file(registered_inputs_path) != registered_inputs_file_sha256:
        raise RuntimeError("registered input file changed during evaluation")

    properties = torch.cuda.get_device_properties(args.gpu_index)
    payload = {
        "schema_version": EVAL_SCHEMA_VERSION,
        "result": "pass",
        "claim_scope": {
            "baseline": "same-target exact greedy SGLang server",
            "candidate": "NEXTLAT using SGLang DSpark ragged verify/control primitives",
            "model_vs_model_dspark_baseline": False,
            "matched_qwen3_dspark_drafter_checkpoint": False,
            "fallback": None,
        },
        "arguments": {
            "batch_sizes": list(EVAL_BATCH_SIZES),
            "max_draft_tokens": list(EVAL_MAX_DRAFT_TOKENS),
            "max_new_tokens": args.max_new_tokens,
            "repetitions": args.repetitions,
            "gpu_index": args.gpu_index,
            "mem_fraction_static": args.mem_fraction_static,
            "memory_sample_interval": args.memory_sample_interval,
        },
        "provenance": {
            "harness_sha256": _sha256_file(Path(__file__).resolve()),
            "registered_inputs_path": str(registered_inputs_path),
            "registered_inputs_file_sha256": registered_inputs_file_sha256,
            "registered_inputs_canonical_sha256": _json_sha256(registered_inputs),
            "registered_inputs": registered_inputs,
            "model_root": str(target.root),
            "target_manifest_sha256": target.manifest_sha256,
            "target_payloads": [
                {"path": name, "size": size, "sha256": digest}
                for name, size, digest in target.payloads
            ],
            "sidecar_root": str(sidecar.root),
            "sidecar_config_sha256": _sha256_file(sidecar.root / "config.json"),
            "dynamics_sha256": sidecar.dynamics_sha256,
            "sglang": {
                "expected_commit": AUDITED_SGLANG_COMMIT,
                "source_receipt_sha256": source.receipt_sha256,
                "audited_file_count": len(source.audited_files),
            },
            "runtime_packages": {
                "sglang": _package_version("sglang"),
                "sglang-nextlat": _package_version("sglang-nextlat"),
                "torch": torch.__version__,
            },
            "cuda_runtime": torch.version.cuda,
            "gpu": {
                "name": properties.name,
                "compute_capability": [properties.major, properties.minor],
                "total_memory_bytes": properties.total_memory,
            },
        },
        "parity": parity,
        "exact_greedy": baseline,
        "nextlat_dspark_verify": nextlat_servers,
    }
    return _write_receipt(args.receipt, payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--registered-inputs", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--port", type=int, default=31000)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--mem-fraction-static", type=float, default=0.5)
    parser.add_argument("--memory-sample-interval", type=float, default=0.1)
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    return parser


def main() -> None:
    receipt = run(build_parser().parse_args())
    print(receipt)


if __name__ == "__main__":
    main()
