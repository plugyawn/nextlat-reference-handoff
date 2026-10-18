"""One-server H100 phase probe for the production NEXTLAT decode path.

This is a diagnostic screen before the larger timed evaluator.  It runs only
NEXTLAT M=8 at B=1/2/4/8, with one full-width and one compact-ragged verify
geometry per batch size.  CUDA events are emitted by the real worker around the
real recurrent transition, tied vocabulary head, compact target verify, and
the residual in-worker scheduler/control path.  There is no synthetic model,
CPU timing fallback, or model-vs-model DSpark claim.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shlex
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .admission import admit_sglang_source, admit_target_checkpoint
from .checkpoint import read_sidecar_metadata
from .constants import AUDITED_SGLANG_COMMIT
from .dspark_eval import (
    _assert_port_unused,
    _assert_server_identity,
    _json_sha256,
    _package_version,
    _read_log_tail,
    _read_registered_inputs,
    _response_items,
    _server_command,
    _server_environment,
    _termination_guard,
    _write_receipt,
)
from .image_contract import plugin_manifest_sha256, verify_plugin_source_manifest
from .smoke_gpu import (
    _request_json,
    _require_installed_plugin,
    _terminate_owned_process,
    _wait_ready,
)


PROBE_BATCH_SIZES = (1, 2, 4, 8)
PROBE_MAX_DRAFT_TOKENS = 8
PROBE_REGIMES = ("fixed", "ragged")
_PRIMARY_PHASES = (
    "recurrent_dynamics_sum",
    "vocab_head_sum",
    "exact_target_verify",
    "scheduler_control_residual",
)
_SUMMARY_PHASES = (
    "decode_total",
    "combined_draft_loop",
    "embedding_sum",
    *_PRIMARY_PHASES,
    "pre_draft_control",
    "draft_loop_control",
    "verify_setup_control",
    "post_verify_control",
)
_CUDA_KEYS = frozenset(_SUMMARY_PHASES)
_TRACE_KEYS = frozenset(
    {
        "schema_version",
        "event",
        "forward_iter",
        "request_ids",
        "batch_size",
        "configured_max_drafts",
        "proposal_lens",
        "num_correct_drafts",
        "accept_lens",
        "total_verify_rows",
        "executed_recurrent_steps",
        "host_critical_path_ms",
        "cuda_ms",
        "per_step_cuda_ms",
        "selected_bonus_margin",
    }
)
_BONUS_MARGIN_KEYS = frozenset(
    {
        "margin_authority",
        "low_margin_logit_gap_threshold",
        "selected_target_top1_minus_top2_logit",
        "request_selected_bonus_low_margin",
        "low_margin_selected_bonus_count",
        "verify_request_count",
        "excluded_margin_diagnostic_cuda_ms",
    }
)
_LOW_MARGIN_LOGIT_GAP = 0.125
# Dockerfile.sm90 copies the protected source tree here before installing the
# wheel into SGLang's Python prefix.  __file__ therefore names the installed
# wheel, not the source root whose manifest the image admitted at build time.
_PACKAGE_ROOT = Path("/opt/sglang-nextlat")


def _finite(value: Any, *, label: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        suffix = "" if minimum is None else f" >= {minimum}"
        raise RuntimeError(f"{label} must be finite{suffix}")
    return result


def _ragged_proposal_lens(batch_size: int) -> list[int]:
    if batch_size not in PROBE_BATCH_SIZES:
        raise ValueError(f"batch_size must be one of {PROBE_BATCH_SIZES}")
    if batch_size == 1:
        return [PROBE_MAX_DRAFT_TOKENS // 2]
    return [
        round(PROBE_MAX_DRAFT_TOKENS * (batch_size - 1 - index) / (batch_size - 1))
        for index in range(batch_size)
    ]


def _proposal_lens(batch_size: int, regime: str) -> list[int]:
    if regime == "fixed":
        return [PROBE_MAX_DRAFT_TOKENS] * batch_size
    if regime == "ragged":
        return _ragged_proposal_lens(batch_size)
    raise ValueError(f"unknown probe regime: {regime}")


def _generation_payload(
    input_ids: list[list[int]], *, proposal_lens: list[int], request_prefix: str
) -> dict[str, Any]:
    if len(input_ids) != len(proposal_lens):
        raise ValueError("input_ids and proposal_lens differ in batch size")
    sampling_params = []
    for proposal_len in proposal_lens:
        # Prefill emits one token.  The first decode then sees
        # remaining_outputs-1 == proposal_len recurrent slots.
        sampling_params.append(
            {
                "max_new_tokens": proposal_len + 2,
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
        )
    return {
        "rid": [f"{request_prefix}-{index}" for index in range(len(input_ids))],
        "input_ids": input_ids,
        "sampling_params": sampling_params,
        "return_logprob": False,
        "stream": False,
        "log_metrics": True,
    }


def _read_trace(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("NEXTLAT worker did not publish a regular phase trace")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"phase trace line {line_number} is invalid JSON") from exc
        if not isinstance(record, dict):
            raise RuntimeError(f"phase trace line {line_number} is not an object")
        records.append(record)
    if not records or records[0].get("event") != "header":
        raise RuntimeError("phase trace has no worker header")
    return records


def _validate_trace_sample(
    record: dict[str, Any], *, request_ids: list[str], proposal_lens: list[int]
) -> dict[str, Any]:
    if set(record) != _TRACE_KEYS:
        raise RuntimeError("decode phase trace fields differ from schema v1")
    expected = {
        "schema_version": 1,
        "event": "decode_phase_timing",
        "request_ids": request_ids,
        "batch_size": len(request_ids),
        "configured_max_drafts": PROBE_MAX_DRAFT_TOKENS,
        "proposal_lens": proposal_lens,
        "total_verify_rows": sum(proposal_lens) + len(proposal_lens),
        "executed_recurrent_steps": PROBE_MAX_DRAFT_TOKENS,
    }
    for name, value in expected.items():
        if record.get(name) != value:
            raise RuntimeError(f"phase trace {name} differs from requested geometry")
    num_correct = record.get("num_correct_drafts")
    accept_lens = record.get("accept_lens")
    if (
        not isinstance(num_correct, list)
        or len(num_correct) != len(request_ids)
        or not isinstance(accept_lens, list)
        or len(accept_lens) != len(request_ids)
    ):
        raise RuntimeError("phase trace acceptance vectors differ from batch")
    for index, (correct, emitted, proposed) in enumerate(
        zip(num_correct, accept_lens, proposal_lens)
    ):
        if (
            isinstance(correct, bool)
            or not isinstance(correct, int)
            or correct < 0
            or correct > proposed
            or isinstance(emitted, bool)
            or not isinstance(emitted, int)
            or emitted != correct + 1
        ):
            raise RuntimeError(f"phase trace acceptance algebra differs at row {index}")
    if isinstance(record.get("forward_iter"), bool) or not isinstance(
        record.get("forward_iter"), int
    ):
        raise RuntimeError("phase trace forward_iter must be an integer")
    _finite(record.get("host_critical_path_ms"), label="host critical path", minimum=0.0)
    cuda_ms = record.get("cuda_ms")
    if not isinstance(cuda_ms, dict) or set(cuda_ms) != _CUDA_KEYS:
        raise RuntimeError("phase trace CUDA phase roster differs")
    normalized: dict[str, float] = {}
    for name in _SUMMARY_PHASES:
        minimum = -0.05 if name.endswith("control") or name.endswith("residual") else 0.0
        normalized[name] = _finite(
            cuda_ms.get(name), label=f"CUDA phase {name}", minimum=minimum
        )
    if normalized["decode_total"] <= 0 or normalized["combined_draft_loop"] <= 0:
        raise RuntimeError("decode and draft-loop timing must be positive")

    component_sum = sum(normalized[name] for name in _PRIMARY_PHASES)
    tolerance = max(0.02, normalized["decode_total"] * 0.002)
    if abs(component_sum - normalized["decode_total"]) > tolerance:
        raise RuntimeError("Amdahl phase accounting does not close to decode_total")
    control_subphases = sum(
        normalized[name]
        for name in (
            "pre_draft_control",
            "draft_loop_control",
            "verify_setup_control",
            "post_verify_control",
        )
    )
    if abs(control_subphases - normalized["scheduler_control_residual"]) > tolerance:
        raise RuntimeError("scheduler/control subphases do not close to their residual")

    step_rows = record.get("per_step_cuda_ms")
    if not isinstance(step_rows, list) or len(step_rows) != PROBE_MAX_DRAFT_TOKENS:
        raise RuntimeError("phase trace must contain exactly eight recurrent step rows")
    for step, row in enumerate(step_rows):
        if not isinstance(row, dict) or set(row) != {
            "step",
            "embedding_ms",
            "recurrent_dynamics_ms",
            "vocab_head_ms",
        }:
            raise RuntimeError("recurrent step timing fields differ")
        if row["step"] != step:
            raise RuntimeError("recurrent step timing order differs")
        for name in ("embedding_ms", "recurrent_dynamics_ms", "vocab_head_ms"):
            _finite(row[name], label=f"step {step} {name}", minimum=0.0)

    bonus_margin = record.get("selected_bonus_margin")
    if not isinstance(bonus_margin, dict) or set(bonus_margin) != _BONUS_MARGIN_KEYS:
        raise RuntimeError("selected-bonus margin fields differ")
    if bonus_margin["margin_authority"] != "exact target selected bonus row":
        raise RuntimeError("selected-bonus margin authority differs")
    if bonus_margin["low_margin_logit_gap_threshold"] != _LOW_MARGIN_LOGIT_GAP:
        raise RuntimeError("selected-bonus low-margin threshold differs")
    margins = bonus_margin["selected_target_top1_minus_top2_logit"]
    low_margin = bonus_margin["request_selected_bonus_low_margin"]
    if (
        not isinstance(margins, list)
        or len(margins) != len(request_ids)
        or not isinstance(low_margin, list)
        or len(low_margin) != len(request_ids)
        or any(type(value) is not bool for value in low_margin)
    ):
        raise RuntimeError("selected-bonus margin vectors differ from batch")
    for index, margin in enumerate(margins):
        _finite(margin, label=f"selected bonus margin {index}", minimum=0.0)
        if low_margin[index] != (float(margin) <= _LOW_MARGIN_LOGIT_GAP):
            raise RuntimeError("selected-bonus low-margin classification is inconsistent")
    low_margin_count = sum(low_margin)
    if (
        bonus_margin["low_margin_selected_bonus_count"] != low_margin_count
        or bonus_margin["verify_request_count"] != len(request_ids)
    ):
        raise RuntimeError("selected-bonus margin counts differ from request vectors")
    _finite(
        bonus_margin["excluded_margin_diagnostic_cuda_ms"],
        label="excluded margin diagnostic",
        minimum=0.0,
    )
    return record


def _select_trace_sample(
    trace_path: Path, *, request_ids: list[str], proposal_lens: list[int]
) -> dict[str, Any]:
    matches = [
        record
        for record in _read_trace(trace_path)
        if record.get("event") == "decode_phase_timing"
        and record.get("request_ids") == request_ids
        and record.get("proposal_lens") == proposal_lens
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "expected exactly one first-decode phase row for the requested geometry, "
            f"observed {len(matches)}"
        )
    return _validate_trace_sample(
        matches[0], request_ids=request_ids, proposal_lens=proposal_lens
    )


def _measure_case(
    *,
    base_url: str,
    inputs: list[dict[str, Any]],
    batch_size: int,
    regime: str,
    label: str,
    trace_path: Path,
    request_timeout: float,
) -> dict[str, Any]:
    proposal_lens = _proposal_lens(batch_size, regime)
    payload = _generation_payload(
        [item["input_ids"] for item in inputs[:batch_size]],
        proposal_lens=proposal_lens,
        request_prefix=label,
    )
    started = time.monotonic()
    response = _request_json(
        f"{base_url}/generate", payload, timeout=request_timeout
    )
    client_wall_ms = (time.monotonic() - started) * 1000.0
    items = _response_items(response, expected_rids=payload["rid"])
    for item, expected_len in zip(items, proposal_lens):
        output_ids = item.get("output_ids")
        meta = item.get("meta_info")
        expected_completion = expected_len + 2
        if not isinstance(output_ids, list) or len(output_ids) != expected_completion:
            raise RuntimeError("probe response output length differs from max_new_tokens")
        if not isinstance(meta, dict) or meta.get("completion_tokens") != expected_completion:
            raise RuntimeError("probe response completion-token metric differs")
    record = _select_trace_sample(
        trace_path, request_ids=payload["rid"], proposal_lens=proposal_lens
    )
    return {
        "client_wall_ms": client_wall_ms,
        "output_ids_sha256": _json_sha256([item["output_ids"] for item in items]),
        "trace": record,
    }


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty timing list")
    return {
        "minimum": min(values),
        "median": statistics.median(values),
        "p90": _percentile(values, 0.90),
        "maximum": max(values),
    }


def _amdahl(samples: list[dict[str, Any]]) -> dict[str, Any]:
    fractions: dict[str, list[float]] = {name: [] for name in _PRIMARY_PHASES}
    two_x: dict[str, list[float]] = {name: [] for name in _PRIMARY_PHASES}
    ceilings: dict[str, list[float]] = {
        **{name: [] for name in _PRIMARY_PHASES},
        "combined_draft_loop": [],
    }
    for sample in samples:
        cuda_ms = sample["trace"]["cuda_ms"]
        total = float(cuda_ms["decode_total"])
        for name in _PRIMARY_PHASES:
            phase = float(cuda_ms[name])
            fractions[name].append(phase / total)
            two_x[name].append(total / (total - phase / 2.0))
            ceilings[name].append(total / (total - phase))
        draft = float(cuda_ms["combined_draft_loop"])
        ceilings["combined_draft_loop"].append(total / (total - draft))
    low_margin_count = sum(
        int(
            sample["trace"]["selected_bonus_margin"][
                "low_margin_selected_bonus_count"
            ]
        )
        for sample in samples
    )
    verify_request_count = sum(
        int(sample["trace"]["selected_bonus_margin"]["verify_request_count"])
        for sample in samples
    )
    target_verify_ms = sum(
        float(sample["trace"]["cuda_ms"]["exact_target_verify"])
        for sample in samples
    )
    verify_round_count = sum(len(sample["trace"]["accept_lens"]) for sample in samples)
    emitted_token_count = sum(
        sum(int(value) for value in sample["trace"]["accept_lens"])
        for sample in samples
    )
    accepted_draft_count = sum(
        sum(int(value) for value in sample["trace"]["num_correct_drafts"])
        for sample in samples
    )
    proposed_draft_count = sum(
        sum(int(value) for value in sample["trace"]["proposal_lens"])
        for sample in samples
    )
    return {
        "formula": "speedup(r)=1/((1-f)+f/r); ceiling=1/(1-f)",
        "median_fraction_of_decode": {
            name: statistics.median(values) for name, values in fractions.items()
        },
        "median_speedup_if_phase_is_2x_faster": {
            name: statistics.median(values) for name, values in two_x.items()
        },
        "median_speedup_ceiling_if_phase_is_eliminated": {
            name: statistics.median(values) for name, values in ceilings.items()
        },
        "yield": {
            "verify_request_count": verify_round_count,
            "accepted_draft_count": accepted_draft_count,
            "proposed_draft_count": proposed_draft_count,
            "accepted_draft_rate": (
                None
                if proposed_draft_count == 0
                else accepted_draft_count / proposed_draft_count
            ),
            "emitted_token_count": emitted_token_count,
            "mean_emitted_tokens_per_verify": (
                emitted_token_count / verify_round_count
            ),
            "cuda_ms_amortized_per_emitted_token": {
                name: sum(
                    float(sample["trace"]["cuda_ms"][name]) for sample in samples
                )
                / emitted_token_count
                for name in (
                    "decode_total",
                    "combined_draft_loop",
                    "recurrent_dynamics_sum",
                    "vocab_head_sum",
                    "exact_target_verify",
                    "scheduler_control_residual",
                )
            },
        },
        "selected_bonus_low_margin": {
            "low_margin_logit_gap_threshold": _LOW_MARGIN_LOGIT_GAP,
            "margin_authority": "exact target selected bonus row",
            "low_margin_selected_bonus_count": low_margin_count,
            "verify_request_count": verify_request_count,
            "frequency_per_verify_request": low_margin_count / verify_request_count,
            "observed_exact_target_verify_ms_amortized_per_verify_request": (
                target_verify_ms / verify_request_count
            ),
            "excluded_margin_diagnostic_ms": _summary(
                [
                    float(
                        sample["trace"]["selected_bonus_margin"][
                            "excluded_margin_diagnostic_cuda_ms"
                        ]
                    )
                    for sample in samples
                ]
            ),
        },
        "hypothetical_canonical_correction": {
            "status": "not_implemented",
            "trigger": "selected exact-target bonus gap <= 0.125",
            "measured_per_trigger_cost_ms": None,
            "amortized_incremental_ms_per_verify_request": None,
            "formula": (
                "amortized_incremental_ms_per_verify_request = "
                "selected_bonus_low_margin_frequency * canonical_correction_cost_ms"
            ),
            "amdahl_note": (
                "once measured, add the amortized correction term to decode_total before "
                "recomputing all phase fractions; ordinary draft rejection is not a "
                "canonical correction, and target-verify time is not substituted for the "
                "unimplemented correction path"
            ),
        },
    }


def _summarize_case(
    *, batch_size: int, regime: str, samples: list[dict[str, Any]]
) -> dict[str, Any]:
    phase_summary = {
        name: _summary([float(sample["trace"]["cuda_ms"][name]) for sample in samples])
        for name in _SUMMARY_PHASES
    }
    return {
        "batch_size": batch_size,
        "length_regime": regime,
        "proposal_lens": _proposal_lens(batch_size, regime),
        "total_verify_rows": sum(_proposal_lens(batch_size, regime)) + batch_size,
        "sample_count": len(samples),
        "phase_summary_ms": phase_summary,
        "amdahl": _amdahl(samples),
        "samples": samples,
    }


def _competitiveness(results: list[dict[str, Any]]) -> dict[str, Any]:
    ragged_vs_fixed = []
    by_key = {(row["batch_size"], row["length_regime"]): row for row in results}
    for batch_size in PROBE_BATCH_SIZES:
        fixed = by_key[(batch_size, "fixed")]
        ragged = by_key[(batch_size, "ragged")]
        fixed_ms = fixed["phase_summary_ms"]["combined_draft_loop"]["median"]
        ragged_ms = ragged["phase_summary_ms"]["combined_draft_loop"]["median"]
        ragged_vs_fixed.append(
            {
                "batch_size": batch_size,
                "fixed_over_ragged_draft_loop_speedup": fixed_ms / ragged_ms,
                "note": (
                    "production currently executes configured M=8 recurrence for both "
                    "geometries; this ratio measures current behavior, not a tail-skip claim"
                ),
            }
        )
    return {
        "ragged_vs_fixed": ragged_vs_fixed,
        "matched_dspark_draft_model": {
            "status": "unavailable",
            "checkpoint": None,
            "timing_receipt": None,
            "reason": (
                "the repository admits DSpark ragged verify/control primitives but no "
                "matched Qwen3-1.7B DSpark drafter/confidence checkpoint"
            ),
        },
        "gates": [
            {
                "id": "phase_decomposition",
                "status": "pass",
                "threshold": (
                    "all B=1/2/4/8 fixed and ragged samples execute the production "
                    "worker and close CUDA-event Amdahl accounting"
                ),
            },
            {
                "id": "nextlat_vs_matched_dspark_latency_and_yield",
                "status": "unavailable",
                "threshold": (
                    "on the same H100/image/target/inputs and identical ordered proposal "
                    "lengths, NEXTLAT combined-draft p50 must not exceed matched DSpark "
                    "drafter p50, and decode CUDA ms per emitted token must be no worse; "
                    "no B/regime may be more than 5% slower"
                ),
            },
            {
                "id": "full_timed_eval_admission",
                "status": "not_admitted",
                "threshold": (
                    "phase decomposition passes and the matched DSpark draft-latency gate "
                    "is available and passes before a model-vs-model timed evaluation"
                ),
            },
        ],
    }


def _run_server_probe(
    args: argparse.Namespace, *, inputs: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], str]:
    _assert_port_unused(args.port)
    command = _server_command(
        args, kind="nextlat_dspark_verify", max_draft_tokens=PROBE_MAX_DRAFT_TOKENS
    )
    base_url = f"http://127.0.0.1:{args.port}"
    with tempfile.TemporaryDirectory(prefix="nextlat-phase-probe-") as directory:
        root = Path(directory)
        trace_path = root / "phase-trace.jsonl"
        log_path = root / "server.log"
        environment = _server_environment(
            kind="nextlat_dspark_verify", gpu_index=args.gpu_index
        )
        environment.pop("NEXTLAT_DIAGNOSTIC_TRACE", None)
        environment.pop("NEXTLAT_DIAGNOSTIC_RID", None)
        environment["NEXTLAT_PHASE_TRACE"] = str(trace_path)
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
                        kind="nextlat_dspark_verify",
                        max_draft_tokens=PROBE_MAX_DRAFT_TOKENS,
                        timeout=args.request_timeout,
                    )
                    warmups: list[dict[str, Any]] = []
                    timed: dict[tuple[int, str], list[dict[str, Any]]] = {
                        (batch_size, regime): []
                        for batch_size in PROBE_BATCH_SIZES
                        for regime in PROBE_REGIMES
                    }
                    for batch_size in PROBE_BATCH_SIZES:
                        for warmup in range(args.warmups):
                            for regime in PROBE_REGIMES:
                                warmups.append(
                                    _measure_case(
                                        base_url=base_url,
                                        inputs=inputs,
                                        batch_size=batch_size,
                                        regime=regime,
                                        label=f"phase-warmup-{warmup}-{regime}-b{batch_size}",
                                        trace_path=trace_path,
                                        request_timeout=args.request_timeout,
                                    )
                                )
                        for sample in range(args.samples):
                            for regime in PROBE_REGIMES:
                                timed[(batch_size, regime)].append(
                                    _measure_case(
                                        base_url=base_url,
                                        inputs=inputs,
                                        batch_size=batch_size,
                                        regime=regime,
                                        label=f"phase-sample-{sample}-{regime}-b{batch_size}",
                                        trace_path=trace_path,
                                        request_timeout=args.request_timeout,
                                    )
                                )
                    results = [
                        _summarize_case(
                            batch_size=batch_size,
                            regime=regime,
                            samples=timed[(batch_size, regime)],
                        )
                        for batch_size in PROBE_BATCH_SIZES
                        for regime in PROBE_REGIMES
                    ]
                    trace_sha256 = _json_sha256(_read_trace(trace_path))
                except Exception as exc:
                    tail = _read_log_tail(log)
                    raise RuntimeError(f"NEXTLAT phase probe failed: {exc}\n{tail}") from exc
                finally:
                    _terminate_owned_process(process)
    return identity, warmups, results, trace_sha256


def _validate_receipt(payload: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "result_state",
        "created_at_utc",
        "completed_at_utc",
        "claim_scope",
        "arguments",
        "provenance",
        "trace_contract",
        "server_identity",
        "warmup_count",
        "results",
        "competitiveness",
    }
    if set(payload) != required or payload.get("schema_version") != 1:
        raise RuntimeError("phase-probe receipt fields differ from schema v1")
    if payload.get("result_state") != "diagnostic":
        raise RuntimeError("phase probe can publish only a diagnostic receipt")
    results = payload.get("results")
    expected = {
        (batch_size, regime)
        for batch_size in PROBE_BATCH_SIZES
        for regime in PROBE_REGIMES
    }
    if not isinstance(results, list) or {
        (row.get("batch_size"), row.get("length_regime"))
        for row in results
        if isinstance(row, dict)
    } != expected:
        raise RuntimeError("phase-probe result geometry is incomplete")
    matched = payload.get("competitiveness", {}).get("matched_dspark_draft_model")
    if not isinstance(matched, dict) or matched.get("status") != "unavailable":
        raise RuntimeError("receipt must not invent a matched DSpark drafter result")


def run(args: argparse.Namespace) -> Path:
    _require_installed_plugin()
    if args.warmups < 1 or args.warmups > 3:
        raise ValueError("warmups must be in [1, 3] for this bounded diagnostic")
    if args.samples < 2 or args.samples > 10:
        raise ValueError("samples must be in [2, 10] for this bounded diagnostic")
    if not 0.0 < args.mem_fraction_static < 1.0:
        raise ValueError("mem-fraction-static must be strictly between zero and one")
    if not torch.cuda.is_available():
        raise RuntimeError("NEXTLAT phase probe has no CPU timing fallback")
    if args.gpu_index < 0 or args.gpu_index >= torch.cuda.device_count():
        raise ValueError("gpu-index is outside the visible CUDA device range")
    properties = torch.cuda.get_device_properties(args.gpu_index)
    if (properties.major, properties.minor) != (9, 0) or "H100" not in properties.name:
        raise RuntimeError("NEXTLAT phase probe is registered only for an SM90 H100")

    created_at = datetime.now(timezone.utc).isoformat()
    sidecar = read_sidecar_metadata(args.sidecar)
    registered = _read_registered_inputs(
        args.registered_inputs, vocab_size=sidecar.contract.vocab_size
    )
    source = admit_sglang_source()
    target = admit_target_checkpoint(
        args.model,
        expected_manifest_sha256=sidecar.target_checkpoint_sha256,
        contract=sidecar.contract,
    )
    manifest_sha256 = plugin_manifest_sha256(_PACKAGE_ROOT)
    manifest = verify_plugin_source_manifest(
        _PACKAGE_ROOT, expected_manifest_sha256=manifest_sha256
    )
    identity, warmups, results, trace_sha256 = _run_server_probe(
        args, inputs=registered["inputs"]
    )
    payload = {
        "schema_version": 1,
        "result_state": "diagnostic",
        "created_at_utc": created_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim_scope": {
            "candidate": "NEXTLAT production M=8 worker with DSpark ragged verify/control",
            "matched_dspark_draft_model_comparison": False,
            "synthetic_timing": False,
            "full_timed_evaluation": False,
        },
        "arguments": {
            "batch_sizes": list(PROBE_BATCH_SIZES),
            "max_draft_tokens": PROBE_MAX_DRAFT_TOKENS,
            "length_regimes": list(PROBE_REGIMES),
            "warmups": args.warmups,
            "samples": args.samples,
            "gpu_index": args.gpu_index,
            "port": args.port,
            "mem_fraction_static": args.mem_fraction_static,
        },
        "provenance": {
            "command": shlex.join(sys.argv),
            "cwd": str(Path.cwd()),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "runtime_packages": {
                "sglang": _package_version("sglang"),
                "sglang-nextlat": _package_version("sglang-nextlat"),
            },
            "gpu": {
                "name": properties.name,
                "compute_capability": [properties.major, properties.minor],
                "total_memory_bytes": properties.total_memory,
            },
            "expected_sglang_commit": AUDITED_SGLANG_COMMIT,
            "sglang_source_receipt_sha256": source.receipt_sha256,
            "plugin_source_manifest": manifest,
            "target_root": str(target.root),
            "target_manifest_sha256": target.manifest_sha256,
            "sidecar_root": str(sidecar.root),
            "dynamics_sha256": sidecar.dynamics_sha256,
            "registered_inputs_canonical_sha256": _json_sha256(registered),
            "raw_phase_trace_canonical_sha256": trace_sha256,
        },
        "trace_contract": {
            "authority": "torch.cuda.Event on the production worker stream",
            "target_verify": (
                "target_worker.forward_batch_generation(is_verify=True) through exact "
                "target-logit argmax"
            ),
            "scheduler_control_residual": (
                "decode_total - recurrent_dynamics - vocab_head - exact_target_verify; "
                "includes embeddings, ragged-window setup, acceptance, KV commit, and "
                "in-worker host launch gaps; excludes HTTP, queueing before worker entry, "
                "and response serialization"
            ),
            "combined_draft_loop": (
                "production propose_greedy including embedding, recurrent dynamics, tied "
                "vocabulary projection, argmax, and loop/control"
            ),
            "selected_bonus_and_hypothetical_correction": (
                "every active verify request is low margin when its selected exact-target "
                "bonus row has top1-top2 logit gap <=0.125, independent of draft "
                "acceptance. The post-decode top2 diagnostic is excluded from every "
                "production phase. Canonical correction is not implemented: its measured "
                "cost and amortized overhead remain null, with only p_low*C_ms recorded "
                "as the future Amdahl formula"
            ),
        },
        "server_identity": identity,
        "warmup_count": len(warmups),
        "results": results,
        "competitiveness": _competitiveness(results),
    }
    _validate_receipt(payload)
    return _write_receipt(args.receipt, payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--registered-inputs", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--port", type=int, default=31000)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--mem-fraction-static", type=float, default=0.5)
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    return parser


def main() -> None:
    receipt = run(build_parser().parse_args())
    print(receipt)


if __name__ == "__main__":
    main()
