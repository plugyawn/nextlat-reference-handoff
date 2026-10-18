"""One-context H100 replay separating historical KV drift from ragged verify math.

The registered parity failure occurs after a NEXTLAT M=2 target window consumes
three known tokens.  This probe rebuilds the prefix in a fresh request, then
compares:

* ordinary target decoding of those three tokens one at a time; and
* one NEXTLAT ragged target-verify window over the identical tokens.

It is diagnostic-only.  It does not correct output tokens, alter target KV, or
report performance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

from .admission import admit_sglang_source, admit_target_checkpoint
from .checkpoint import read_sidecar_metadata
from .constants import (
    AUDITED_SGLANG_COMMIT,
    QWEN3_1_7B_CONTRACT,
    SERVING_ATTENTION_BACKENDS,
)
from .dspark_eval import (
    _assert_port_unused,
    _assert_server_identity,
    _json_sha256,
    _parse_request_result,
    _read_log_tail,
    _server_command,
    _server_environment,
    _sha256_file,
    _termination_guard,
    _write_receipt,
)
from .parity_diagnostic import (
    MAX_DRAFT_TOKENS,
    _exact_meta_item,
    _first_mismatch,
    _logprob_position,
    _read_trace,
    _single_payload,
    _trace_output,
)
from .smoke_gpu import (
    _request_json,
    _require_installed_plugin,
    _terminate_owned_process,
    _wait_ready,
)


SCHEMA_VERSION = 1
COMPLETION_TOKENS = MAX_DRAFT_TOKENS + 2
REQUEST_ID_NEXTLAT = "nextlat-fresh-prefix-m2-state-replay"
REQUEST_ID_EXACT = "exact-fresh-prefix-sequential-state-replay"
ATTENTION_BACKENDS = tuple(
    backend
    for backend in ("flashinfer", "fa3")
    if backend in SERVING_ATTENTION_BACKENDS
)


def _with_attention_backend(command: list[str], backend: str) -> list[str]:
    if backend not in ATTENTION_BACKENDS:
        raise ValueError(f"unsupported attention backend: {backend!r}")
    positions = [
        index for index, value in enumerate(command) if value == "--attention-backend"
    ]
    if len(positions) != 1:
        raise ValueError(
            "server command must contain exactly one --attention-backend flag"
        )
    position = positions[0]
    if position + 1 >= len(command) or command[position + 1].startswith("--"):
        raise ValueError("server command attention backend has no value")
    result = list(command)
    result[position + 1] = backend
    if result.count("--attention-backend") != 1:
        raise RuntimeError("attention backend substitution introduced a duplicate flag")
    return result


def _cache_snapshot(root: Path) -> dict[str, Any]:
    if not root.is_dir() or root.is_symlink():
        raise ValueError("cache root must be an existing non-symlink directory")
    entries: list[dict[str, Any]] = []
    total_regular_bytes = 0
    for path in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            entries.append(
                {"path": relative, "type": "symlink", "target": os.readlink(path)}
            )
        elif path.is_file():
            size = path.stat().st_size
            total_regular_bytes += size
            entries.append(
                {
                    "path": relative,
                    "type": "regular",
                    "size": size,
                    "sha256": _sha256_file(path),
                }
            )
    return {
        "root": str(root),
        "entry_count": len(entries),
        "regular_file_count": sum(
            entry["type"] == "regular" for entry in entries
        ),
        "total_regular_bytes": total_regular_bytes,
        "tree_sha256": _json_sha256(entries),
        "entries": entries,
    }


def _cache_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_by_path = {entry["path"]: entry for entry in before["entries"]}
    after_by_path = {entry["path"]: entry for entry in after["entries"]}
    added = sorted(set(after_by_path) - set(before_by_path))
    removed = sorted(set(before_by_path) - set(after_by_path))
    modified = sorted(
        path
        for path in set(before_by_path) & set(after_by_path)
        if before_by_path[path] != after_by_path[path]
    )
    changed = bool(added or removed or modified)
    return {
        "content_changed": changed,
        "behavior": "cache_content_mutated" if changed else "cache_reused_unchanged",
        "added_paths": added,
        "removed_paths": removed,
        "modified_paths": modified,
    }


def _cache_log_markers(log_text: str) -> list[str]:
    needles = ("cache", "compile", "fa3", "flash attention", "flashattention", "jit")
    markers: list[str] = []
    for line in log_text.splitlines():
        if any(needle in line.lower() for needle in needles):
            markers.append(line[:1000])
        if len(markers) == 100:
            break
    return markers


def _read_receipt(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file() or path.is_symlink():
        raise ValueError("tie receipt must be a regular file")
    file_sha256 = _sha256_file(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("tie receipt is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("tie receipt root must be an object")
    claimed = value.get("receipt_sha256")
    unhashed = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if not isinstance(claimed, str) or claimed != _json_sha256(unhashed):
        raise ValueError("tie receipt self-hash is invalid")
    return value, file_sha256


def derive_fresh_replay_case(tie_receipt: dict[str, Any]) -> dict[str, Any]:
    """Derive the exact prefix/window from a witnessed B=1/M=2 mismatch."""

    if tie_receipt.get("result") != "mismatch_observed":
        raise ValueError("tie receipt did not witness a mismatch")
    mismatch = tie_receipt.get("first_mismatch_position")
    if isinstance(mismatch, bool) or not isinstance(mismatch, int) or mismatch < 3:
        raise ValueError("tie receipt mismatch position is invalid")

    provenance = tie_receipt.get("provenance")
    exact = tie_receipt.get("exact")
    nextlat = tie_receipt.get("nextlat")
    if not all(isinstance(value, dict) for value in (provenance, exact, nextlat)):
        raise ValueError("tie receipt lacks exact provenance/results")
    record = provenance.get("registered_input")
    parsed = exact.get("parsed")
    mismatch_record = nextlat.get("mismatch_trace_record")
    if not all(isinstance(value, dict) for value in (record, parsed, mismatch_record)):
        raise ValueError("tie receipt lacks the registered mismatch record")

    prompt_ids = record.get("input_ids")
    exact_output_ids = parsed.get("output_ids")
    prefix_seq_lens = mismatch_record.get("prefix_seq_lens")
    compact_tokens = mismatch_record.get("compact_verify_tokens")
    local_index = mismatch_record.get("mismatch_local_output_index")
    if (
        not isinstance(prompt_ids, list)
        or any(isinstance(token, bool) or not isinstance(token, int) for token in prompt_ids)
        or not isinstance(exact_output_ids, list)
        or any(
            isinstance(token, bool) or not isinstance(token, int)
            for token in exact_output_ids
        )
        or not isinstance(prefix_seq_lens, list)
        or len(prefix_seq_lens) != 1
        or isinstance(prefix_seq_lens[0], bool)
        or not isinstance(prefix_seq_lens[0], int)
        or not isinstance(compact_tokens, list)
        or len(compact_tokens) != MAX_DRAFT_TOKENS + 1
        or any(isinstance(token, bool) or not isinstance(token, int) for token in compact_tokens)
        or local_index != MAX_DRAFT_TOKENS
    ):
        raise ValueError("tie receipt mismatch geometry is not the registered B1/M2 case")

    output_prefix_count = prefix_seq_lens[0] - len(prompt_ids)
    if output_prefix_count < 0 or mismatch != output_prefix_count + MAX_DRAFT_TOKENS + 1:
        raise ValueError("tie receipt prefix length does not locate the mismatch")
    expected_window = exact_output_ids[
        output_prefix_count : output_prefix_count + MAX_DRAFT_TOKENS + 1
    ]
    if compact_tokens != expected_window:
        raise ValueError("ragged compact tokens differ from the exact trajectory")
    expected_completion = exact_output_ids[
        output_prefix_count : output_prefix_count + COMPLETION_TOKENS
    ]
    if len(expected_completion) != COMPLETION_TOKENS:
        raise ValueError("tie receipt is too short for the fresh replay")

    fresh_prefix = prompt_ids + exact_output_ids[:output_prefix_count]
    if len(fresh_prefix) != prefix_seq_lens[0]:
        raise ValueError("derived fresh prefix length is inconsistent")
    return {
        "mismatch_position": mismatch,
        "registered_input": record,
        "fresh_prefix_input_ids": fresh_prefix,
        "fresh_prefix_output_count": output_prefix_count,
        "expected_completion_ids": expected_completion,
        "expected_ragged_window_ids": expected_window,
    }


def _serve_once(
    args: argparse.Namespace,
    *,
    kind: str,
    max_draft_tokens: int | None,
    payload: dict[str, Any],
    request_id: str,
    trace_path: Path | None = None,
) -> dict[str, Any]:
    _assert_port_unused(args.port)
    command = _with_attention_backend(
        _server_command(args, kind=kind, max_draft_tokens=max_draft_tokens),
        args.attention_backend,
    )
    if kind == "exact_greedy":
        command.extend(["--return-hidden-states-mode", "last"])
    environment = _server_environment(kind=kind, gpu_index=args.gpu_index)
    for name in (
        "NEXTLAT_DIAGNOSTIC_TRACE",
        "NEXTLAT_DIAGNOSTIC_RID",
        "NEXTLAT_DIAGNOSTIC_RIDS",
        "NEXTLAT_PHASE_TRACE",
    ):
        environment.pop(name, None)
    if trace_path is not None:
        if kind != "nextlat_dspark_verify":
            raise ValueError("only the NEXTLAT replay may write a diagnostic trace")
        environment["NEXTLAT_DIAGNOSTIC_TRACE"] = str(trace_path)
        environment["NEXTLAT_DIAGNOSTIC_RIDS"] = json.dumps(
            [request_id], separators=(",", ":")
        )

    base_url = f"http://127.0.0.1:{args.port}"
    process_wall_started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix=f"{kind}-state-replay-server-") as directory:
        log_path = Path(directory) / "server.log"
        with log_path.open("w+b") as log, _termination_guard():
            process = subprocess.Popen(
                command,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                _wait_ready(base_url, process, args.startup_timeout)
                ready_wall = time.perf_counter()
                identity = _assert_server_identity(
                    base_url,
                    args=args,
                    kind=kind,
                    max_draft_tokens=max_draft_tokens,
                    timeout=args.request_timeout,
                )
                request_wall_started = time.perf_counter()
                response = _request_json(
                    f"{base_url}/generate", payload, timeout=args.request_timeout
                )
                request_wall_finished = time.perf_counter()
                item = _exact_meta_item(response, request_id=request_id)
            except Exception as exc:
                tail = _read_log_tail(log)
                raise RuntimeError(f"{kind} state replay failed: {exc}\n{tail}") from exc
            finally:
                _terminate_owned_process(process)
            process_wall_finished = time.perf_counter()
            log.flush()
            log.seek(0)
            log_bytes = log.read()
            log_text = log_bytes.decode("utf-8", errors="replace")
    return {
        "command": command,
        "command_sha256": _json_sha256(command),
        "identity": identity,
        "log_sha256": hashlib.sha256(log_bytes).hexdigest(),
        "log_bytes": len(log_bytes),
        "cache_jit_log_markers": _cache_log_markers(log_text),
        "wall_timing_seconds": {
            "process_to_ready": ready_wall - process_wall_started,
            "measured_http_request": request_wall_finished - request_wall_started,
            "process_lifetime_through_shutdown": (
                process_wall_finished - process_wall_started
            ),
        },
        "item": item,
    }


def _hidden_summary(meta: dict[str, Any]) -> dict[str, Any]:
    values = meta.get("hidden_states")
    if (
        not isinstance(values, list)
        or len(values) != QWEN3_1_7B_CONTRACT.hidden_size
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in values
        )
    ):
        raise RuntimeError("exact replay did not return one finite last hidden state")
    numeric = [float(value) for value in values]
    return {
        "capture_contract": "ordinary exact decode, return_hidden_states=last",
        "element_count": len(numeric),
        "canonical_json_sha256": _json_sha256(numeric),
        "l2_norm": math.sqrt(sum(value * value for value in numeric)),
        "max_abs": max(abs(value) for value in numeric),
        "values": numeric,
    }


def _fresh_trace_record(
    records: list[dict[str, Any]], *, prefix_length: int
) -> dict[str, Any]:
    verifies = [record for record in records if record.get("event") == "verify"]
    if len(verifies) != 1:
        raise RuntimeError("fresh-prefix NEXTLAT replay did not execute exactly one verify")
    record = verifies[0]
    if (
        record.get("request_ids") != [REQUEST_ID_NEXTLAT]
        or record.get("prefix_seq_lens") != [prefix_length]
        or record.get("proposal_lens") != [MAX_DRAFT_TOKENS]
    ):
        raise RuntimeError("fresh-prefix verify geometry differs from B1/M2")
    return record


def classify_replay(
    *,
    expected_completion: list[int],
    exact_output: list[int],
    nextlat_output: list[int],
) -> str:
    if exact_output != expected_completion:
        raise RuntimeError("fresh exact replay differs from the witnessed exact trajectory")
    if nextlat_output[: MAX_DRAFT_TOKENS + 1] != expected_completion[
        : MAX_DRAFT_TOKENS + 1
    ]:
        return "inconclusive_identical_window_not_reproduced"
    if nextlat_output[-1] == exact_output[-1]:
        return "fresh_prefix_matches_exact_accumulated_path_drift_implicated"
    return "fresh_prefix_ragged_divergence_observed"


def run(args: argparse.Namespace) -> Path:
    _require_installed_plugin()
    if not torch.cuda.is_available():
        raise RuntimeError("canonical state replay has no CPU fallback")
    if args.gpu_index < 0 or args.gpu_index >= torch.cuda.device_count():
        raise ValueError("gpu-index is outside visible CUDA devices")
    if (
        not args.runtime_image_id.startswith("sha256:")
        or len(args.runtime_image_id) != 71
        or any(
            char not in "0123456789abcdef" for char in args.runtime_image_id[7:]
        )
    ):
        raise ValueError("runtime-image-id must be a full lowercase sha256 image ID")
    if not args.runtime_image_tag or any(char.isspace() for char in args.runtime_image_tag):
        raise ValueError("runtime-image-tag must be a non-empty token")

    cache_before = _cache_snapshot(args.cache_root)

    tie, tie_file_sha256 = _read_receipt(args.tie_receipt.expanduser())
    case = derive_fresh_replay_case(tie)
    sidecar = read_sidecar_metadata(args.sidecar)
    source = admit_sglang_source()
    target = admit_target_checkpoint(
        args.model,
        expected_manifest_sha256=sidecar.target_checkpoint_sha256,
        contract=sidecar.contract,
    )
    tie_provenance = tie["provenance"]
    if (
        tie_provenance.get("target_manifest_sha256") != target.manifest_sha256
        or tie_provenance.get("dynamics_sha256") != sidecar.dynamics_sha256
        or tie_provenance.get("sglang_expected_commit") != AUDITED_SGLANG_COMMIT
    ):
        raise RuntimeError("tie receipt provenance differs from admitted replay artifacts")

    synthetic_record = {
        **case["registered_input"],
        "input_ids": case["fresh_prefix_input_ids"],
    }
    nextlat_payload = _single_payload(
        request_id=REQUEST_ID_NEXTLAT,
        input_ids=case["fresh_prefix_input_ids"],
        max_new_tokens=COMPLETION_TOKENS,
    )

    paired_wall_started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="nextlat-fresh-state-replay-") as directory:
        trace_path = Path(directory) / "fresh-nextlat-trace.jsonl"
        nextlat_server = _serve_once(
            args,
            kind="nextlat_dspark_verify",
            max_draft_tokens=MAX_DRAFT_TOKENS,
            payload=nextlat_payload,
            request_id=REQUEST_ID_NEXTLAT,
            trace_path=trace_path,
        )
        nextlat_result = _parse_request_result(
            nextlat_server["item"],
            request_id=REQUEST_ID_NEXTLAT,
            registered_input=synthetic_record,
            expected_completion_tokens=COMPLETION_TOKENS,
            max_draft_tokens=MAX_DRAFT_TOKENS,
        )
        trace = _read_trace(trace_path)
        trace_sha256 = _sha256_file(trace_path)
        if _trace_output(trace, request_id=REQUEST_ID_NEXTLAT) != nextlat_result[
            "output_ids"
        ]:
            raise RuntimeError("fresh NEXTLAT trace does not reconstruct HTTP output")
        fresh_verify = _fresh_trace_record(
            trace, prefix_length=len(case["fresh_prefix_input_ids"])
        )
        if fresh_verify.get("compact_verify_tokens") != case[
            "expected_ragged_window_ids"
        ]:
            raise RuntimeError("fresh NEXTLAT replay did not verify the witnessed token window")

        requested_token_ids = sorted(
            {
                nextlat_result["output_ids"][-1],
                case["expected_completion_ids"][-1],
            }
        )
        exact_payload = _single_payload(
            request_id=REQUEST_ID_EXACT,
            input_ids=case["fresh_prefix_input_ids"],
            max_new_tokens=COMPLETION_TOKENS,
            top_logprobs_num=args.top_logprobs,
            token_ids_logprob=requested_token_ids,
        )
        exact_payload["return_hidden_states"] = "last"
        exact_server = _serve_once(
            args,
            kind="exact_greedy",
            max_draft_tokens=None,
            payload=exact_payload,
            request_id=REQUEST_ID_EXACT,
        )
        exact_result = _parse_request_result(
            exact_server["item"],
            request_id=REQUEST_ID_EXACT,
            registered_input=synthetic_record,
            expected_completion_tokens=COMPLETION_TOKENS,
            max_draft_tokens=None,
        )
    paired_wall_seconds = time.perf_counter() - paired_wall_started
    cache_after = _cache_snapshot(args.cache_root)
    cache_delta = _cache_delta(cache_before, cache_after)

    classification = classify_replay(
        expected_completion=case["expected_completion_ids"],
        exact_output=exact_result["output_ids"],
        nextlat_output=nextlat_result["output_ids"],
    )
    mismatch = _first_mismatch(exact_result["output_ids"], nextlat_result["output_ids"])
    exact_meta = exact_server["item"].get("meta_info")
    if not isinstance(exact_meta, dict):
        raise RuntimeError("fresh exact replay response lacks meta_info")
    exact_focus = _logprob_position(
        exact_meta,
        position=COMPLETION_TOKENS - 1,
        requested_token_ids=[nextlat_result["output_ids"][-1]],
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "result": classification,
        "claim_scope": {
            "purpose": (
                "single known-context state replay separating accumulated NEXTLAT "
                "KV history from current ragged target-verify divergence"
            ),
            "performance_evidence": False,
            "production_correction": False,
            "fallback": None,
        },
        "arguments": {
            "gpu_index": args.gpu_index,
            "port": args.port,
            "top_logprobs": args.top_logprobs,
            "attention_backend": args.attention_backend,
            "cache_root": str(args.cache_root),
        },
        "provenance": {
            "harness_sha256": _sha256_file(Path(__file__).resolve()),
            "tie_receipt_file_sha256": tie_file_sha256,
            "tie_receipt_sha256": tie["receipt_sha256"],
            "worker_sha256": _sha256_file(Path(__file__).with_name("worker.py")),
            "registered_input": case["registered_input"],
            "target_root": str(target.root),
            "target_manifest_sha256": target.manifest_sha256,
            "sidecar_root": str(sidecar.root),
            "dynamics_sha256": sidecar.dynamics_sha256,
            "sglang_expected_commit": AUDITED_SGLANG_COMMIT,
            "sglang_source_receipt_sha256": source.receipt_sha256,
            "runtime_image_tag": args.runtime_image_tag,
            "runtime_image_id": args.runtime_image_id,
            "image_plugin_source_manifest_sha256": os.environ.get(
                "NEXTLAT_PLUGIN_MANIFEST_SHA256"
            ),
            "gpu": torch.cuda.get_device_name(args.gpu_index),
        },
        "case": {
            "original_mismatch_position": case["mismatch_position"],
            "fresh_prefix_length": len(case["fresh_prefix_input_ids"]),
            "fresh_prefix_input_ids_sha256": _json_sha256(
                case["fresh_prefix_input_ids"]
            ),
            "fresh_prefix_output_count": case["fresh_prefix_output_count"],
            "expected_ragged_window_ids": case["expected_ragged_window_ids"],
            "expected_completion_ids": case["expected_completion_ids"],
            "fresh_first_mismatch_position": mismatch,
        },
        "nextlat_fresh_prefix": {
            "request_payload_sha256": _json_sha256(nextlat_payload),
            "server": {
                key: value for key, value in nextlat_server.items() if key != "item"
            },
            "parsed": nextlat_result,
            "trace_sha256": trace_sha256,
            "verify_record": fresh_verify,
        },
        "exact_fresh_prefix": {
            "request_payload_sha256": _json_sha256(exact_payload),
            "server": {
                key: value for key, value in exact_server.items() if key != "item"
            },
            "parsed": exact_result,
            "final_position_logprobs": exact_focus,
            "last_decode_hidden_state": _hidden_summary(exact_meta),
        },
        "interpretation": {
            "historical_nextlat_kv_drift_required": (
                False
                if classification == "fresh_prefix_ragged_divergence_observed"
                else None
            ),
            "bounded_cause_if_divergence_reproduces": (
                "current ragged multi-token target-forward state/numerics versus "
                "ordinary one-token decode; identical recurrent proposals mean the "
                "recurrent model is not the final-token selector"
            ),
            "not_distinguished": (
                "attention-kernel reduction/layout arithmetic versus another "
                "within-window ragged-forward implementation difference"
            ),
        },
        "timing_seconds": {
            "paired_wall_including_two_server_startups_and_shutdowns": (
                paired_wall_seconds
            ),
            "sum_server_reported_request_e2e": (
                nextlat_result["timing_seconds"]["server_e2e_latency"]
                + exact_result["timing_seconds"]["server_e2e_latency"]
            ),
            "sum_measured_http_request_wall": (
                nextlat_server["wall_timing_seconds"]["measured_http_request"]
                + exact_server["wall_timing_seconds"]["measured_http_request"]
            ),
            "authority_note": (
                "server-reported request e2e includes scheduler/model service time; "
                "the paired wall also includes both cold server lifetimes"
            ),
        },
        "cache_jit": {
            "before": cache_before,
            "after": cache_after,
            "delta": cache_delta,
        },
    }
    return _write_receipt(args.receipt, payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--tie-receipt", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--port", type=int, default=31000)
    parser.add_argument(
        "--attention-backend", choices=ATTENTION_BACKENDS, default="flashinfer"
    )
    parser.add_argument("--cache-root", type=Path, default=Path("/root/.cache"))
    parser.add_argument("--runtime-image-tag", required=True)
    parser.add_argument("--runtime-image-id", required=True)
    parser.add_argument("--top-logprobs", type=int, default=8)
    parser.add_argument("--mem-fraction-static", type=float, default=0.5)
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    return parser


def main() -> None:
    print(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
