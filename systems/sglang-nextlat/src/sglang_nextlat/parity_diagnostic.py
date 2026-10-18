"""One-case H100 diagnostic for exact-vs-NEXTLAT greedy parity failures."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import tempfile
from collections.abc import Callable
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
    _parse_request_result,
    _read_log_tail,
    _read_registered_inputs,
    _response_items,
    _server_command,
    _server_environment,
    _sha256_file,
    _termination_guard,
    _write_receipt,
)
from .smoke_gpu import (
    _request_json,
    _require_installed_plugin,
    _terminate_owned_process,
    _wait_ready,
)

DIAGNOSTIC_SCHEMA_VERSION = 1
MAX_DRAFT_TOKENS = 2


def _sampling(max_new_tokens: int) -> dict[str, Any]:
    return {
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


def _single_payload(
    *,
    request_id: str,
    input_ids: list[int],
    max_new_tokens: int,
    top_logprobs_num: int = 0,
    token_ids_logprob: list[int] | None = None,
) -> dict[str, Any]:
    return {
        "rid": request_id,
        "input_ids": input_ids,
        "sampling_params": _sampling(max_new_tokens),
        "return_logprob": top_logprobs_num > 0 or token_ids_logprob is not None,
        "top_logprobs_num": top_logprobs_num,
        "token_ids_logprob": token_ids_logprob,
        "return_text_in_logprobs": False,
        "stream": False,
        "log_metrics": True,
    }


def _serve(
    args: argparse.Namespace,
    *,
    kind: str,
    max_draft_tokens: int | None,
    trace_path: Path | None,
    trace_request_ids: list[str] | None,
    interact: Callable[[str], Any],
) -> dict[str, Any]:
    _assert_port_unused(args.port)
    command = _server_command(
        args, kind=kind, max_draft_tokens=max_draft_tokens
    )
    environment = _server_environment(kind=kind, gpu_index=args.gpu_index)
    environment.pop("NEXTLAT_DIAGNOSTIC_TRACE", None)
    environment.pop("NEXTLAT_DIAGNOSTIC_RID", None)
    environment.pop("NEXTLAT_DIAGNOSTIC_RIDS", None)
    if trace_path is not None:
        if (
            not trace_request_ids
            or any(not isinstance(value, str) or not value for value in trace_request_ids)
            or len(set(trace_request_ids)) != len(trace_request_ids)
        ):
            raise ValueError("trace path requires unique diagnostic request ids")
        environment["NEXTLAT_DIAGNOSTIC_TRACE"] = str(trace_path)
        environment["NEXTLAT_DIAGNOSTIC_RIDS"] = json.dumps(
            trace_request_ids, separators=(",", ":")
        )
    elif trace_request_ids is not None:
        raise ValueError("diagnostic request ids require a trace path")
    base_url = f"http://127.0.0.1:{args.port}"
    with tempfile.TemporaryDirectory(prefix="nextlat-parity-server-") as directory:
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
                identity = _assert_server_identity(
                    base_url,
                    args=args,
                    kind=kind,
                    max_draft_tokens=max_draft_tokens,
                    timeout=args.request_timeout,
                )
                result = interact(base_url)
                log_tail = _read_log_tail(log)
            except Exception as exc:
                tail = _read_log_tail(log)
                raise RuntimeError(f"{kind} diagnostic failed: {exc}\n{tail}") from exc
            finally:
                _terminate_owned_process(process)
    return {
        "command": command,
        "command_sha256": _json_sha256(command),
        "identity": identity,
        "log_tail_sha256": hashlib.sha256(log_tail.encode("utf-8")).hexdigest(),
        "result": result,
    }


def _exact_meta_item(response: Any, *, request_id: str) -> dict[str, Any]:
    return _response_items(response, expected_rids=[request_id])[0]


def _warmup(base_url: str, args: argparse.Namespace, *, kind: str, input_ids: list[int]) -> dict[str, Any]:
    request_id = f"warmup-{kind}-b1"
    payload = _single_payload(
        request_id=request_id,
        input_ids=input_ids,
        max_new_tokens=10,
    )
    item = _exact_meta_item(
        _request_json(
            f"{base_url}/generate", payload, timeout=args.request_timeout
        ),
        request_id=request_id,
    )
    output_ids = item.get("output_ids")
    if not isinstance(output_ids, list) or len(output_ids) != 10:
        raise RuntimeError(f"{kind} warmup did not return ten output tokens")
    return {
        "request_payload_sha256": _json_sha256(payload),
        "output_ids_sha256": _json_sha256(output_ids),
    }


def _logprob_tuple(value: Any, *, label: str) -> dict[str, Any]:
    if (
        not isinstance(value, list)
        or len(value) < 2
        or isinstance(value[0], bool)
        or not isinstance(value[0], (int, float))
        or not math.isfinite(float(value[0]))
        or isinstance(value[1], bool)
        or not isinstance(value[1], int)
        or value[1] < 0
    ):
        raise RuntimeError(f"invalid SGLang {label} tuple")
    return {"logprob": float(value[0]), "token_id": value[1]}


def _logprob_position(
    meta: dict[str, Any],
    *,
    position: int,
    requested_token_ids: list[int],
) -> dict[str, Any]:
    top_rows = meta.get("output_top_logprobs")
    selected_rows = meta.get("output_token_logprobs")
    requested_rows = meta.get("output_token_ids_logprobs")
    if (
        not isinstance(top_rows, list)
        or position >= len(top_rows)
        or not isinstance(top_rows[position], list)
        or len(top_rows[position]) < 2
        or not isinstance(selected_rows, list)
        or position >= len(selected_rows)
        or not isinstance(requested_rows, list)
        or position >= len(requested_rows)
        or not isinstance(requested_rows[position], list)
    ):
        raise RuntimeError("SGLang exact response lacks requested output logprobs")
    top = [
        _logprob_tuple(value, label="top-logprob") for value in top_rows[position]
    ]
    requested = [
        _logprob_tuple(value, label="token-id logprob")
        for value in requested_rows[position]
    ]
    by_token = {item["token_id"]: item["logprob"] for item in requested}
    missing = sorted(set(requested_token_ids) - set(by_token))
    if missing:
        raise RuntimeError(f"SGLang omitted requested token logprobs: {missing}")
    return {
        "selected": _logprob_tuple(
            selected_rows[position], label="selected-token logprob"
        ),
        "top": top,
        "top1_minus_top2_logprob": top[0]["logprob"] - top[1]["logprob"],
        "requested_token_logprobs": {
            str(token_id): by_token[token_id] for token_id in requested_token_ids
        },
    }


def _read_trace(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("NEXTLAT diagnostic trace was not published")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid diagnostic trace line {line_number}") from exc
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            raise RuntimeError(f"invalid diagnostic trace record {line_number}")
        records.append(value)
    if not records or records[0].get("event") != "header":
        raise RuntimeError("NEXTLAT diagnostic trace has no header")
    return records


def _trace_output(records: list[dict[str, Any]], *, request_id: str) -> list[int]:
    outputs: list[int] = []
    events = records[1:]
    if not events or events[0].get("event") != "prefill":
        raise RuntimeError("NEXTLAT diagnostic trace has no prefill record")
    prefill = events[0]
    if prefill.get("request_ids") != [request_id]:
        raise RuntimeError("NEXTLAT diagnostic prefill request identity differs")
    tokens = prefill.get("next_token_ids")
    if not isinstance(tokens, list) or len(tokens) != 1 or not isinstance(tokens[0], int):
        raise RuntimeError("NEXTLAT diagnostic prefill token is invalid")
    outputs.append(tokens[0])
    for record in events[1:]:
        if record.get("event") != "verify" or record.get("request_ids") != [request_id]:
            raise RuntimeError("NEXTLAT diagnostic verify record differs")
        rows = record.get("output_tokens")
        if (
            not isinstance(rows, list)
            or len(rows) != 1
            or not isinstance(rows[0], list)
            or any(not isinstance(token, int) for token in rows[0])
        ):
            raise RuntimeError("NEXTLAT diagnostic output row is invalid")
        outputs.extend(rows[0])
    return outputs


def _trace_counters(records: list[dict[str, Any]]) -> dict[str, Any]:
    verifies = [record for record in records if record.get("event") == "verify"]
    histogram = [0, 0, 0]
    actual_proposals = 0
    for record in verifies:
        correct = record.get("num_correct_drafts")
        proposal_lens = record.get("proposal_lens")
        if (
            not isinstance(correct, list)
            or len(correct) != 1
            or not isinstance(correct[0], int)
            or correct[0] < 0
            or correct[0] > MAX_DRAFT_TOKENS
            or not isinstance(proposal_lens, list)
            or len(proposal_lens) != 1
            or not isinstance(proposal_lens[0], int)
        ):
            raise RuntimeError("NEXTLAT trace acceptance counter is invalid")
        histogram[correct[0]] += 1
        actual_proposals += proposal_lens[0]
    return {
        "verify_count": len(verifies),
        "num_correct_drafts": sum(
            accepted * count for accepted, count in enumerate(histogram)
        ),
        "actual_proposed_drafts": actual_proposals,
        "configured_width_proposed_drafts": len(verifies) * MAX_DRAFT_TOKENS,
        "accept_histogram": histogram,
    }


def _first_mismatch(left: list[int], right: list[int]) -> int | None:
    for index, (left_token, right_token) in enumerate(zip(left, right)):
        if left_token != right_token:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def _mismatch_trace_record(
    records: list[dict[str, Any]], *, position: int
) -> dict[str, Any] | None:
    derived_start = 0
    for record in records:
        if record.get("event") not in {"prefill", "verify"}:
            continue
        if record["event"] == "prefill":
            length = 1
        else:
            rows = record.get("output_tokens")
            length = len(rows[0]) if isinstance(rows, list) and rows else 0
        if derived_start <= position < derived_start + length:
            local = position - derived_start
            result = dict(record)
            result["derived_output_start_index"] = derived_start
            result["mismatch_local_output_index"] = local
            if record["event"] == "verify":
                result["mismatch_target_row"] = record["target_rows"][0][local]
            else:
                result["mismatch_target_row"] = record["target_top_logits"][0]
            return result
        derived_start += length
    return None


def run(args: argparse.Namespace) -> Path:
    _require_installed_plugin()
    if not torch.cuda.is_available():
        raise RuntimeError("parity diagnostic has no CPU fallback")
    if args.gpu_index < 0 or args.gpu_index >= torch.cuda.device_count():
        raise ValueError("gpu-index is outside visible CUDA devices")
    if args.max_new_tokens < 4:
        raise ValueError("max-new-tokens must be at least four")
    if not 2 <= args.top_logprobs <= 32:
        raise ValueError("top-logprobs must be in [2, 32]")
    if not 0 <= args.window <= 16:
        raise ValueError("window must be in [0, 16]")

    sidecar = read_sidecar_metadata(args.sidecar)
    registered_path = args.registered_inputs.expanduser()
    registered_sha256 = _sha256_file(registered_path)
    registered = _read_registered_inputs(
        registered_path, vocab_size=sidecar.contract.vocab_size
    )
    if _sha256_file(registered_path) != registered_sha256:
        raise RuntimeError("registered inputs changed during admission")
    record = registered["inputs"][0]
    source = admit_sglang_source()
    target = admit_target_checkpoint(
        args.model,
        expected_manifest_sha256=sidecar.target_checkpoint_sha256,
        contract=sidecar.contract,
    )

    request_id = "nextlat-m2-b1-parity-diagnostic"
    with tempfile.TemporaryDirectory(prefix="nextlat-parity-trace-") as directory:
        trace_path = Path(directory) / "nextlat-trace.jsonl"
        nextlat_payload = _single_payload(
            request_id=request_id,
            input_ids=record["input_ids"],
            max_new_tokens=args.max_new_tokens,
        )

        def nextlat_interaction(base_url: str) -> dict[str, Any]:
            warmup = _warmup(
                base_url,
                args,
                kind="nextlat-m2",
                input_ids=record["input_ids"],
            )
            measured = _request_json(
                f"{base_url}/generate",
                nextlat_payload,
                timeout=args.request_timeout,
            )
            return {"warmup": warmup, "measured": measured}

        nextlat_server = _serve(
            args,
            kind="nextlat_dspark_verify",
            max_draft_tokens=MAX_DRAFT_TOKENS,
            trace_path=trace_path,
            trace_request_ids=[request_id],
            interact=nextlat_interaction,
        )
        nextlat_item = _exact_meta_item(
            nextlat_server["result"]["measured"], request_id=request_id
        )
        nextlat_result = _parse_request_result(
            nextlat_item,
            request_id=request_id,
            registered_input=record,
            expected_completion_tokens=args.max_new_tokens,
            max_draft_tokens=MAX_DRAFT_TOKENS,
        )
        trace = _read_trace(trace_path)
        trace_sha256 = _sha256_file(trace_path)
        trace_ids = _trace_output(trace, request_id=request_id)
        trace_counters = _trace_counters(trace)

        def exact_interaction(base_url: str) -> dict[str, Any]:
            warmup = _warmup(
                base_url,
                args,
                kind="exact",
                input_ids=record["input_ids"],
            )
            requested_ids = sorted(set(nextlat_result["output_ids"]))
            exact_payload = _single_payload(
                request_id="exact-b1-parity-diagnostic",
                input_ids=record["input_ids"],
                max_new_tokens=args.max_new_tokens,
                top_logprobs_num=args.top_logprobs,
                token_ids_logprob=requested_ids,
            )
            exact_response = _request_json(
                f"{base_url}/generate", exact_payload, timeout=args.request_timeout
            )
            exact_item = _exact_meta_item(
                exact_response, request_id="exact-b1-parity-diagnostic"
            )
            exact_result = _parse_request_result(
                exact_item,
                request_id="exact-b1-parity-diagnostic",
                registered_input=record,
                expected_completion_tokens=args.max_new_tokens,
                max_draft_tokens=None,
            )
            mismatch = _first_mismatch(
                exact_result["output_ids"], nextlat_result["output_ids"]
            )
            positions: list[dict[str, Any]] = []
            if mismatch is not None and mismatch < args.max_new_tokens:
                start = max(0, mismatch - args.window)
                stop = min(args.max_new_tokens, mismatch + args.window + 1)
                exact_meta = exact_item["meta_info"]
                for position in range(start, stop):
                    exact_token = exact_result["output_ids"][position]
                    nextlat_token = nextlat_result["output_ids"][position]
                    positions.append(
                        {
                            "position": position,
                            "exact_token_id": exact_token,
                            "nextlat_token_id": nextlat_token,
                            "exact_context_target": _logprob_position(
                                exact_meta,
                                position=position,
                                requested_token_ids=[nextlat_token],
                            ),
                        }
                    )
            return {
                "warmup": warmup,
                "request_payload_sha256": _json_sha256(exact_payload),
                "parsed": exact_result,
                "first_mismatch_position": mismatch,
                "neighborhood": positions,
            }

        exact_server = _serve(
            args,
            kind="exact_greedy",
            max_draft_tokens=None,
            trace_path=None,
            trace_request_ids=None,
            interact=exact_interaction,
        )

    if _sha256_file(registered_path) != registered_sha256:
        raise RuntimeError("registered inputs changed during diagnostic")
    mismatch = exact_server["result"]["first_mismatch_position"]
    trace_match = trace_ids == nextlat_result["output_ids"]
    payload = {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "result": "mismatch_observed" if mismatch is not None else "no_mismatch_observed",
        "claim_scope": {
            "purpose": "single registered B=1 M=2 greedy parity diagnostic",
            "performance_evidence": False,
            "fallback": None,
        },
        "arguments": {
            "max_new_tokens": args.max_new_tokens,
            "top_logprobs": args.top_logprobs,
            "window": args.window,
            "gpu_index": args.gpu_index,
            "port": args.port,
        },
        "provenance": {
            "harness_sha256": _sha256_file(Path(__file__).resolve()),
            "worker_sha256": _sha256_file(Path(__file__).with_name("worker.py")),
            "registered_inputs_file_sha256": registered_sha256,
            "registered_input": record,
            "target_root": str(target.root),
            "target_manifest_sha256": target.manifest_sha256,
            "sidecar_root": str(sidecar.root),
            "dynamics_sha256": sidecar.dynamics_sha256,
            "sglang_expected_commit": AUDITED_SGLANG_COMMIT,
            "sglang_source_receipt_sha256": source.receipt_sha256,
            "gpu": torch.cuda.get_device_name(args.gpu_index),
        },
        "first_mismatch_position": mismatch,
        "nextlat": {
            "request_payload_sha256": _json_sha256(nextlat_payload),
            "warmup": nextlat_server["result"]["warmup"],
            "server": {key: value for key, value in nextlat_server.items() if key != "result"},
            "parsed": nextlat_result,
            "trace_sha256": trace_sha256,
            "trace_reconstructed_output_ids": trace_ids,
            "trace_matches_http_output": trace_match,
            "trace_counters": trace_counters,
            "trace_counters_match_http": (
                trace_counters["verify_count"]
                == nextlat_result["speculative"]["verify_count"]
                and trace_counters["num_correct_drafts"]
                == nextlat_result["speculative"]["num_correct_drafts"]
                and trace_counters["accept_histogram"]
                == nextlat_result["speculative"]["accept_histogram"]
            ),
            "trace": trace,
            "mismatch_trace_record": (
                None
                if mismatch is None
                else _mismatch_trace_record(trace, position=mismatch)
            ),
        },
        "exact": {
            "server": {key: value for key, value in exact_server.items() if key != "result"},
            **exact_server["result"],
        },
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
    parser.add_argument("--top-logprobs", type=int, default=8)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--mem-fraction-static", type=float, default=0.5)
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    return parser


def main() -> None:
    print(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
