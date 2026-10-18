"""Focused B=2 M=2 reducer for the registered request-1 position-1 mismatch."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any

import torch

from .admission import admit_sglang_source, admit_target_checkpoint
from .checkpoint import read_sidecar_metadata
from .constants import AUDITED_SGLANG_COMMIT
from .dspark_eval import (
    _json_sha256,
    _parse_request_result,
    _read_registered_inputs,
    _response_items,
    _sha256_file,
    _write_receipt,
)
from .parity_diagnostic import (
    MAX_DRAFT_TOKENS,
    _first_mismatch,
    _logprob_position,
    _read_trace,
    _sampling,
    _serve,
)
from .smoke_gpu import _request_json, _require_installed_plugin

DIAGNOSTIC_SCHEMA_VERSION = 1
BATCH_SIZE = 2
FOCUS_REQUEST_INDEX = 1
FOCUS_OUTPUT_POSITION = 1


def _batch_payload(
    *,
    request_ids: list[str],
    input_ids: list[list[int]],
    max_new_tokens: int,
    top_logprobs_num: int = 0,
    token_ids_logprob: list[list[int]] | None = None,
) -> dict[str, Any]:
    if len(request_ids) != BATCH_SIZE or len(input_ids) != BATCH_SIZE:
        raise ValueError("B=2 parity reducer requires exactly two request rows")
    wants_logprobs = top_logprobs_num > 0 or token_ids_logprob is not None
    if token_ids_logprob is not None and len(token_ids_logprob) != BATCH_SIZE:
        raise ValueError("batched token_ids_logprob must have exactly two rows")
    return {
        "rid": request_ids,
        "input_ids": input_ids,
        "sampling_params": [
            _sampling(max_new_tokens) for _ in range(BATCH_SIZE)
        ],
        "return_logprob": [wants_logprobs] * BATCH_SIZE,
        "top_logprobs_num": [top_logprobs_num] * BATCH_SIZE,
        "token_ids_logprob": token_ids_logprob,
        "return_text_in_logprobs": False,
        "stream": False,
        "log_metrics": True,
    }


def _batch_request(
    base_url: str,
    args: argparse.Namespace,
    *,
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    response = _request_json(
        f"{base_url}/generate", payload, timeout=args.request_timeout
    )
    return _response_items(response, expected_rids=payload["rid"])


def _warmup(
    base_url: str,
    args: argparse.Namespace,
    *,
    kind: str,
    input_ids: list[list[int]],
) -> dict[str, Any]:
    request_ids = [f"warmup-{kind}-b2-{index}" for index in range(BATCH_SIZE)]
    payload = _batch_payload(
        request_ids=request_ids,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
    )
    items = _batch_request(base_url, args, payload=payload)
    output_ids = [item.get("output_ids") for item in items]
    if any(
        not isinstance(row, list) or len(row) != args.max_new_tokens
        for row in output_ids
    ):
        raise RuntimeError(f"{kind} B=2 warmup returned an invalid output shape")
    return {
        "request_payload_sha256": _json_sha256(payload),
        "output_ids_sha256": _json_sha256(output_ids),
    }


def _first_batched_verify(
    records: list[dict[str, Any]], *, request_ids: list[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    header = records[0]
    if header.get("request_ids") != request_ids:
        raise RuntimeError("NEXTLAT diagnostic header request roster differs")
    events = records[1:]
    if not events or events[0].get("event") != "prefill":
        raise RuntimeError("NEXTLAT B=2 trace has no leading prefill record")
    prefill = events[0]
    prefill_request_ids = prefill.get("request_ids")
    if (
        not isinstance(prefill_request_ids, list)
        or len(prefill_request_ids) != BATCH_SIZE
        or set(prefill_request_ids) != set(request_ids)
    ):
        raise RuntimeError("NEXTLAT B=2 prefill request roster differs")
    verify = next(
        (record for record in events[1:] if record.get("event") == "verify"),
        None,
    )
    verify_request_ids = None if verify is None else verify.get("request_ids")
    if (
        verify is None
        or not isinstance(verify_request_ids, list)
        or len(verify_request_ids) != BATCH_SIZE
        or set(verify_request_ids) != set(request_ids)
    ):
        raise RuntimeError("NEXTLAT B=2 trace has no joint first verify record")
    return prefill, verify


def _trace_focus(
    records: list[dict[str, Any]], *, request_ids: list[str]
) -> dict[str, Any]:
    prefill, verify = _first_batched_verify(records, request_ids=request_ids)
    focus_request_id = request_ids[FOCUS_REQUEST_INDEX]
    prefill_row_index = prefill["request_ids"].index(focus_request_id)
    verify_row_index = verify["request_ids"].index(focus_request_id)
    prefill_tokens = prefill.get("next_token_ids")
    output_rows = verify.get("output_tokens")
    target_rows = verify.get("target_rows")
    if (
        not isinstance(prefill_tokens, list)
        or len(prefill_tokens) != BATCH_SIZE
        or any(not isinstance(token, int) for token in prefill_tokens)
        or not isinstance(output_rows, list)
        or len(output_rows) != BATCH_SIZE
        or any(not isinstance(row, list) or not row for row in output_rows)
        or not isinstance(target_rows, list)
        or len(target_rows) != BATCH_SIZE
        or any(not isinstance(row, list) or not row for row in target_rows)
    ):
        raise RuntimeError("NEXTLAT first B=2 verify trace has an invalid row shape")
    focus_target = target_rows[verify_row_index][0]
    if (
        not isinstance(focus_target, dict)
        or not isinstance(focus_target.get("top_token_ids"), list)
        or not focus_target["top_token_ids"]
        or not isinstance(focus_target.get("top_logits"), list)
        or not focus_target["top_logits"]
    ):
        raise RuntimeError("NEXTLAT focused target-logit trace row is invalid")
    return {
        "request_id": focus_request_id,
        "prefill_row_index": prefill_row_index,
        "verify_row_index": verify_row_index,
        "prefill_token_id": prefill_tokens[prefill_row_index],
        "position_1_output_token_id": output_rows[verify_row_index][0],
        "position_1_target": focus_target,
        "first_verify": verify,
    }


def run(args: argparse.Namespace) -> Path:
    _require_installed_plugin()
    if not torch.cuda.is_available():
        raise RuntimeError("B=2 parity diagnostic has no CPU fallback")
    if args.gpu_index < 0 or args.gpu_index >= torch.cuda.device_count():
        raise ValueError("gpu-index is outside visible CUDA devices")
    if args.max_new_tokens < 4:
        raise ValueError("max-new-tokens must be at least four to exercise M=2")
    if not 2 <= args.top_logprobs <= 32:
        raise ValueError("top-logprobs must be in [2, 32]")

    sidecar = read_sidecar_metadata(args.sidecar)
    registered_path = args.registered_inputs.expanduser()
    registered_sha256 = _sha256_file(registered_path)
    registered = _read_registered_inputs(
        registered_path, vocab_size=sidecar.contract.vocab_size
    )
    if _sha256_file(registered_path) != registered_sha256:
        raise RuntimeError("registered inputs changed during admission")
    selected = registered["inputs"][:BATCH_SIZE]
    if len(selected) != BATCH_SIZE:
        raise RuntimeError("registered input artifact has fewer than two rows")
    input_ids = [record["input_ids"] for record in selected]
    source = admit_sglang_source()
    target = admit_target_checkpoint(
        args.model,
        expected_manifest_sha256=sidecar.target_checkpoint_sha256,
        contract=sidecar.contract,
    )

    nextlat_request_ids = [
        f"nextlat-m2-b2-parity-diagnostic-{index}"
        for index in range(BATCH_SIZE)
    ]
    nextlat_payload = _batch_payload(
        request_ids=nextlat_request_ids,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
    )
    with tempfile.TemporaryDirectory(prefix="nextlat-b2-parity-trace-") as directory:
        trace_path = Path(directory) / "nextlat-b2-trace.jsonl"

        def nextlat_interaction(base_url: str) -> dict[str, Any]:
            return {
                "warmup": _warmup(
                    base_url,
                    args,
                    kind="nextlat-m2",
                    input_ids=input_ids,
                ),
                "measured": _batch_request(
                    base_url, args, payload=nextlat_payload
                ),
            }

        nextlat_server = _serve(
            args,
            kind="nextlat_dspark_verify",
            max_draft_tokens=MAX_DRAFT_TOKENS,
            trace_path=trace_path,
            trace_request_ids=nextlat_request_ids,
            interact=nextlat_interaction,
        )
        nextlat_items = nextlat_server["result"]["measured"]
        nextlat_results = [
            _parse_request_result(
                item,
                request_id=request_id,
                registered_input=record,
                expected_completion_tokens=args.max_new_tokens,
                max_draft_tokens=MAX_DRAFT_TOKENS,
            )
            for item, request_id, record in zip(
                nextlat_items, nextlat_request_ids, selected
            )
        ]
        trace = _read_trace(trace_path)
        trace_sha256 = _sha256_file(trace_path)
        focus_trace = _trace_focus(trace, request_ids=nextlat_request_ids)
        if (
            focus_trace["prefill_token_id"]
            != nextlat_results[FOCUS_REQUEST_INDEX]["output_ids"][0]
            or focus_trace["position_1_output_token_id"]
            != nextlat_results[FOCUS_REQUEST_INDEX]["output_ids"][
                FOCUS_OUTPUT_POSITION
            ]
        ):
            raise RuntimeError("focused trace tokens do not reconstruct HTTP positions 0/1")

        exact_request_ids = [
            f"exact-b2-parity-diagnostic-{index}" for index in range(BATCH_SIZE)
        ]
        requested_token_ids = [
            sorted(set(result["output_ids"])) for result in nextlat_results
        ]
        exact_payload = _batch_payload(
            request_ids=exact_request_ids,
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            top_logprobs_num=args.top_logprobs,
            token_ids_logprob=requested_token_ids,
        )

        def exact_interaction(base_url: str) -> dict[str, Any]:
            return {
                "warmup": _warmup(
                    base_url,
                    args,
                    kind="exact",
                    input_ids=input_ids,
                ),
                "measured": _batch_request(base_url, args, payload=exact_payload),
            }

        exact_server = _serve(
            args,
            kind="exact_greedy",
            max_draft_tokens=None,
            trace_path=None,
            trace_request_ids=None,
            interact=exact_interaction,
        )
        exact_items = exact_server["result"]["measured"]
        exact_results = [
            _parse_request_result(
                item,
                request_id=request_id,
                registered_input=record,
                expected_completion_tokens=args.max_new_tokens,
                max_draft_tokens=None,
            )
            for item, request_id, record in zip(
                exact_items, exact_request_ids, selected
            )
        ]

    if _sha256_file(registered_path) != registered_sha256:
        raise RuntimeError("registered inputs changed during diagnostic")
    mismatches = [
        _first_mismatch(exact["output_ids"], nextlat["output_ids"])
        for exact, nextlat in zip(exact_results, nextlat_results)
    ]
    exact_focus_token = exact_results[FOCUS_REQUEST_INDEX]["output_ids"][
        FOCUS_OUTPUT_POSITION
    ]
    nextlat_focus_token = nextlat_results[FOCUS_REQUEST_INDEX]["output_ids"][
        FOCUS_OUTPUT_POSITION
    ]
    exact_focus = _logprob_position(
        exact_items[FOCUS_REQUEST_INDEX]["meta_info"],
        position=FOCUS_OUTPUT_POSITION,
        requested_token_ids=[nextlat_focus_token],
    )
    nextlat_target = focus_trace["position_1_target"]
    comparison = {
        "request_index": FOCUS_REQUEST_INDEX,
        "output_position": FOCUS_OUTPUT_POSITION,
        "exact_output_token_id": exact_focus_token,
        "nextlat_output_token_id": nextlat_focus_token,
        "output_token_ids_match": exact_focus_token == nextlat_focus_token,
        "exact_context": exact_focus,
        "nextlat_ragged_target": nextlat_target,
        "target_top1_token_ids_match": (
            exact_focus["top"][0]["token_id"]
            == nextlat_target["top_token_ids"][0]
        ),
    }
    observed_registered_failure = (
        mismatches[FOCUS_REQUEST_INDEX] == FOCUS_OUTPUT_POSITION
    )
    payload = {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "result": (
            "registered_failure_observed"
            if observed_registered_failure
            else "registered_failure_not_observed"
        ),
        "claim_scope": {
            "purpose": (
                "focused registered B=2 M=2 request-index-1 output-position-1 "
                "greedy parity reducer"
            ),
            "performance_evidence": False,
            "fallback": None,
        },
        "arguments": {
            "max_new_tokens": args.max_new_tokens,
            "top_logprobs": args.top_logprobs,
            "gpu_index": args.gpu_index,
            "port": args.port,
        },
        "provenance": {
            "harness_sha256": _sha256_file(Path(__file__).resolve()),
            "base_harness_sha256": _sha256_file(
                Path(__file__).with_name("parity_diagnostic.py")
            ),
            "worker_sha256": _sha256_file(Path(__file__).with_name("worker.py")),
            "registered_inputs_file_sha256": registered_sha256,
            "registered_inputs": selected,
            "target_root": str(target.root),
            "target_manifest_sha256": target.manifest_sha256,
            "sidecar_root": str(sidecar.root),
            "dynamics_sha256": sidecar.dynamics_sha256,
            "sglang_expected_commit": AUDITED_SGLANG_COMMIT,
            "sglang_source_receipt_sha256": source.receipt_sha256,
            "gpu": torch.cuda.get_device_name(args.gpu_index),
        },
        "first_mismatch_positions": mismatches,
        "focused_comparison": comparison,
        "nextlat": {
            "request_payload_sha256": _json_sha256(nextlat_payload),
            "warmup": nextlat_server["result"]["warmup"],
            "server": {
                key: value for key, value in nextlat_server.items() if key != "result"
            },
            "parsed": nextlat_results,
            "trace_sha256": trace_sha256,
            "trace_request_ids": nextlat_request_ids,
            "trace": trace,
            "focused_first_verify": focus_trace,
        },
        "exact": {
            "request_payload_sha256": _json_sha256(exact_payload),
            "warmup": exact_server["result"]["warmup"],
            "server": {
                key: value for key, value in exact_server.items() if key != "result"
            },
            "parsed": exact_results,
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
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--top-logprobs", type=int, default=8)
    parser.add_argument("--mem-fraction-static", type=float, default=0.5)
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    return parser


def main() -> None:
    print(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
