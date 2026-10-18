"""Fail-closed full-M8 CUDA-graph bank for the observed B8 drain path.

This is an isolated diagnostic overlay, not a production abstraction.  It
captures only the five request batch sizes observed in the registered B8/32
canary.  Each graph is keyed by the exact compact verify-row count ``9 * B``;
mixed proposal widths and every unregistered batch size remain eager.

The pinned SGLang ragged capturer normally chooses ``min(num_tokens, max_bs)``
request slots for every token bucket.  That is correct for arbitrary ragged
layouts, but it would pad a 9-row full-M8 B1 bucket to eight request slots in a
multi-bucket bank.  The narrow slot override below maps the admitted uniform
M8 buckets back to their exact request counts before native SGLang capture and
replay.  No attention, model, KV, or output implementation is replaced.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import torch

from .config import target_verify_graph_diagnostic_enabled


GRAPH_BATCH_SIZES = (1, 3, 4, 7, 8)
GRAPH_VERIFY_WIDTH = 9
GRAPH_PROPOSAL_LEN = GRAPH_VERIFY_WIDTH - 1
GRAPH_TOKEN_BUCKETS = tuple(
    batch_size * GRAPH_VERIFY_WIDTH for batch_size in GRAPH_BATCH_SIZES
)

_PATCHED = False


def _is_nextlat_model_runner(model_runner: Any) -> bool:
    algorithm = getattr(model_runner, "spec_algorithm", None)
    return getattr(algorithm, "name", None) == "NEXTLAT"


def _is_nextlat_runner(runner: Any) -> bool:
    return _is_nextlat_model_runner(getattr(runner, "model_runner", None))


def _batch_size_for_token_bucket(num_tokens: int) -> int:
    if num_tokens not in GRAPH_TOKEN_BUCKETS:
        raise RuntimeError(
            "NEXTLAT graph bank received an unregistered capture token bucket"
        )
    batch_size, remainder = divmod(num_tokens, GRAPH_VERIFY_WIDTH)
    if remainder or batch_size not in GRAPH_BATCH_SIZES:
        raise RuntimeError("NEXTLAT graph bank token bucket is not exact full-M8")
    return batch_size


def _is_exact_runtime_geometry(forward_batch: Any) -> bool:
    spec_info = getattr(forward_batch, "spec_info", None)
    layout = getattr(spec_info, "ragged_verify_layout", None)
    batch_size = getattr(forward_batch, "batch_size", None)
    if batch_size not in GRAPH_BATCH_SIZES or layout is None:
        return False
    rows = batch_size * GRAPH_VERIFY_WIDTH
    return bool(
        layout.bs == batch_size
        and layout.verify_lens_cpu == [GRAPH_VERIFY_WIDTH] * batch_size
        and layout.total_verify_tokens == rows
        and layout.graph_num_tokens == rows
    )


def install_target_verify_graph_overlay() -> None:
    """Patch only exact capture slots/spec construction and graph admission."""

    global _PATCHED
    if not target_verify_graph_diagnostic_enabled() or _PATCHED:
        return

    from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
        DecodeCudaGraphRunner,
    )
    from sglang.srt.speculative.dflash_info import DFlashVerifyInput
    from sglang.srt.speculative.dflash_utils import (
        resolve_dflash_verify_mask_policy,
    )

    original_ragged_capture_slots = DecodeCudaGraphRunner._ragged_capture_slots
    original_get_spec_info = DecodeCudaGraphRunner.get_spec_info
    original_can_run_graph = DecodeCudaGraphRunner.can_run_graph

    def ragged_capture_slots(runner: Any, num_tokens: int) -> int:
        if not _is_nextlat_runner(runner):
            return original_ragged_capture_slots(runner, num_tokens)
        return _batch_size_for_token_bucket(num_tokens)

    def get_spec_info(runner: Any, num_tokens: int):
        if not _is_nextlat_runner(runner):
            return original_get_spec_info(runner, num_tokens)
        if runner.model_runner.is_draft_worker:
            raise RuntimeError("NEXTLAT graph bank cannot run on a draft worker")
        _, build_custom_mask = resolve_dflash_verify_mask_policy(runner.attn_backend)
        return DFlashVerifyInput(
            draft_token=None,
            positions=None,
            draft_token_num=runner.captured_req_width,
            custom_mask=runner.buffers.custom_mask if build_custom_mask else None,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            ragged_verify_layout=runner._capture_ragged_verify_layout(num_tokens),
        )

    def can_run_graph(runner: Any, forward_batch: Any):
        if _is_nextlat_runner(runner) and not _is_exact_runtime_geometry(forward_batch):
            return False
        return original_can_run_graph(runner, forward_batch)

    DecodeCudaGraphRunner._ragged_capture_slots = ragged_capture_slots
    DecodeCudaGraphRunner.get_spec_info = get_spec_info
    DecodeCudaGraphRunner.can_run_graph = can_run_graph
    _PATCHED = True


def assert_captured_runner(runner: Any) -> dict[str, Any]:
    """Prove all five native captures are FULL, compact, and unpadded."""

    from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

    if runner is None or not _is_nextlat_runner(runner):
        raise RuntimeError("NEXTLAT graph bank has no target graph runner")
    expected = {
        "ragged_verify_mode": True,
        "disable_padding": False,
        "captured_req_width": GRAPH_VERIFY_WIDTH,
        "capture_bs": list(GRAPH_BATCH_SIZES),
        "capture_num_tokens": list(GRAPH_TOKEN_BUCKETS),
        "max_bs": max(GRAPH_BATCH_SIZES),
    }
    observed = {
        "ragged_verify_mode": runner.ragged_verify_mode,
        "disable_padding": runner.disable_padding,
        "captured_req_width": runner.captured_req_width,
        "capture_bs": list(runner.capture_bs),
        "capture_num_tokens": list(runner.capture_num_tokens or []),
        "max_bs": runner.max_bs,
    }
    if observed != expected:
        raise RuntimeError(
            f"NEXTLAT graph bank capture contract differs: {observed!r} != {expected!r}"
        )
    if runner.capture_hidden_mode != CaptureHiddenMode.FULL:
        raise RuntimeError("NEXTLAT graph bank did not capture FULL hidden states")
    if not runner.attn_backend.supports_ragged_verify_graph:
        raise RuntimeError("NEXTLAT graph bank backend lacks ragged-verify support")

    layouts: dict[str, Any] = {}
    for batch_size, rows in zip(GRAPH_BATCH_SIZES, GRAPH_TOKEN_BUCKETS):
        layout = runner._captured_ragged_layouts.get(rows)
        if (
            layout is None
            or layout.bs != batch_size
            or layout.verify_lens_cpu != [GRAPH_VERIFY_WIDTH] * batch_size
            or layout.total_verify_tokens != rows
            or layout.graph_num_tokens != rows
        ):
            raise RuntimeError(
                f"NEXTLAT graph bank captured layout is not exact B{batch_size}x9"
            )
        layouts[str(batch_size)] = {
            "rows": rows,
            "verify_lens": list(layout.verify_lens_cpu),
            "layout_bs": layout.bs,
        }

    output_buffer = getattr(runner.buffers, "next_token_logits_buffer", None)
    if (
        not isinstance(output_buffer, torch.Tensor)
        or output_buffer.shape[0] < GRAPH_TOKEN_BUCKETS[-1]
    ):
        raise RuntimeError("NEXTLAT graph bank has no max-width static logits buffer")
    return {
        **observed,
        "capture_hidden_mode": runner.capture_hidden_mode.name,
        "captured_layouts_by_batch_size": layouts,
        "static_logits_buffer_shape": list(output_buffer.shape),
    }


def replay_snapshot(
    runner: Any,
    *,
    request_ids: list[str],
    proposal_lens: list[int],
    verify_lens: list[int],
) -> dict[str, Any]:
    """Prove one registered replay selected its exact unpadded native graph."""

    batch_size = len(request_ids)
    if batch_size not in GRAPH_BATCH_SIZES:
        raise RuntimeError("NEXTLAT graph bank replay has an unregistered batch size")
    rows = batch_size * GRAPH_VERIFY_WIDTH
    if proposal_lens != [GRAPH_PROPOSAL_LEN] * batch_size:
        raise RuntimeError("NEXTLAT graph bank replay received non-M8 proposals")
    if verify_lens != [GRAPH_VERIFY_WIDTH] * batch_size:
        raise RuntimeError("NEXTLAT graph bank replay received non-9-row verification")

    row_starts = list(range(0, rows, GRAPH_VERIFY_WIDTH))
    key = runner._replay_graph_key
    geometry = {
        "batch_size": batch_size,
        "proposal_lens": proposal_lens,
        "verify_lens": verify_lens,
        "row_starts": row_starts,
        "total_verify_rows": rows,
        "raw_bs": runner.raw_bs,
        "padded_bs": runner.bs,
        "raw_rows": runner.raw_num_token,
        "padded_rows": runner._ragged_graph_size,
        "graph_key_size": getattr(key, "size", None),
        "capture_token_buckets": list(runner.capture_num_tokens or []),
    }
    expected = {
        "batch_size": batch_size,
        "proposal_lens": [GRAPH_PROPOSAL_LEN] * batch_size,
        "verify_lens": [GRAPH_VERIFY_WIDTH] * batch_size,
        "row_starts": row_starts,
        "total_verify_rows": rows,
        "raw_bs": batch_size,
        "padded_bs": batch_size,
        "raw_rows": rows,
        "padded_rows": rows,
        "graph_key_size": rows,
        "capture_token_buckets": list(GRAPH_TOKEN_BUCKETS),
    }
    if geometry != expected:
        raise RuntimeError(
            f"NEXTLAT graph bank replay padded or drifted: {geometry!r} != {expected!r}"
        )

    pointers: dict[str, int] = {}
    for name in (
        "input_ids",
        "positions",
        "out_cache_loc",
        "req_pool_indices",
        "seq_lens",
        "next_token_logits_buffer",
    ):
        tensor = getattr(runner.buffers, name, None)
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError(f"NEXTLAT graph bank static buffer {name!r} is missing")
        pointers[name] = tensor.data_ptr()
    canonical = json.dumps(geometry, sort_keys=True, separators=(",", ":")).encode()
    return {
        "request_ids": request_ids,
        "graph_bucket_batch_size": batch_size,
        "ordered_geometry": geometry,
        "ordered_geometry_sha256": hashlib.sha256(canonical).hexdigest(),
        "static_input_output_pointers": pointers,
    }
