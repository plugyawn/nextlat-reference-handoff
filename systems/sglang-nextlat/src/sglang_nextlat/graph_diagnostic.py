"""Fail-closed overlay for one exact B=8, M=8 FA3 CUDA-graph diagnostic.

This module is deliberately not a production graph integration.  It is active
only when ``NEXTLAT_TARGET_VERIFY_GRAPH_DIAGNOSTIC=1`` passed config admission.
It teaches the pinned SGLang graph capturer about NEXTLAT's DFlash-shaped FULL
hidden-state verify input without changing NEXTLAT's scheduler identity, and it
rejects every graph replay except the one registered 8 x 9 compact geometry.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import torch

from .config import (
    TARGET_VERIFY_GRAPH_VERIFY_WIDTH,
    target_verify_graph_batch_size,
    target_verify_graph_diagnostic_enabled,
)

_PATCHED = False


def _is_nextlat_runner(runner: Any) -> bool:
    algorithm = getattr(getattr(runner, "model_runner", None), "spec_algorithm", None)
    return getattr(algorithm, "name", None) == "NEXTLAT"


def _is_exact_runtime_geometry(forward_batch: Any) -> bool:
    batch_size = target_verify_graph_batch_size()
    rows = batch_size * TARGET_VERIFY_GRAPH_VERIFY_WIDTH
    spec_info = getattr(forward_batch, "spec_info", None)
    layout = getattr(spec_info, "ragged_verify_layout", None)
    return bool(
        layout is not None
        and forward_batch.batch_size == batch_size
        and layout.bs == batch_size
        and layout.verify_lens_cpu
        == [TARGET_VERIFY_GRAPH_VERIFY_WIDTH] * batch_size
        and layout.total_verify_tokens == rows
        and layout.graph_num_tokens == rows
    )


def install_target_verify_graph_overlay() -> None:
    """Patch only capture-spec construction and exact-shape graph admission."""

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

    original_get_spec_info = DecodeCudaGraphRunner.get_spec_info
    original_can_run_graph = DecodeCudaGraphRunner.can_run_graph

    def get_spec_info(runner: Any, num_tokens: int):
        if not _is_nextlat_runner(runner):
            return original_get_spec_info(runner, num_tokens)
        if runner.model_runner.is_draft_worker:
            raise RuntimeError("NEXTLAT graph diagnostic cannot run on a draft worker")
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

    DecodeCudaGraphRunner.get_spec_info = get_spec_info
    DecodeCudaGraphRunner.can_run_graph = can_run_graph
    _PATCHED = True


def assert_captured_runner(runner: Any) -> dict[str, Any]:
    """Prove the capture is FULL, compact, one-bucket, and pointer-backed."""

    from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

    if runner is None or not _is_nextlat_runner(runner):
        raise RuntimeError("NEXTLAT graph diagnostic has no target graph runner")
    batch_size = target_verify_graph_batch_size()
    rows = batch_size * TARGET_VERIFY_GRAPH_VERIFY_WIDTH
    expected = {
        "ragged_verify_mode": True,
        # Native compact-ragged capture rejects the global disable-padding flag.
        # The NEXTLAT can_run overlay is stricter: only the registered exact bucket
        # is admitted, and replay_snapshot below proves raw == padded.
        "disable_padding": False,
        "captured_req_width": TARGET_VERIFY_GRAPH_VERIFY_WIDTH,
        "capture_bs": [batch_size],
        "capture_num_tokens": [rows],
        "max_bs": batch_size,
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
            f"NEXTLAT graph capture contract differs: {observed!r} != {expected!r}"
        )
    if runner.capture_hidden_mode != CaptureHiddenMode.FULL:
        raise RuntimeError("NEXTLAT graph did not capture FULL hidden states")
    if not runner.attn_backend.supports_ragged_verify_graph:
        raise RuntimeError("NEXTLAT graph backend lacks ragged-verify support")
    layout = runner._captured_ragged_layouts.get(rows)
    if (
        layout is None
        or layout.bs != batch_size
        or layout.verify_lens_cpu
        != [TARGET_VERIFY_GRAPH_VERIFY_WIDTH] * batch_size
        or layout.total_verify_tokens != rows
        or layout.graph_num_tokens != rows
    ):
        raise RuntimeError("NEXTLAT graph captured layout is not exact 8 x 9")
    output_buffer = getattr(runner.buffers, "next_token_logits_buffer", None)
    if not isinstance(output_buffer, torch.Tensor) or output_buffer.shape[0] < rows:
        raise RuntimeError("NEXTLAT graph has no exact static logits output buffer")
    return {
        **observed,
        "capture_hidden_mode": runner.capture_hidden_mode.name,
        "captured_verify_lens": list(layout.verify_lens_cpu),
        "static_logits_buffer_shape": list(output_buffer.shape),
    }


def replay_snapshot(
    runner: Any,
    *,
    request_ids: list[str],
    proposal_lens: list[int],
    verify_lens: list[int],
) -> dict[str, Any]:
    """Capture exact ordered geometry and static graph-buffer pointers."""

    batch_size = target_verify_graph_batch_size()
    rows = batch_size * TARGET_VERIFY_GRAPH_VERIFY_WIDTH
    if proposal_lens != [8] * batch_size or verify_lens != [9] * batch_size:
        raise RuntimeError("NEXTLAT graph replay snapshot received non-fixed-M8 geometry")
    row_starts: list[int] = []
    cursor = 0
    for length in verify_lens:
        row_starts.append(cursor)
        cursor += length
    key = runner._replay_graph_key
    geometry = {
        "proposal_lens": proposal_lens,
        "verify_lens": verify_lens,
        "row_starts": row_starts,
        "total_verify_rows": cursor,
        "raw_bs": runner.raw_bs,
        "padded_bs": runner.bs,
        "raw_rows": runner.raw_num_token,
        "padded_rows": runner._ragged_graph_size,
        "graph_key_size": getattr(key, "size", None),
        "capture_token_buckets": list(runner.capture_num_tokens or []),
    }
    expected_geometry = {
        "proposal_lens": [8] * batch_size,
        "verify_lens": [9] * batch_size,
        "row_starts": list(range(0, rows, 9)),
        "total_verify_rows": rows,
        "raw_bs": batch_size,
        "padded_bs": batch_size,
        "raw_rows": rows,
        "padded_rows": rows,
        "graph_key_size": rows,
        "capture_token_buckets": [rows],
    }
    if geometry != expected_geometry:
        raise RuntimeError(
            f"NEXTLAT graph replay padded or drifted: {geometry!r} != {expected_geometry!r}"
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
            raise RuntimeError(f"NEXTLAT graph static buffer {name!r} is missing")
        pointers[name] = tensor.data_ptr()
    canonical = json.dumps(geometry, sort_keys=True, separators=(",", ":")).encode()
    return {
        "request_ids": request_ids,
        "ordered_geometry": geometry,
        "ordered_geometry_sha256": hashlib.sha256(canonical).hexdigest(),
        "static_input_output_pointers": pointers,
    }
