"""Exact B8/M8/global-tail4 native FA3 target-verify graph canary.

This is a one-geometry diagnostic derived from the independently qualified
compact-ragged graph mechanism.  The server remains configured for M=8 and a
dense ``[B, 8]`` draft carrier.  Only the host-known global-tail geometry
``proposal_lens=[4] * 8`` is admitted to a native 40-row target graph; every
other NEXTLAT geometry remains eager.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import torch

from .config import target_verify_graph_diagnostic_enabled


GRAPH_BATCH_SIZE = 8
CONFIGURED_MAX_DRAFTS = 8
CONFIGURED_VERIFY_WIDTH = 9
GRAPH_PROPOSAL_LEN = 4
GRAPH_VERIFY_WIDTH = 5
GRAPH_ROWS = GRAPH_BATCH_SIZE * GRAPH_VERIFY_WIDTH

_PATCHED = False


def _is_nextlat_model_runner(model_runner: Any) -> bool:
    algorithm = getattr(model_runner, "spec_algorithm", None)
    return getattr(algorithm, "name", None) == "NEXTLAT"


def _is_nextlat_runner(runner: Any) -> bool:
    return _is_nextlat_model_runner(getattr(runner, "model_runner", None))


def _is_exact_runtime_geometry(forward_batch: Any) -> bool:
    spec_info = getattr(forward_batch, "spec_info", None)
    layout = getattr(spec_info, "ragged_verify_layout", None)
    return bool(
        layout is not None
        and forward_batch.batch_size == GRAPH_BATCH_SIZE
        and layout.bs == GRAPH_BATCH_SIZE
        and layout.verify_lens_cpu == [GRAPH_VERIFY_WIDTH] * GRAPH_BATCH_SIZE
        and layout.total_verify_tokens == GRAPH_ROWS
        and layout.graph_num_tokens == GRAPH_ROWS
    )


def install_target_verify_graph_overlay() -> None:
    """Patch capture width and admission only for the registered M4 tier."""

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

    original_init = DecodeCudaGraphRunner.__init__
    original_get_spec_info = DecodeCudaGraphRunner.get_spec_info
    original_can_run_graph = DecodeCudaGraphRunner.can_run_graph

    def init(
        runner: Any,
        model_runner: Any,
        *,
        attn_backend=None,
        speculative_num_steps=None,
        speculative_num_draft_tokens=None,
    ) -> None:
        if _is_nextlat_model_runner(model_runner):
            if model_runner.is_draft_worker:
                raise RuntimeError("NEXTLAT M4 graph canary cannot run on a draft worker")
            if speculative_num_draft_tokens not in (None, CONFIGURED_VERIFY_WIDTH):
                raise RuntimeError(
                    "NEXTLAT M4 graph canary received a non-M8 configured verify width"
                )
            # The server and worker stay configured for M8.  This target runner
            # captures only the host-selected M4 verify tier: 8 * (4 + 1) rows.
            speculative_num_draft_tokens = GRAPH_VERIFY_WIDTH
        original_init(
            runner,
            model_runner,
            attn_backend=attn_backend,
            speculative_num_steps=speculative_num_steps,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
        )

    def get_spec_info(runner: Any, num_tokens: int):
        if not _is_nextlat_runner(runner):
            return original_get_spec_info(runner, num_tokens)
        if runner.model_runner.is_draft_worker:
            raise RuntimeError("NEXTLAT M4 graph canary cannot run on a draft worker")
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

    DecodeCudaGraphRunner.__init__ = init
    DecodeCudaGraphRunner.get_spec_info = get_spec_info
    DecodeCudaGraphRunner.can_run_graph = can_run_graph
    _PATCHED = True


def assert_captured_runner(runner: Any) -> dict[str, Any]:
    """Prove exact FULL-hidden-state, 40-row, one-bucket capture."""

    from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

    if runner is None or not _is_nextlat_runner(runner):
        raise RuntimeError("NEXTLAT M4 graph canary has no target graph runner")
    expected = {
        "ragged_verify_mode": True,
        "disable_padding": False,
        "captured_req_width": GRAPH_VERIFY_WIDTH,
        "capture_bs": [GRAPH_BATCH_SIZE],
        "capture_num_tokens": [GRAPH_ROWS],
        "max_bs": GRAPH_BATCH_SIZE,
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
            f"NEXTLAT M4 graph capture contract differs: {observed!r} != {expected!r}"
        )
    if runner.capture_hidden_mode != CaptureHiddenMode.FULL:
        raise RuntimeError("NEXTLAT M4 graph did not capture FULL hidden states")
    if not runner.attn_backend.supports_ragged_verify_graph:
        raise RuntimeError("NEXTLAT M4 graph backend lacks ragged-verify support")
    layout = runner._captured_ragged_layouts.get(GRAPH_ROWS)
    if (
        layout is None
        or layout.bs != GRAPH_BATCH_SIZE
        or layout.verify_lens_cpu != [GRAPH_VERIFY_WIDTH] * GRAPH_BATCH_SIZE
        or layout.total_verify_tokens != GRAPH_ROWS
        or layout.graph_num_tokens != GRAPH_ROWS
    ):
        raise RuntimeError("NEXTLAT M4 graph captured layout is not exact 8 x 5")
    output_buffer = getattr(runner.buffers, "next_token_logits_buffer", None)
    if not isinstance(output_buffer, torch.Tensor) or output_buffer.shape[0] < GRAPH_ROWS:
        raise RuntimeError("NEXTLAT M4 graph has no exact static logits output buffer")
    return {
        **observed,
        "configured_server_max_drafts": CONFIGURED_MAX_DRAFTS,
        "configured_server_verify_width": CONFIGURED_VERIFY_WIDTH,
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
    """Capture ordered 40-row geometry and stable native graph pointers."""

    if proposal_lens != [GRAPH_PROPOSAL_LEN] * GRAPH_BATCH_SIZE:
        raise RuntimeError("NEXTLAT M4 graph received non-tail4 proposal lengths")
    if verify_lens != [GRAPH_VERIFY_WIDTH] * GRAPH_BATCH_SIZE:
        raise RuntimeError("NEXTLAT M4 graph received non-5-row verify lengths")
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
        "proposal_lens": [GRAPH_PROPOSAL_LEN] * GRAPH_BATCH_SIZE,
        "verify_lens": [GRAPH_VERIFY_WIDTH] * GRAPH_BATCH_SIZE,
        "row_starts": list(range(0, GRAPH_ROWS, GRAPH_VERIFY_WIDTH)),
        "total_verify_rows": GRAPH_ROWS,
        "raw_bs": GRAPH_BATCH_SIZE,
        "padded_bs": GRAPH_BATCH_SIZE,
        "raw_rows": GRAPH_ROWS,
        "padded_rows": GRAPH_ROWS,
        "graph_key_size": GRAPH_ROWS,
        "capture_token_buckets": [GRAPH_ROWS],
    }
    if geometry != expected_geometry:
        raise RuntimeError(
            f"NEXTLAT M4 graph replay padded or drifted: "
            f"{geometry!r} != {expected_geometry!r}"
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
            raise RuntimeError(f"NEXTLAT M4 graph static buffer {name!r} is missing")
        pointers[name] = tensor.data_ptr()
    canonical = json.dumps(geometry, sort_keys=True, separators=(",", ":")).encode()
    return {
        "request_ids": request_ids,
        "ordered_geometry": geometry,
        "ordered_geometry_sha256": hashlib.sha256(canonical).hexdigest(),
        "static_input_output_pointers": pointers,
    }
