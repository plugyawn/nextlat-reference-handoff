"""One-shape native compact-ragged target-verify graph for B32/M4.

This is an opt-in performance cut.  Unlike the diagonal verifier, it preserves
SGLang's native ``TARGET_VERIFY`` attention layout and captures exactly 32
requests with five compact rows each.  Every other geometry stays inadmissible.
"""

from __future__ import annotations

import os
from typing import Any

from .constants import QWEN3_1_7B_CONTRACT


ENABLED_ENV = "NEXTLAT_NATIVE_RAGGED_VERIFY_GRAPH"
BATCH_SIZE = 32
DRAFT_TOKENS = 4
VERIFY_WIDTH = 5
VERIFY_ROWS = BATCH_SIZE * VERIFY_WIDTH

_PATCHED = False


def native_ragged_verify_enabled() -> bool:
    value = os.environ.get(ENABLED_ENV)
    if value is None:
        return False
    if value != "1":
        raise ValueError(f"{ENABLED_ENV} must be exactly '1'")
    return True


def is_exact_native_geometry(forward_batch: Any) -> bool:
    spec_info = getattr(forward_batch, "spec_info", None)
    layout = getattr(spec_info, "ragged_verify_layout", None)
    return bool(
        layout is not None
        and forward_batch.batch_size == BATCH_SIZE
        and layout.bs == BATCH_SIZE
        and layout.verify_lens_cpu == [VERIFY_WIDTH] * BATCH_SIZE
        and layout.total_verify_tokens == VERIFY_ROWS
        and layout.graph_num_tokens == VERIFY_ROWS
    )


def _is_nextlat_runner(runner: Any) -> bool:
    model_runner = getattr(runner, "model_runner", None)
    algorithm = getattr(model_runner, "spec_algorithm", None)
    return getattr(algorithm, "name", None) == "NEXTLAT"


def install_native_ragged_verify_graph_overlay() -> None:
    """Teach the pinned graph runner only the exact B32/M4 ragged shape."""

    global _PATCHED
    if not native_ragged_verify_enabled() or _PATCHED:
        return

    from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
        DecodeCudaGraphRunner,
    )
    from sglang.srt.speculative.dflash_info import DFlashVerifyInput
    from sglang.srt.speculative.dflash_utils import resolve_dflash_verify_mask_policy

    original_get_spec_info = DecodeCudaGraphRunner.get_spec_info
    original_can_run_graph = DecodeCudaGraphRunner.can_run_graph

    def get_spec_info(runner: Any, num_tokens: int):
        if not _is_nextlat_runner(runner):
            return original_get_spec_info(runner, num_tokens)
        if runner.model_runner.is_draft_worker:
            raise RuntimeError("NEXTLAT native graph cannot run on a draft worker")
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
        if _is_nextlat_runner(runner) and not is_exact_native_geometry(forward_batch):
            return False
        return original_can_run_graph(runner, forward_batch)

    DecodeCudaGraphRunner.get_spec_info = get_spec_info
    DecodeCudaGraphRunner.can_run_graph = can_run_graph
    _PATCHED = True


def assert_native_captured_runner(runner: Any) -> dict[str, Any]:
    from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

    if runner is None or not _is_nextlat_runner(runner):
        raise RuntimeError("NEXTLAT native graph has no target graph runner")
    observed = {
        "ragged_verify_mode": runner.ragged_verify_mode,
        "disable_padding": runner.disable_padding,
        "captured_req_width": runner.captured_req_width,
        "capture_bs": list(runner.capture_bs),
        "capture_num_tokens": list(runner.capture_num_tokens or []),
        "max_bs": runner.max_bs,
    }
    expected = {
        "ragged_verify_mode": True,
        "disable_padding": False,
        "captured_req_width": VERIFY_WIDTH,
        "capture_bs": [BATCH_SIZE],
        "capture_num_tokens": [VERIFY_ROWS],
        "max_bs": BATCH_SIZE,
    }
    if observed != expected:
        raise RuntimeError(
            f"NEXTLAT native graph capture differs: {observed!r} != {expected!r}"
        )
    if runner.capture_hidden_mode != CaptureHiddenMode.FULL:
        raise RuntimeError("NEXTLAT native graph did not capture full hidden states")
    if not runner.attn_backend.supports_ragged_verify_graph:
        raise RuntimeError("NEXTLAT native backend lacks ragged-verify graphs")
    layout = runner._captured_ragged_layouts.get(VERIFY_ROWS)
    if (
        layout is None
        or layout.verify_lens_cpu != [VERIFY_WIDTH] * BATCH_SIZE
        or layout.total_verify_tokens != VERIFY_ROWS
    ):
        raise RuntimeError("NEXTLAT native graph captured the wrong ragged layout")
    return {
        **observed,
        "capture_hidden_mode": runner.capture_hidden_mode.name,
        "hidden_size": QWEN3_1_7B_CONTRACT.hidden_size,
    }


def native_replay_snapshot(runner: Any) -> dict[str, Any]:
    key = runner._replay_graph_key
    return {
        "native_ragged_verify": True,
        "raw_bs": runner.raw_bs,
        "padded_bs": runner.bs,
        "raw_rows": runner.raw_num_token,
        "padded_rows": runner._ragged_graph_size,
        "graph_key_size": getattr(key, "size", None),
    }
