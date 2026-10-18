"""Native one-shape CUDA graph for the B1/N14 NextLat verify tree.

The top-2 recurrent experiment produces an irregular 14-node tree.  SGLang's
FA3 backend already has the required native ``TARGET_VERIFY`` path: capture
uses an EAGLE-shaped placeholder, while replay prepares the live positions and
tree mask out of graph.  This overlay only exposes that existing path to the
custom NEXTLAT identity for one fixed shape.  It never aliases NEXTLAT to
EAGLE and it never converts the tree to diagonal ordinary decode.

The H8/N14 full tree is the only graph-admitted runtime geometry.  Short final
tails remain eager so the capture-time tree depth cannot silently describe a
different acceptance layout.
"""

from __future__ import annotations

from typing import Any

import torch

from .beam_tree import TREE_HORIZON, TREE_NODE_BUDGET
from .config import top2_tree_verify_enabled
from .constants import QWEN3_1_7B_CONTRACT


TREE_GRAPH_BATCH_SIZE = 1

_PATCHED = False


def _name(value: Any) -> str | None:
    return getattr(value, "name", None)


def _is_nextlat_runner(runner: Any) -> bool:
    model_runner = getattr(runner, "model_runner", None)
    return _name(getattr(model_runner, "spec_algorithm", None)) == "NEXTLAT"


def is_exact_tree_verify_geometry(forward_batch: Any) -> bool:
    """Admit only the full B1/H8/N14 irregular-tree target forward."""

    spec_info = getattr(forward_batch, "spec_info", None)
    tensors = (
        getattr(forward_batch, "input_ids", None),
        getattr(forward_batch, "positions", None),
        getattr(forward_batch, "out_cache_loc", None),
        getattr(spec_info, "draft_token", None),
    )
    retrieval = (
        getattr(spec_info, "retrieve_index", None),
        getattr(spec_info, "retrieve_next_token", None),
        getattr(spec_info, "retrieve_next_sibling", None),
    )
    custom_mask = getattr(spec_info, "custom_mask", None)
    seq_lens_sum = getattr(spec_info, "seq_lens_sum", None)
    return bool(
        _name(getattr(forward_batch, "forward_mode", None)) == "TARGET_VERIFY"
        and getattr(forward_batch, "batch_size", None) == TREE_GRAPH_BATCH_SIZE
        and all(
            isinstance(tensor, torch.Tensor)
            and tensor.ndim == 1
            and tensor.numel() == TREE_NODE_BUDGET
            for tensor in tensors
        )
        and all(
            isinstance(tensor, torch.Tensor)
            and tuple(tensor.shape) == (TREE_GRAPH_BATCH_SIZE, TREE_NODE_BUDGET)
            for tensor in retrieval
        )
        and isinstance(custom_mask, torch.Tensor)
        and custom_mask.dtype == torch.bool
        and custom_mask.ndim == 1
        and isinstance(seq_lens_sum, int)
        and seq_lens_sum > 0
        and custom_mask.numel()
        == TREE_NODE_BUDGET * (seq_lens_sum + TREE_NODE_BUDGET)
        and getattr(spec_info, "draft_token_num", None) == TREE_NODE_BUDGET
        and getattr(spec_info, "num_tokens_per_req", None) == TREE_NODE_BUDGET
        and getattr(spec_info, "spec_steps", None) == TREE_HORIZON
        # Runtime uses the generic linked-tree traversal for this irregular
        # topology.  The capture placeholder uses configured topk=2 instead.
        and getattr(spec_info, "topk", None) == -1
        and _name(getattr(spec_info, "capture_hidden_mode", None)) == "FULL"
        and _name(getattr(forward_batch, "capture_hidden_mode", None)) == "FULL"
        and getattr(forward_batch, "return_logprob", False) is False
    )


def install_top2_tree_verify_graph_overlay() -> None:
    """Teach the pinned decode graph runner exactly one NEXTLAT tree shape."""

    global _PATCHED
    if not top2_tree_verify_enabled() or _PATCHED:
        return

    from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
        DecodeCudaGraphRunner,
    )
    from sglang.srt.speculative.eagle_info import EagleVerifyInput

    original_get_spec_info = DecodeCudaGraphRunner.get_spec_info
    original_can_run_graph = DecodeCudaGraphRunner.can_run_graph

    def get_spec_info(runner: Any, num_tokens: int):
        if not _is_nextlat_runner(runner):
            return original_get_spec_info(runner, num_tokens)
        if runner.model_runner.is_draft_worker:
            raise RuntimeError("NEXTLAT tree graph cannot capture a draft worker")
        if num_tokens != TREE_NODE_BUDGET:
            raise RuntimeError(
                "NEXTLAT tree graph capture requires exactly 14 target rows"
            )
        return EagleVerifyInput(
            draft_token=None,
            custom_mask=runner.buffers.custom_mask,
            positions=None,
            retrieve_index=None,
            retrieve_next_token=None,
            retrieve_next_sibling=None,
            retrieve_cum_len=None,
            spec_steps=TREE_HORIZON,
            topk=2,
            draft_token_num=TREE_NODE_BUDGET,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            seq_lens_sum=None,
            seq_lens_cpu=None,
        )

    def can_run_graph(runner: Any, forward_batch: Any):
        if not _is_nextlat_runner(runner):
            return original_can_run_graph(runner, forward_batch)
        if not is_exact_tree_verify_geometry(forward_batch):
            return False
        admitted = bool(original_can_run_graph(runner, forward_batch))
        if admitted:
            runner._nextlat_tree_graph_admission_count = (
                getattr(runner, "_nextlat_tree_graph_admission_count", 0) + 1
            )
        return admitted

    DecodeCudaGraphRunner.get_spec_info = get_spec_info
    DecodeCudaGraphRunner.can_run_graph = can_run_graph
    _PATCHED = True


def assert_tree_captured_runner(runner: Any) -> dict[str, Any]:
    """Fail startup unless capture is native B1/N14 TARGET_VERIFY + FULL."""

    from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

    if runner is None or not _is_nextlat_runner(runner):
        raise RuntimeError("NEXTLAT tree graph has no target graph runner")
    observed = {
        "capture_forward_mode": _name(runner.capture_forward_mode),
        "ragged_verify_mode": runner.ragged_verify_mode,
        "captured_req_width": runner.captured_req_width,
        "capture_bs": list(runner.capture_bs),
        "capture_num_tokens": runner.capture_num_tokens,
        "max_bs": runner.max_bs,
        "attention_topk": getattr(runner.attn_backend, "topk", None),
    }
    expected = {
        "capture_forward_mode": "TARGET_VERIFY",
        "ragged_verify_mode": False,
        "captured_req_width": TREE_NODE_BUDGET,
        "capture_bs": [TREE_GRAPH_BATCH_SIZE],
        "capture_num_tokens": None,
        "max_bs": TREE_GRAPH_BATCH_SIZE,
        "attention_topk": 2,
    }
    if observed != expected:
        raise RuntimeError(
            f"NEXTLAT tree graph capture differs: {observed!r} != {expected!r}"
        )
    if runner.capture_hidden_mode != CaptureHiddenMode.FULL:
        raise RuntimeError("NEXTLAT tree graph did not capture FULL hidden states")
    custom_mask = getattr(runner.buffers, "custom_mask", None)
    if (
        not isinstance(custom_mask, torch.Tensor)
        or custom_mask.dtype != torch.bool
        or custom_mask.numel() < TREE_NODE_BUDGET * TREE_NODE_BUDGET
    ):
        raise RuntimeError("NEXTLAT tree graph lacks its static custom-mask buffer")
    logits = getattr(runner.buffers, "next_token_logits_buffer", None)
    if not isinstance(logits, torch.Tensor) or logits.shape[0] < TREE_NODE_BUDGET:
        raise RuntimeError("NEXTLAT tree graph lacks its static logits buffer")
    runner._nextlat_tree_graph_admission_count = 0
    return {
        **observed,
        "capture_hidden_mode": runner.capture_hidden_mode.name,
        "hidden_size": QWEN3_1_7B_CONTRACT.hidden_size,
        "custom_mask_capacity": custom_mask.numel(),
    }


def tree_replay_snapshot(runner: Any, forward_batch: Any) -> dict[str, Any]:
    """Return a compact executed-path check for one admitted replay."""

    if not is_exact_tree_verify_geometry(forward_batch):
        raise RuntimeError("NEXTLAT tree graph replay received a non-H8/N14 shape")
    observed = {
        "raw_bs": runner.raw_bs,
        "padded_bs": runner.bs,
        "raw_rows": runner.raw_num_token,
        "graph_key_size": getattr(runner._replay_graph_key, "size", None),
    }
    expected = {
        "raw_bs": TREE_GRAPH_BATCH_SIZE,
        "padded_bs": TREE_GRAPH_BATCH_SIZE,
        "raw_rows": TREE_NODE_BUDGET,
        # Uniform target-verify graphs are keyed by request batch size.
        "graph_key_size": TREE_GRAPH_BATCH_SIZE,
    }
    if observed != expected:
        raise RuntimeError(
            f"NEXTLAT tree graph replay padded or drifted: {observed!r} != {expected!r}"
        )
    admissions = getattr(runner, "_nextlat_tree_graph_admission_count", None)
    if not isinstance(admissions, int) or admissions <= 0:
        raise RuntimeError("NEXTLAT tree graph recorded no admitted replay")
    return {
        **observed,
        "tree_graph_admission_count": admissions,
        "draft_token_num": TREE_NODE_BUDGET,
        "spec_steps": TREE_HORIZON,
    }
