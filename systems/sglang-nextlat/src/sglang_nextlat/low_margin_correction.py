"""Fail-closed helpers for canonical low-margin target correction.

The speculative worker may inspect a multi-row target-verify result, but it
must not use one of those rows as a substitute for an ordinary target decode.
When a would-be committed target row is numerically fragile, the caller uses
``canonical_decode_batch_view`` to construct the same one-token view that the
non-speculative scheduler presents to the target model.  The context manager
always restores the scheduler-owned ``ScheduleBatch`` fields before returning.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any, Iterator

import torch


CANONICAL_CORRECTION_LOGIT_GAP = 0.25


def accepted_path_margins(
    logits: torch.Tensor,
    accept_index: torch.Tensor,
    *,
    vocab_size: int,
) -> torch.Tensor:
    """Return target top-1/top-2 gaps for every would-be committed row.

    ``accept_index`` includes accepted drafts and the selected target bonus;
    negative entries are padding.  Rejected suffix rows are intentionally not
    part of the correction decision.
    """

    if logits.ndim != 2 or vocab_size < 2 or logits.shape[1] < vocab_size:
        raise ValueError("logits must have shape [rows, >= vocab_size >= 2]")
    if accept_index.ndim != 2 or accept_index.dtype not in (torch.int32, torch.int64):
        raise ValueError("accept_index must be a rank-two integer tensor")
    valid = accept_index >= 0
    rows = accept_index[valid].to(torch.int64)
    if rows.numel() == 0:
        raise ValueError("accepted path must contain at least one target row")
    if rows.device != logits.device:
        raise ValueError("accept_index and logits must share a device")
    torch._assert_async(torch.all(rows < logits.shape[0]), "accept_index is out of range")
    selected = logits.index_select(0, rows)[:, :vocab_size].float()
    top2 = torch.topk(selected, k=2, dim=-1).values
    return top2[:, 0] - top2[:, 1]


def requires_canonical_correction(
    logits: torch.Tensor,
    accept_index: torch.Tensor,
    *,
    vocab_size: int,
    threshold: float = CANONICAL_CORRECTION_LOGIT_GAP,
) -> tuple[bool, torch.Tensor]:
    """Synchronously decide whether the entire speculative batch must fall back."""

    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("correction threshold must be finite and nonnegative")
    margins = accepted_path_margins(
        logits,
        accept_index,
        vocab_size=vocab_size,
    )
    if not bool(torch.all(torch.isfinite(margins)).item()):
        raise RuntimeError("accepted-path margin is nonfinite")
    # A host decision is deliberate: no speculative KV commit may race ahead
    # of a fail-closed correction decision.
    required = bool(torch.any(margins <= threshold).item())
    return required, margins


def pad_canonical_tokens(tokens: torch.Tensor, *, verify_width: int) -> torch.Tensor:
    """Pack one canonical output per request into SGLang's spec-v2 stride."""

    if tokens.ndim != 1 or tokens.dtype not in (torch.int32, torch.int64):
        raise ValueError("canonical tokens must be a rank-one integer tensor")
    if verify_width < 1:
        raise ValueError("verify_width must be positive")
    packed = torch.zeros(
        (tokens.shape[0], verify_width),
        dtype=torch.int64,
        device=tokens.device,
    )
    packed[:, 0] = tokens.to(torch.int64)
    return packed.flatten()


@contextmanager
def canonical_decode_batch_view(
    batch: Any,
    *,
    anchor_tokens: torch.Tensor,
    canonical_out_cache_loc: torch.Tensor,
    decode_forward_mode: Any,
    no_spec_algorithm: Any,
) -> Iterator[None]:
    """Temporarily expose a canonical one-token decode view of ``batch``.

    At NextLat worker entry, ``batch.seq_lens`` is the committed prefix length
    ``L`` and the emitted anchor is not in target KV.  Ordinary SGLang decode
    has already advanced its view to ``L + 1`` and derives the anchor position
    as ``seq_lens - 1``.  This context mirrors that contract while leaving the
    scheduler-visible batch at ``L`` until the returned result commits one
    token.
    """

    required = (
        "forward_mode",
        "spec_algorithm",
        "spec_info",
        "input_ids",
        "input_embeds",
        "out_cache_loc",
        "seq_lens",
        "seq_lens_cpu",
        "seq_lens_sum",
        "orig_seq_lens",
    )
    missing = [name for name in required if not hasattr(batch, name)]
    if missing:
        raise ValueError(f"batch lacks canonical decode fields: {missing}")
    if batch.seq_lens_cpu is None:
        raise ValueError("canonical decode requires seq_lens_cpu")
    if not torch.is_tensor(batch.orig_seq_lens):
        raise ValueError("canonical decode requires tensor orig_seq_lens")
    if batch.orig_seq_lens.shape != batch.seq_lens.shape:
        raise ValueError("orig_seq_lens must match seq_lens")
    qwen_absent_rails = (
        "out_cache_loc_dsv4",
        "mamba_track_indices",
        "mamba_track_mask",
        "mamba_track_seqlens",
        "mamba_cow_src_indices",
        "mamba_cow_dst_indices",
        "mamba_clear_indices",
    )
    missing_rails = [name for name in qwen_absent_rails if not hasattr(batch, name)]
    if missing_rails:
        raise ValueError(f"batch lacks admitted-Qwen rail fields: {missing_rails}")
    active_rails = [name for name in qwen_absent_rails if getattr(batch, name) is not None]
    if active_rails:
        raise RuntimeError(
            f"canonical Qwen correction found unsupported DSV4/Mamba rails: {active_rails}"
        )
    if anchor_tokens.ndim != 1 or canonical_out_cache_loc.shape != anchor_tokens.shape:
        raise ValueError("anchor tokens and canonical cache locations must be [batch]")
    if anchor_tokens.device != batch.seq_lens.device:
        raise ValueError("anchor tokens and batch sequence lengths must share a device")
    if canonical_out_cache_loc.device != batch.seq_lens.device:
        raise ValueError("canonical cache locations must be on the batch device")

    snapshot = {name: getattr(batch, name) for name in required}
    try:
        batch.forward_mode = decode_forward_mode
        batch.spec_algorithm = no_spec_algorithm
        batch.spec_info = None
        batch.input_ids = anchor_tokens.to(torch.int64).contiguous()
        batch.input_embeds = None
        batch.out_cache_loc = canonical_out_cache_loc.to(torch.int64).contiguous()
        batch.seq_lens = snapshot["seq_lens"] + 1
        batch.seq_lens_cpu = snapshot["seq_lens_cpu"] + 1
        batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
        batch.orig_seq_lens = snapshot["orig_seq_lens"] + 1
        yield
    finally:
        for name, value in snapshot.items():
            setattr(batch, name, value)


@contextmanager
def force_eager_decode(model_runner: Any) -> Iterator[None]:
    """Explicitly disable graph admission for one synchronous decode call.

    ``ForwardBatch.can_run_decode_cuda_graph = False`` is not a general eager
    rail in the pinned SGLang runner: it is consulted only by some distributed
    graph configurations.  Removing the runner for the duration of the call is
    the smallest unambiguous rail.  NextLat admission forbids overlap, so no
    concurrent target forward can observe this temporary state.
    """

    if not hasattr(model_runner, "decode_cuda_graph_runner"):
        raise ValueError("model runner lacks decode_cuda_graph_runner")
    runner = model_runner.decode_cuda_graph_runner
    try:
        model_runner.decode_cuda_graph_runner = None
        yield
    finally:
        model_runner.decode_cuda_graph_runner = runner
