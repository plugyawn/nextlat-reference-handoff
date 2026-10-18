"""Global-tail recurrent proposal semantics for one bounded diagnostic.

This module does not compact rows or choose a learned variable length.  It
only shortens the recurrent loop to the host-known maximum proposal length in
the current batch, then restores the configured dense ``[B, M]`` token shape
required by SGLang's existing ragged verifier.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

from .constants import MAX_DRAFT_TOKENS
from .decoder import NextLatProposal, propose_greedy


def admit_global_tail_steps(
    proposal_lens_cpu: list[int], *, batch_size: int, configured_max_drafts: int
) -> int:
    """Validate host proposal lengths and return their exact global maximum."""

    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer")
    if (
        isinstance(configured_max_drafts, bool)
        or not isinstance(configured_max_drafts, int)
        or not 1 <= configured_max_drafts <= MAX_DRAFT_TOKENS
    ):
        raise ValueError(
            f"configured_max_drafts must be in [1, {MAX_DRAFT_TOKENS}]"
        )
    if not isinstance(proposal_lens_cpu, list) or len(proposal_lens_cpu) != batch_size:
        raise ValueError("proposal_lens_cpu must be one integer per batch row")
    for row, value in enumerate(proposal_lens_cpu):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= configured_max_drafts
        ):
            raise ValueError(
                f"proposal_lens_cpu[{row}] must be an integer in "
                f"[0, {configured_max_drafts}]"
            )
    return max(proposal_lens_cpu)


def propose_greedy_global_tail(
    *,
    current_states: torch.Tensor,
    anchor_tokens: torch.Tensor,
    proposal_lens: torch.Tensor,
    proposal_lens_cpu: list[int],
    dynamics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    embed_tokens: Callable[[torch.Tensor], torch.Tensor],
    project_logits: Callable[[torch.Tensor], torch.Tensor],
    vocab_size: int,
    configured_max_drafts: int,
    executed_recurrent_steps: int,
    phase_observer: Callable[[str, int], None] | None = None,
) -> NextLatProposal:
    """Execute a strict global tail shorter than ``M`` and restore ``[B, M]``.

    The literal full-width route remains owned by the worker and must call
    :func:`decoder.propose_greedy` directly.  Calling this helper when the
    global maximum equals ``M`` is therefore rejected rather than silently
    introducing a second full-width implementation.
    """

    batch_size = int(anchor_tokens.shape[0]) if anchor_tokens.ndim == 1 else -1
    admitted_steps = admit_global_tail_steps(
        proposal_lens_cpu,
        batch_size=batch_size,
        configured_max_drafts=configured_max_drafts,
    )
    if (
        isinstance(executed_recurrent_steps, bool)
        or not isinstance(executed_recurrent_steps, int)
        or executed_recurrent_steps != admitted_steps
    ):
        raise ValueError("executed_recurrent_steps differs from the admitted host maximum")
    if executed_recurrent_steps >= configured_max_drafts:
        raise ValueError("full-width geometry must use literal propose_greedy")
    if proposal_lens.ndim != 1 or proposal_lens.shape != anchor_tokens.shape:
        raise ValueError("proposal_lens must have shape [batch]")
    if proposal_lens.dtype not in (torch.int32, torch.int64):
        raise ValueError("proposal_lens must be an integer tensor")
    if proposal_lens.device != anchor_tokens.device:
        raise ValueError("proposal_lens and anchors must share a device")
    if proposal_lens.device.type == "cpu" and proposal_lens.tolist() != proposal_lens_cpu:
        raise ValueError("proposal_lens tensor differs from proposal_lens_cpu")

    if executed_recurrent_steps == 0:
        padded_tokens = torch.zeros(
            (batch_size, configured_max_drafts),
            dtype=torch.int64,
            device=anchor_tokens.device,
        )
    else:
        short = propose_greedy(
            current_states=current_states,
            anchor_tokens=anchor_tokens,
            proposal_lens=proposal_lens,
            dynamics=dynamics,
            embed_tokens=embed_tokens,
            project_logits=project_logits,
            vocab_size=vocab_size,
            max_draft_tokens=executed_recurrent_steps,
            phase_observer=phase_observer,
        )
        padded_tokens = F.pad(
            short.draft_tokens,
            (0, configured_max_drafts - executed_recurrent_steps),
            mode="constant",
            value=0,
        )

    result = NextLatProposal(
        anchor_tokens=anchor_tokens.to(torch.int64),
        draft_tokens=padded_tokens,
        proposal_lens=proposal_lens,
    )
    result.validate()
    return result
