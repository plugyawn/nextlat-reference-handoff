"""Framework-independent NextLat proposal and verification semantics.

The serving adapter deliberately delegates cache allocation and cache movement to
SGLang.  This module owns only the small, testable tensor contract between the
recurrent proposal and the target verifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from .constants import MAX_DRAFT_TOKENS


def _require_tensor_true(condition: torch.Tensor, message: str) -> None:
    """Fail immediately on CPU and without a device-to-host sync on CUDA."""

    if condition.device.type == "cpu":
        if not bool(condition.item()):
            raise ValueError(message)
    else:
        torch._assert_async(condition, message)


@dataclass(frozen=True)
class NextLatProposal:
    """One anchor and a configured maximum recurrent window per request."""

    anchor_tokens: torch.Tensor
    draft_tokens: torch.Tensor
    proposal_lens: torch.Tensor
    # Per-step probabilities over the reduced draft vocabulary.  Greedy
    # proposals leave this unset; sampled proposals retain it for exact
    # speculative rejection sampling against the target distribution.
    q_hot: torch.Tensor | None = None

    def validate(self) -> None:
        if self.anchor_tokens.ndim != 1:
            raise ValueError("anchor_tokens must have shape [batch]")
        if self.draft_tokens.ndim != 2:
            raise ValueError("draft_tokens must have shape [batch, max_drafts]")
        if not 1 <= self.draft_tokens.shape[1] <= MAX_DRAFT_TOKENS:
            raise ValueError(
                "draft_tokens max_drafts dimension must be in "
                f"[1, {MAX_DRAFT_TOKENS}]"
            )
        if self.proposal_lens.ndim != 1:
            raise ValueError("proposal_lens must have shape [batch]")
        batch_size = self.anchor_tokens.shape[0]
        if self.draft_tokens.shape[0] != batch_size:
            raise ValueError("draft_tokens batch dimension differs from anchors")
        if self.proposal_lens.shape[0] != batch_size:
            raise ValueError("proposal_lens batch dimension differs from anchors")
        if self.anchor_tokens.dtype not in (torch.int32, torch.int64):
            raise ValueError("anchor_tokens must contain integer token ids")
        if self.draft_tokens.dtype not in (torch.int32, torch.int64):
            raise ValueError("draft_tokens must contain integer token ids")
        if self.proposal_lens.dtype not in (torch.int32, torch.int64):
            raise ValueError("proposal_lens must be an integer tensor")
        if self.anchor_tokens.device != self.draft_tokens.device:
            raise ValueError("anchor_tokens and draft_tokens must share a device")
        if self.anchor_tokens.device != self.proposal_lens.device:
            raise ValueError("proposal_lens must be on the proposal device")
        if self.q_hot is not None:
            expected = (
                batch_size,
                self.max_draft_tokens,
                int(self.q_hot.shape[-1]) if self.q_hot.ndim == 3 else -1,
            )
            if self.q_hot.ndim != 3 or tuple(self.q_hot.shape) != expected:
                raise ValueError(
                    "q_hot must have shape [batch, max_drafts, draft_vocab]"
                )
            if not self.q_hot.is_floating_point():
                raise ValueError("q_hot must contain floating-point probabilities")
            if self.q_hot.device != self.anchor_tokens.device:
                raise ValueError("q_hot must be on the proposal device")
            _require_tensor_true(
                torch.all(torch.isfinite(self.q_hot)),
                "q_hot must contain finite probabilities",
            )
            _require_tensor_true(
                torch.all(self.q_hot >= 0),
                "q_hot probabilities must be non-negative",
            )
        _require_tensor_true(
            torch.all(
                (self.proposal_lens >= 0)
                & (self.proposal_lens <= self.max_draft_tokens)
            ),
            f"proposal_lens must be in [0, {self.max_draft_tokens}]",
        )

    @property
    def max_draft_tokens(self) -> int:
        return int(self.draft_tokens.shape[1])

    @property
    def verify_width(self) -> int:
        return self.max_draft_tokens + 1

    @property
    def verify_lens(self) -> torch.Tensor:
        """Target rows per request, including the mandatory anchor row."""

        return self.proposal_lens + 1

    @property
    def candidates(self) -> torch.Tensor:
        """Dense ``[anchor, draft_1, ..., draft_3]`` representation."""

        return torch.cat(
            (self.anchor_tokens[:, None], self.draft_tokens.to(torch.int64)), dim=1
        )

    @property
    def dspark_draft_block_ids(self) -> torch.Tensor:
        """DSpark ragged-window anchor carrier with row stride ``max_drafts``.

        ``BuildRaggedVerifyWindow`` takes the proposed tokens separately.  Its
        Triton gather reads only column zero from ``draft_block_ids`` and
        infers that tensor's row stride from ``draft_tokens.shape[1]``.  Keep
        this carrier at ``[batch, max_drafts]``; passing ``candidates`` here
        would shift every request after row zero by one token.
        """

        self.validate()
        return torch.cat(
            (
                self.anchor_tokens.to(torch.int64)[:, None],
                self.draft_tokens.to(torch.int64)[:, :-1],
            ),
            dim=1,
        ).contiguous()

    def compact_verify_tokens(self) -> torch.Tensor:
        """Compact variable-length rows in request order."""

        self.validate()
        columns = torch.arange(self.verify_width, device=self.anchor_tokens.device)
        valid = columns[None, :] < self.verify_lens[:, None]
        return self.candidates[valid]


@dataclass(frozen=True)
class NextLatVerification:
    """Commit/output plan produced by exact target verification."""

    num_correct_drafts: torch.Tensor
    accept_lens: torch.Tensor
    accept_index: torch.Tensor
    output_tokens: torch.Tensor
    bonus_tokens: torch.Tensor
    new_seq_lens: torch.Tensor
    selected_hidden_indices: torch.Tensor


def propose_greedy(
    *,
    current_states: torch.Tensor,
    anchor_tokens: torch.Tensor,
    proposal_lens: torch.Tensor,
    dynamics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    embed_tokens: Callable[[torch.Tensor], torch.Tensor],
    project_logits: Callable[[torch.Tensor], torch.Tensor],
    vocab_size: int,
    max_draft_tokens: int,
    token_id_map: torch.Tensor | None = None,
    phase_observer: Callable[[str, int], None] | None = None,
) -> NextLatProposal:
    """Run a configured, shape-stable recurrent top-1 proposal window.

    ``current_states`` is the exact target state at the last committed KV token.
    The anchor was emitted by the previous target step but is not in target KV
    yet, so the first recurrence consumes its embedding.
    """

    if current_states.ndim != 2:
        raise ValueError("current_states must have shape [batch, hidden]")
    if anchor_tokens.ndim != 1 or anchor_tokens.shape[0] != current_states.shape[0]:
        raise ValueError("anchor_tokens must have shape [batch]")
    if proposal_lens.shape != anchor_tokens.shape:
        raise ValueError("proposal_lens must have shape [batch]")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if token_id_map is not None:
        if (
            token_id_map.ndim != 1
            or token_id_map.shape[0] != vocab_size
            or token_id_map.dtype != torch.int64
            or token_id_map.device != current_states.device
        ):
            raise ValueError(
                "token_id_map must be an int64 [vocab_size] tensor on the state device"
            )
    if not 1 <= max_draft_tokens <= MAX_DRAFT_TOKENS:
        raise ValueError(
            f"max_draft_tokens must be in [1, {MAX_DRAFT_TOKENS}]"
        )

    state = current_states
    input_tokens = anchor_tokens.to(torch.int64)
    proposals: list[torch.Tensor] = []
    for step in range(max_draft_tokens):
        if phase_observer is not None:
            phase_observer("embedding_start", step)
        embeddings = embed_tokens(input_tokens)
        if phase_observer is not None:
            phase_observer("embedding_end", step)
        if embeddings.shape != state.shape:
            raise ValueError(
                "target embeddings and recurrent states must have identical shapes"
            )
        if phase_observer is not None:
            phase_observer("dynamics_start", step)
        state = dynamics(state, embeddings)
        if phase_observer is not None:
            phase_observer("dynamics_end", step)
            phase_observer("vocab_head_start", step)
        logits = project_logits(state)
        if logits.ndim != 2 or logits.shape[0] != state.shape[0]:
            raise ValueError("project_logits must return [batch, vocab] logits")
        if logits.shape[1] < vocab_size:
            raise ValueError(
                f"projected vocabulary has width {logits.shape[1]}, expected at "
                f"least {vocab_size}"
            )
        reduced_token_ids = torch.argmax(logits[:, :vocab_size], dim=-1).to(
            torch.int64
        )
        input_tokens = (
            reduced_token_ids
            if token_id_map is None
            else token_id_map[reduced_token_ids]
        )
        if phase_observer is not None:
            phase_observer("vocab_head_end", step)
        proposals.append(input_tokens)

    proposal = NextLatProposal(
        anchor_tokens=anchor_tokens.to(torch.int64),
        draft_tokens=torch.stack(proposals, dim=1),
        proposal_lens=proposal_lens,
    )
    proposal.validate()
    return proposal


def propose_temperature_sampled(
    *,
    current_states: torch.Tensor,
    anchor_tokens: torch.Tensor,
    proposal_lens: torch.Tensor,
    temperatures: torch.Tensor,
    dynamics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    embed_tokens: Callable[[torch.Tensor], torch.Tensor],
    project_logits: Callable[[torch.Tensor], torch.Tensor],
    vocab_size: int,
    max_draft_tokens: int,
    token_id_map: torch.Tensor,
    sample_probabilities: Callable[[torch.Tensor], torch.Tensor] | None = None,
    phase_observer: Callable[[str, int], None] | None = None,
) -> NextLatProposal:
    """Sample a recurrent proposal and retain every proposal probability.

    This is deliberately the smallest lossless sampled tier: temperature is
    the only transform and every draw is from the reduced hot vocabulary.
    ``q_hot`` is retained so the verifier can scatter it into the target
    vocabulary before invoking SGLang's rejection-sampling kernel.
    """

    if current_states.ndim != 2:
        raise ValueError("current_states must have shape [batch, hidden]")
    batch_size = current_states.shape[0]
    if anchor_tokens.ndim != 1 or anchor_tokens.shape[0] != batch_size:
        raise ValueError("anchor_tokens must have shape [batch]")
    if proposal_lens.shape != anchor_tokens.shape:
        raise ValueError("proposal_lens must have shape [batch]")
    if temperatures.ndim == 2 and temperatures.shape == (batch_size, 1):
        temperatures = temperatures[:, 0]
    if temperatures.shape != anchor_tokens.shape:
        raise ValueError("temperatures must have shape [batch] or [batch, 1]")
    if not temperatures.is_floating_point():
        raise ValueError("temperatures must be floating point")
    if temperatures.device != current_states.device:
        raise ValueError("temperatures must be on the state device")
    _require_tensor_true(
        torch.all(torch.isfinite(temperatures) & (temperatures > 0)),
        "temperatures must be finite and strictly positive",
    )
    if not 1 <= max_draft_tokens <= MAX_DRAFT_TOKENS:
        raise ValueError(
            f"max_draft_tokens must be in [1, {MAX_DRAFT_TOKENS}]"
        )
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if (
        token_id_map.ndim != 1
        or token_id_map.shape[0] != vocab_size
        or token_id_map.dtype != torch.int64
        or token_id_map.device != current_states.device
    ):
        raise ValueError(
            "token_id_map must be an int64 [vocab_size] tensor on the state device"
        )

    state = current_states
    input_tokens = anchor_tokens.to(torch.int64)
    proposals: list[torch.Tensor] = []
    proposal_probs: list[torch.Tensor] = []
    for step in range(max_draft_tokens):
        if phase_observer is not None:
            phase_observer("embedding_start", step)
        embeddings = embed_tokens(input_tokens)
        if phase_observer is not None:
            phase_observer("embedding_end", step)
        if embeddings.shape != state.shape:
            raise ValueError(
                "target embeddings and recurrent states must have identical shapes"
            )
        if phase_observer is not None:
            phase_observer("dynamics_start", step)
        state = dynamics(state, embeddings)
        if phase_observer is not None:
            phase_observer("dynamics_end", step)
            phase_observer("vocab_head_start", step)
        logits = project_logits(state)
        if logits.ndim != 2 or logits.shape[0] != batch_size:
            raise ValueError("project_logits must return [batch, vocab] logits")
        if logits.shape[1] < vocab_size:
            raise ValueError(
                f"projected vocabulary has width {logits.shape[1]}, expected at "
                f"least {vocab_size}"
            )
        q_hot = torch.softmax(
            logits[:, :vocab_size].float() / temperatures[:, None].float(), dim=-1
        )
        if sample_probabilities is None:
            reduced_token_ids = torch.multinomial(q_hot, num_samples=1).squeeze(1)
        else:
            reduced_token_ids = sample_probabilities(q_hot)
        if reduced_token_ids.shape != (batch_size,) or reduced_token_ids.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError(
                "sample_probabilities must return integer token ids with shape [batch]"
            )
        reduced_token_ids = reduced_token_ids.to(torch.int64)
        _require_tensor_true(
            torch.all((reduced_token_ids >= 0) & (reduced_token_ids < vocab_size)),
            "sampled reduced token id is outside the draft vocabulary",
        )
        input_tokens = token_id_map[reduced_token_ids]
        if phase_observer is not None:
            phase_observer("vocab_head_end", step)
        proposals.append(input_tokens)
        proposal_probs.append(q_hot)

    proposal = NextLatProposal(
        anchor_tokens=anchor_tokens.to(torch.int64),
        draft_tokens=torch.stack(proposals, dim=1),
        proposal_lens=proposal_lens,
        q_hot=torch.stack(proposal_probs, dim=1),
    )
    proposal.validate()
    return proposal


def scatter_hot_probabilities(
    q_hot: torch.Tensor,
    *,
    token_id_map: torch.Tensor,
    target_vocab_size: int,
) -> torch.Tensor:
    """Scatter reduced-vocabulary draft probabilities into target-token space."""

    if q_hot.ndim != 3 or not q_hot.is_floating_point():
        raise ValueError("q_hot must be floating point with shape [batch, steps, hot]")
    if target_vocab_size <= 0:
        raise ValueError("target_vocab_size must be positive")
    if (
        token_id_map.ndim != 1
        or token_id_map.shape[0] != q_hot.shape[2]
        or token_id_map.dtype != torch.int64
        or token_id_map.device != q_hot.device
    ):
        raise ValueError(
            "token_id_map must be int64 [hot_vocab] on the q_hot device"
        )
    _require_tensor_true(
        torch.all((token_id_map >= 0) & (token_id_map < target_vocab_size)),
        "token_id_map contains a target-vocabulary id out of range",
    )
    full = torch.zeros(
        (*q_hot.shape[:2], target_vocab_size),
        dtype=q_hot.dtype,
        device=q_hot.device,
    )
    full.index_copy_(2, token_id_map, q_hot)
    return full


def mask_embedding_only_logits(
    logits: torch.Tensor, *, tokenizer_vocab_size: int
) -> torch.Tensor:
    """Return logits with embedding-padding IDs removed from sampling support."""

    if logits.ndim != 2:
        raise ValueError("logits must have shape [rows, head_vocab]")
    if not 0 < tokenizer_vocab_size <= logits.shape[1]:
        raise ValueError("tokenizer_vocab_size is outside the LM-head width")
    masked = logits.clone(memory_format=torch.contiguous_format)
    masked[:, tokenizer_vocab_size:] = -torch.inf
    return masked


def finish_sampled_verification(
    *,
    proposal: NextLatProposal,
    predicts: torch.Tensor,
    kernel_accept_index: torch.Tensor,
    num_correct_drafts: torch.Tensor,
    prefix_seq_lens: torch.Tensor,
    active_drafts: int,
) -> NextLatVerification:
    """Adapt SGLang's B1 chain-rejection output to the NextLat commit contract."""

    proposal.validate()
    if proposal.anchor_tokens.shape[0] != 1:
        raise ValueError("sampled verification currently supports batch size one")
    if not 0 <= active_drafts <= proposal.max_draft_tokens:
        raise ValueError("active_drafts is outside the configured proposal window")
    active_width = active_drafts + 1
    if predicts.ndim == 2 and predicts.shape == (1, active_width):
        predicts = predicts[0]
    if predicts.shape != (active_width,) or predicts.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("predicts must contain active_width integer tokens")
    if kernel_accept_index.shape != (1, active_width):
        raise ValueError("kernel_accept_index must have shape [1, active_width]")
    if kernel_accept_index.dtype not in (torch.int32, torch.int64):
        raise ValueError("kernel_accept_index must be an integer tensor")
    if num_correct_drafts.shape != (1,) or num_correct_drafts.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("num_correct_drafts must be an integer [1] tensor")
    if prefix_seq_lens.shape != (1,):
        raise ValueError("prefix_seq_lens must have shape [1]")
    device = proposal.anchor_tokens.device
    for tensor in (
        predicts,
        kernel_accept_index,
        num_correct_drafts,
        prefix_seq_lens,
    ):
        if tensor.device != device:
            raise ValueError("sampled verification tensors must share a device")
    _require_tensor_true(
        torch.all(
            (num_correct_drafts >= 0) & (num_correct_drafts <= active_drafts)
        ),
        "kernel accepted-draft count is outside the active proposal",
    )

    accepted = num_correct_drafts.to(torch.int64)
    selected_hidden_indices = kernel_accept_index.gather(1, accepted[:, None]).squeeze(
        1
    ).to(torch.int64)
    _require_tensor_true(
        torch.all(
            (selected_hidden_indices >= 0)
            & (selected_hidden_indices < active_width)
        ),
        "kernel selected an invalid target row",
    )
    bonus_tokens = predicts[selected_hidden_indices].to(torch.int64)
    accept_lens = num_correct_drafts.to(torch.int32) + 1

    width = proposal.verify_width
    columns = torch.arange(width, device=device)
    accepted_draft = columns[None, :] < accepted[:, None]
    bonus_column = columns[None, :] == accepted[:, None]
    padded_drafts = torch.nn.functional.pad(
        proposal.draft_tokens.to(torch.int64), (0, 1)
    )
    output_tokens = torch.where(accepted_draft, padded_drafts, 0)
    output_tokens = torch.where(bonus_column, bonus_tokens[:, None], output_tokens)

    active_columns = torch.arange(active_width, device=device)
    active_commit = active_columns[None, :] <= accepted[:, None]
    active_accept_index = torch.where(
        active_commit,
        kernel_accept_index.to(torch.int64),
        torch.full_like(kernel_accept_index, -1, dtype=torch.int64),
    )
    accept_index = torch.full((1, width), -1, dtype=torch.int64, device=device)
    accept_index[:, :active_width] = active_accept_index
    new_seq_lens = prefix_seq_lens + accept_lens.to(prefix_seq_lens.dtype)

    return NextLatVerification(
        num_correct_drafts=num_correct_drafts.to(torch.int32),
        accept_lens=accept_lens,
        accept_index=accept_index,
        output_tokens=output_tokens,
        bonus_tokens=bonus_tokens,
        new_seq_lens=new_seq_lens,
        selected_hidden_indices=selected_hidden_indices,
    )


def verify_greedy(
    *,
    proposal: NextLatProposal,
    compact_target_token_ids: torch.Tensor,
    prefix_seq_lens: torch.Tensor,
) -> NextLatVerification:
    """Compute consecutive acceptance and a canonical target-KV commit plan.

    Target row ``j`` predicts candidate ``j + 1``.  If ``k`` drafts match
    consecutively, KV commits the anchor plus those ``k`` drafts, while the
    user-visible output is the accepted drafts plus the target bonus token.
    """

    proposal.validate()
    if compact_target_token_ids.ndim != 1:
        raise ValueError("compact_target_token_ids must have shape [sum(verify_lens)]")
    if compact_target_token_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("compact_target_token_ids must contain integer token ids")
    if prefix_seq_lens.shape != proposal.anchor_tokens.shape:
        raise ValueError("prefix_seq_lens must have shape [batch]")
    if prefix_seq_lens.device != proposal.anchor_tokens.device:
        raise ValueError("prefix_seq_lens must be on the proposal device")

    verify_lens = proposal.verify_lens.to(torch.int64)
    observed_rows = compact_target_token_ids.numel()
    _require_tensor_true(
        verify_lens.sum() == observed_rows,
        "target row count differs from sum(verify_lens)",
    )
    if compact_target_token_ids.device != proposal.anchor_tokens.device:
        raise ValueError("target token ids must be on the proposal device")

    row_starts = torch.cumsum(verify_lens, dim=0) - verify_lens
    verify_width = proposal.verify_width
    max_draft_tokens = proposal.max_draft_tokens
    columns = torch.arange(verify_width, device=row_starts.device)
    gather_index = row_starts[:, None] + columns[None, :]
    valid_target = columns[None, :] < verify_lens[:, None]
    safe_index = gather_index.clamp(max=max(observed_rows - 1, 0))
    target_strided = compact_target_token_ids.to(torch.int64)[safe_index]
    target_strided = torch.where(valid_target, target_strided, 0)

    draft_columns = torch.arange(max_draft_tokens, device=row_starts.device)
    proposed = draft_columns[None, :] < proposal.proposal_lens[:, None]
    token_matches = proposal.draft_tokens.to(torch.int64) == target_strided[:, :-1]
    consecutive = torch.cumprod((proposed & token_matches).to(torch.int32), dim=1)
    num_correct = consecutive.sum(dim=1).to(torch.int32)
    accept_lens = num_correct + 1

    bonus_tokens = target_strided.gather(
        1, num_correct.to(torch.int64)[:, None]
    ).squeeze(1)

    output_columns = torch.arange(verify_width, device=row_starts.device)
    accepted_draft = output_columns[None, :] < num_correct[:, None]
    bonus_column = output_columns[None, :] == num_correct[:, None]
    draft_for_output = torch.nn.functional.pad(
        proposal.draft_tokens.to(torch.int64), (0, 1)
    )
    output_tokens = torch.where(accepted_draft, draft_for_output, 0)
    output_tokens = torch.where(bonus_column, bonus_tokens[:, None], output_tokens)

    committed = output_columns[None, :] <= num_correct[:, None]
    accept_index = torch.where(
        committed,
        row_starts[:, None] + output_columns[None, :],
        torch.full_like(gather_index, -1),
    ).to(torch.int64)
    selected_hidden_indices = row_starts + num_correct.to(torch.int64)
    new_seq_lens = prefix_seq_lens + accept_lens.to(prefix_seq_lens.dtype)

    return NextLatVerification(
        num_correct_drafts=num_correct,
        accept_lens=accept_lens,
        accept_index=accept_index,
        output_tokens=output_tokens,
        bonus_tokens=bonus_tokens,
        new_seq_lens=new_seq_lens,
        selected_hidden_indices=selected_hidden_indices,
    )
