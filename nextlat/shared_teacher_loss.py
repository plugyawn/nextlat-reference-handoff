"""Opt-in shared-teacher CE + recurrent KL schedules.

The first schedule provides an objective-preserving way to reduce the d=4
full-vocabulary projection work from nine logical projections to five:

* one exact-state projection, used both for causal CE and (detached) teacher
  distributions at every recurrent horizon; and
* one student projection for each of the four recurrent horizons.

The opt-in horizon-stacked variant concatenates the four student groups for a
CE chunk.  It therefore executes one exact projection and one student
projection per chunk while returning tokenwise KL values so each horizon keeps
its independent reduction and denominator.

Index proof
-----------
Let ``T`` be sequence length and let a horizon-``s`` recurrent row have local
index ``t``.  The official objective compares

``student = recurrent_s[:, t]`` with ``teacher = hidden[:, t + s]``.

Both distributions predict token ``input_ids[:, t + s + 1]``.  Causal CE
already projects ``hidden[:, j]`` for ``j = 0 .. T-2`` and predicts token
``j+1``.  Substituting ``j = t+s`` shows that the KL teacher is exactly a CE
row, not merely a state with the same target token.

Moreover, the official horizon-``s`` KL mask requires attention at offsets
``t .. t+s+1`` and the assistant mask at ``t+s+1``.  The CE mask at ``j=t+s``
requires the final two of those attention positions and the same assistant
position.  Therefore ``kl_mask_s[:, t] => ce_mask[:, t+s]``.  The compact
index map below checks this implication fail-closed before any projection.

Gradient contract
-----------------
The exact projection remains in the ordinary PyTorch CE graph, so CE updates
both exact hidden states and the tied LM head exactly as before.  Teacher log
probabilities are detached before KL.  The KL custom autograd function saves
only the already-computed hidden-width student gradient, so KL updates only
the recurrent student states and, transitively, the connected recurrence.
It intentionally supports first-order optimization only.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn.functional as F

from .core import (
    NextLatDynamics,
    NextLatLossConfig,
    nextlat_position_masks,
    noisy_training_anchor,
)


class _StudentKLFromTeacherLogProbs(torch.autograd.Function):
    """Token-summed KL with a detached cached teacher and detached LM head.

    ``teacher_log_probs`` must come from the exact-state projection that is
    still live in the CE graph.  Only a hidden-width student gradient is kept
    for backward; no vocabulary-width tensor is retained by this function.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx: Any,
        teacher_log_probs: torch.Tensor,
        student_states: torch.Tensor,
        detached_head_weight: torch.Tensor,
    ) -> torch.Tensor:
        student_logits = F.linear(student_states, detached_head_weight)
        student_log_probs = F.log_softmax(student_logits.float(), dim=-1)
        total = F.kl_div(
            student_log_probs,
            teacher_log_probs,
            log_target=True,
            reduction="sum",
        )

        # This is log_softmax's exact first-order formula.  Retaining the
        # teacher mass makes it agree with PyTorch even if exp(log_softmax())
        # sums to something infinitesimally different from one.
        teacher_probs = teacher_log_probs.exp()
        probability_delta = (
            student_log_probs.exp()
            * teacher_probs.sum(dim=-1, keepdim=True)
            - teacher_probs
        )
        student_gradient = probability_delta.to(student_logits.dtype).matmul(
            detached_head_weight
        )
        ctx.save_for_backward(student_gradient.to(student_states.dtype))
        return total

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx: Any, grad_output: torch.Tensor):
        if torch.is_grad_enabled():
            raise RuntimeError(
                "shared-teacher KL supports first-order gradients only"
            )
        (student_gradient,) = ctx.saved_tensors
        return None, student_gradient * grad_output, None


class _HorizonStackedStudentKLFromTeacherLogProbs(torch.autograd.Function):
    """Tokenwise KL for one projection containing all d=4 horizons.

    Returning one value per row lets the caller perform the four horizon sums
    independently.  Backward receives the resulting per-row upstream scales,
    so unequal horizon denominators remain exact while only one hidden-width
    gradient tensor is retained.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx: Any,
        teacher_log_probs: torch.Tensor,
        student_states: torch.Tensor,
        detached_head_weight: torch.Tensor,
    ) -> torch.Tensor:
        student_logits = F.linear(student_states, detached_head_weight)
        student_log_probs = F.log_softmax(student_logits.float(), dim=-1)
        tokenwise = F.kl_div(
            student_log_probs,
            teacher_log_probs,
            log_target=True,
            reduction="none",
        ).sum(dim=-1)

        teacher_probs = teacher_log_probs.exp()
        probability_delta = (
            student_log_probs.exp()
            * teacher_probs.sum(dim=-1, keepdim=True)
            - teacher_probs
        )
        student_gradient = probability_delta.to(student_logits.dtype).matmul(
            detached_head_weight
        )
        ctx.save_for_backward(student_gradient.to(student_states.dtype))
        return tokenwise

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx: Any, grad_output: torch.Tensor):
        if torch.is_grad_enabled():
            raise RuntimeError(
                "horizon-stacked shared-teacher KL supports first-order "
                "gradients only"
            )
        (student_gradient,) = ctx.saved_tensors
        return None, student_gradient * grad_output.unsqueeze(-1), None


class _HorizonStackedStudentKLTriton(torch.autograd.Function):
    """Tokenwise KL with indirect teacher rows and a fused CUDA epilogue.

    This preserves the same first-order contract as the eager horizon-stacked
    implementation.  The only saved tensor is the final hidden-width student
    gradient; the exact teacher and vocabulary head remain detached from KL.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx: Any,
        teacher_log_probs: torch.Tensor,
        teacher_rows: torch.Tensor,
        student_states: torch.Tensor,
        detached_head_weight: torch.Tensor,
    ) -> torch.Tensor:
        from .triton_horizon_kl import (
            triton_tokenwise_kl_and_probability_delta,
        )

        student_logits = F.linear(student_states, detached_head_weight)
        # Casting as part of log-softmax avoids a separate FP32 logits tensor.
        student_log_probs = F.log_softmax(
            student_logits, dim=-1, dtype=torch.float32
        )
        tokenwise, probability_delta = (
            triton_tokenwise_kl_and_probability_delta(
                student_log_probs=student_log_probs,
                teacher_log_probs=teacher_log_probs,
                teacher_rows=teacher_rows,
                delta_dtype=student_logits.dtype,
            )
        )
        student_gradient = probability_delta.matmul(detached_head_weight)
        ctx.save_for_backward(student_gradient.to(student_states.dtype))
        return tokenwise

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx: Any, grad_output: torch.Tensor):
        if torch.is_grad_enabled():
            raise RuntimeError(
                "horizon-stacked Triton KL supports first-order gradients only"
            )
        (student_gradient,) = ctx.saved_tensors
        return (
            None,
            None,
            student_gradient * grad_output.unsqueeze(-1),
            None,
        )


def _compact_teacher_indices_d4(
    ce_mask: torch.Tensor,
    kl_masks: Sequence[torch.Tensor],
) -> list[torch.Tensor]:
    """Map each selected d=4 KL row to its compact selected CE row.

    The returned horizon-``s`` tensor is in the same flattened order as
    ``student_s[kl_mask_s]`` and contains the row into
    ``hidden[:, :-1][ce_mask]`` that supplies its teacher distribution.
    """

    if ce_mask.ndim != 2:
        raise ValueError("CE mask must be rank two")
    if len(kl_masks) != 4:
        raise ValueError("shared-teacher prototype requires exactly d=4")

    valid_ce = int(ce_mask.sum().detach().item())
    if valid_ce == 0:
        raise ValueError("CE mask has zero valid positions")
    dense_to_compact = torch.full(
        ce_mask.shape,
        -1,
        dtype=torch.long,
        device=ce_mask.device,
    )
    dense_to_compact[ce_mask] = torch.arange(
        valid_ce, dtype=torch.long, device=ce_mask.device
    )

    result: list[torch.Tensor] = []
    for step, kl_mask in enumerate(kl_masks, start=1):
        expected_shape = (ce_mask.shape[0], ce_mask.shape[1] - step)
        if tuple(kl_mask.shape) != expected_shape:
            raise ValueError(
                f"horizon {step} KL mask must have shape {expected_shape}; "
                f"got {tuple(kl_mask.shape)}"
            )
        # Local KL row t is taught by exact hidden row j=t+step.
        shifted_ce_mask = ce_mask[:, step:]
        if bool((kl_mask & ~shifted_ce_mask).any().detach().item()):
            raise RuntimeError(
                f"horizon {step} KL mask contains a row without a CE teacher"
            )
        indices = dense_to_compact[:, step:][kl_mask]
        if indices.numel() == 0:
            raise ValueError(f"horizon {step} KL mask has zero valid positions")
        if bool((indices < 0).any().detach().item()):
            raise RuntimeError(f"horizon {step} teacher compaction failed")
        result.append(indices)
    return result


def _compact_generated_teacher_indices_d4(
    ce_mask: torch.Tensor,
    kl_masks: Sequence[torch.Tensor],
) -> list[torch.Tensor]:
    """Fast compact mapping for masks from ``nextlat_position_masks``.

    The public proof helper above validates arbitrary masks.  The training hot
    path constructs all masks together, so the KL-subset implication is true
    by construction and repeating eight device-to-host assertions every
    forward only serializes the CUDA launch stream.  A prefix rank gives the
    same compact CE row without scalar extraction.
    """

    compact_rank = ce_mask.reshape(-1).to(dtype=torch.long).cumsum(dim=0) - 1
    dense_to_compact = compact_rank.reshape(ce_mask.shape)
    return [
        dense_to_compact[:, step:][kl_mask]
        for step, kl_mask in enumerate(kl_masks, start=1)
    ]


def _masked_smooth_l1_from_count(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Authority SmoothL1 algebra using an already-computed device count."""

    elementwise = F.smooth_l1_loss(predicted, target.detach(), reduction="none")
    weights = mask.unsqueeze(-1).to(dtype=elementwise.dtype)
    return (elementwise * weights).sum() / (valid * predicted.shape[-1])


def _compute_shared_teacher_ce_kl_d4_from_masks(
    *,
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    head_weight: torch.Tensor,
    predicted_states_by_horizon: Sequence[torch.Tensor],
    chunk_tokens: int,
    ce_mask: torch.Tensor,
    kl_masks: Sequence[torch.Tensor],
    stack_horizons: bool = False,
    fused_horizon_kl: bool = False,
) -> dict[str, Any]:
    """Shared-teacher projection schedule for jointly generated masks."""

    if hidden_states.ndim != 3:
        raise ValueError("hidden_states must have shape (B,T,D)")
    if input_ids.shape != hidden_states.shape[:2]:
        raise ValueError("input_ids must match hidden-state batch and time")
    if head_weight.ndim != 2 or head_weight.shape[1] != hidden_states.shape[2]:
        raise ValueError("head_weight must have shape (V,D)")
    if len(predicted_states_by_horizon) != 4:
        raise ValueError("shared-teacher prototype requires exactly d=4")
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    if fused_horizon_kl and not stack_horizons:
        raise ValueError("fused horizon KL requires horizon stacking")

    ce_states = hidden_states[:, :-1][ce_mask]
    ce_targets = input_ids[:, 1:][ce_mask]
    ce_positions = ce_states.shape[0]
    if ce_positions == 0:
        raise ValueError("CE mask has zero valid positions")
    teacher_indices = _compact_generated_teacher_indices_d4(ce_mask, kl_masks)

    students: list[torch.Tensor] = []
    source_rows_by_teacher: list[torch.Tensor] = []
    kl_positions: list[int] = []
    batch, length, width = hidden_states.shape
    for step, (predicted, kl_mask, compact_teacher_rows) in enumerate(
        zip(predicted_states_by_horizon, kl_masks, teacher_indices), start=1
    ):
        expected_shape = (batch, length - step, width)
        if tuple(predicted.shape) != expected_shape:
            raise ValueError(
                f"horizon {step} predicted states must have shape "
                f"{expected_shape}; got {tuple(predicted.shape)}"
            )
        selected = predicted[:, :-1][kl_mask]
        students.append(selected)
        positions = selected.shape[0]
        if positions == 0:
            raise ValueError(f"horizon {step} KL mask has zero valid positions")
        kl_positions.append(positions)

        # Reverse the strictly one-to-one teacher mapping.  This lets each CE
        # chunk gather its matching students without retaining teacher logits.
        source_rows = torch.full(
            (ce_positions,), -1, dtype=torch.long, device=ce_states.device
        )
        source_rows[compact_teacher_rows] = torch.arange(
            positions, dtype=torch.long, device=ce_states.device
        )
        source_rows_by_teacher.append(source_rows)

    ce_sum = torch.zeros((), dtype=torch.float32, device=hidden_states.device)
    kl_sums = [
        torch.zeros((), dtype=torch.float32, device=hidden_states.device)
        for _ in range(4)
    ]
    detached_head = head_weight.detach()
    for start in range(0, ce_positions, chunk_tokens):
        stop = min(start + chunk_tokens, ce_positions)
        # This is the sole exact-state vocabulary projection for the chunk.
        # It stays connected for CE, while its log-probabilities are detached
        # before serving as the shared KL teacher.
        exact_logits = F.linear(ce_states[start:stop], head_weight)
        exact_log_probs = F.log_softmax(exact_logits.float(), dim=-1)
        # NLL(log_softmax(z)) is the operation used by cross_entropy, but
        # spelling it this way lets CE and all detached KL teachers share both
        # the vocabulary projection and the normalization.
        ce_sum = ce_sum + F.nll_loss(
            exact_log_probs, ce_targets[start:stop], reduction="sum"
        )
        teacher_log_probs = exact_log_probs.detach()

        gathered_students: list[torch.Tensor] = []
        gathered_teachers: list[torch.Tensor] = []
        gathered_teacher_rows: list[torch.Tensor] = []
        chunk_horizon_positions: list[int] = []
        for horizon_index in range(4):
            source_rows = source_rows_by_teacher[horizon_index][start:stop]
            present = source_rows >= 0
            if fused_horizon_kl:
                # One compact index list drives both gathers and avoids the
                # eager path's padded [CE rows, hidden] intermediate.
                present_rows = torch.nonzero(
                    present, as_tuple=False
                ).flatten()
                gathered = students[horizon_index].index_select(
                    0, source_rows.index_select(0, present_rows)
                )
            else:
                # Clamp supplies a legal index for absent rows; the boolean
                # select immediately removes them.  Empty [0,D] chunks are
                # valid and avoid a device-to-host synchronization branch.
                gathered = students[horizon_index].index_select(
                    0, source_rows.clamp_min(0)
                )[present]
            if stack_horizons:
                gathered_students.append(gathered)
                if fused_horizon_kl:
                    # Reuse the exact distribution in place instead of
                    # materializing it once per recurrent horizon.
                    gathered_teacher_rows.append(present_rows)
                else:
                    gathered_teachers.append(teacher_log_probs[present])
                chunk_horizon_positions.append(gathered.shape[0])
            else:
                kl_sums[horizon_index] = (
                    kl_sums[horizon_index]
                    + _StudentKLFromTeacherLogProbs.apply(
                        teacher_log_probs[present], gathered, detached_head
                    )
                )

        if stack_horizons:
            stacked_positions = sum(chunk_horizon_positions)
            if stacked_positions == 0:
                continue
            if fused_horizon_kl:
                tokenwise_kl = _HorizonStackedStudentKLTriton.apply(
                    teacher_log_probs,
                    torch.cat(gathered_teacher_rows, dim=0),
                    torch.cat(gathered_students, dim=0),
                    detached_head,
                )
            else:
                tokenwise_kl = (
                    _HorizonStackedStudentKLFromTeacherLogProbs.apply(
                        torch.cat(gathered_teachers, dim=0),
                        torch.cat(gathered_students, dim=0),
                        detached_head,
                    )
                )
            horizon_values = tokenwise_kl.split(chunk_horizon_positions)
            for horizon_index, values in enumerate(horizon_values):
                kl_sums[horizon_index] = kl_sums[horizon_index] + values.sum()

    ce = ce_sum / ce_positions
    kl_by_horizon = torch.stack(
        [total / count for total, count in zip(kl_sums, kl_positions)]
    )
    kl_positions_by_horizon = torch.stack([mask.sum() for mask in kl_masks])
    return {
        "ce": ce,
        "kl": kl_by_horizon.mean(),
        "kl_by_horizon": kl_by_horizon,
        "ce_positions": ce_mask.sum(),
        "kl_positions": kl_positions_by_horizon.sum(),
        "kl_positions_by_horizon": kl_positions_by_horizon,
    }


def compute_shared_teacher_ce_kl_d4(
    *,
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    assistant_loss_mask: torch.Tensor,
    head_weight: torch.Tensor,
    predicted_states_by_horizon: Sequence[torch.Tensor],
    chunk_tokens: int,
    stack_horizons: bool = False,
) -> dict[str, Any]:
    """Compute causal CE and four official KL terms with one teacher pass.

    This function does not construct or alter the recurrence and does not
    compute SmoothL1.  Callers pass the same connected recurrent tensors used
    by the official d=4 objective.  Results preserve the independent CE and
    per-horizon KL means and their exact position denominators.
    """

    if attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must match input_ids")
    if assistant_loss_mask.shape != input_ids.shape:
        raise ValueError("assistant_loss_mask must match input_ids")
    ce_mask, _, kl_masks = nextlat_position_masks(
        attention_mask, assistant_loss_mask, horizon=4
    )
    return _compute_shared_teacher_ce_kl_d4_from_masks(
        hidden_states=hidden_states,
        input_ids=input_ids,
        head_weight=head_weight,
        predicted_states_by_horizon=predicted_states_by_horizon,
        chunk_tokens=chunk_tokens,
        ce_mask=ce_mask,
        kl_masks=kl_masks,
        stack_horizons=stack_horizons,
    )


def compute_nextlat_losses_shared_teacher_d4(
    *,
    hidden_states: torch.Tensor,
    token_embeddings: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    assistant_loss_mask: torch.Tensor,
    head_weight: torch.Tensor,
    dynamics: NextLatDynamics,
    config: NextLatLossConfig,
) -> dict[str, Any]:
    """Compute the complete connected d=4 objective with shared teachers.

    The recurrence and SmoothL1 path are the same operations and reductions as
    :func:`nextlat.core.compute_nextlat_losses`.  Only the CE/KL vocabulary
    schedule differs: the exact CE projection and normalization are reused as
    detached teachers by all four KL horizons.

    The returned dictionary intentionally has exactly the same keys and count
    semantics as the authority implementation so the training loop does not
    need a second interface.
    """

    supported_implementations = {
        "shared_teacher_forward_gradient",
        "shared_teacher_horizon_stacked",
        "shared_teacher_horizon_stacked_triton",
    }
    if config.kl_implementation not in supported_implementations:
        raise ValueError(
            "shared-teacher d4 received an unsupported KL implementation"
        )
    if config.horizon != 4:
        raise ValueError(
            f"{config.kl_implementation} requires NextLat horizon=4"
        )
    if hidden_states.ndim != 3 or hidden_states.shape != token_embeddings.shape:
        raise ValueError("hidden_states and token_embeddings must be matching (B,T,D)")
    if input_ids.shape != hidden_states.shape[:2]:
        raise ValueError("input_ids must match the (B,T) hidden-state prefix")
    if attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must match input_ids")
    if assistant_loss_mask.shape != input_ids.shape:
        raise ValueError("assistant_loss_mask must match input_ids")
    if config.vocab_chunk_tokens <= 0:
        raise ValueError("vocab_chunk_tokens must be positive")

    ce_mask, transition_masks, kl_masks = nextlat_position_masks(
        attention_mask, assistant_loss_mask, horizon=4
    )
    predicted_by_horizon: list[torch.Tensor] = []
    smooth_by_horizon: list[torch.Tensor] = []
    recurrent = noisy_training_anchor(
        hidden_states, dynamics=dynamics, relative=config.anchor_noise_relative
    )

    # Counts stay as device scalars.  Every KL mask is nonempty-checked from
    # its selected tensor below, and KL nonemptiness implies its transition
    # mask is nonempty, so SmoothL1 needs no redundant host synchronization.
    transition_counts = [mask.sum() for mask in transition_masks]

    # Keep the official recurrence connected across all four applications.
    for step in range(1, 5):
        recurrent = dynamics(recurrent[:, :-1], token_embeddings[:, step:])
        predicted_by_horizon.append(recurrent)
        transition_mask = transition_masks[step - 1]
        smooth_by_horizon.append(
            _masked_smooth_l1_from_count(
                recurrent,
                hidden_states[:, step:],
                transition_mask,
                transition_counts[step - 1],
            )
        )

    projection_losses = _compute_shared_teacher_ce_kl_d4_from_masks(
        hidden_states=hidden_states,
        input_ids=input_ids,
        head_weight=head_weight,
        predicted_states_by_horizon=predicted_by_horizon,
        chunk_tokens=config.vocab_chunk_tokens,
        ce_mask=ce_mask,
        kl_masks=kl_masks,
        stack_horizons=(
            config.kl_implementation
            in {
                "shared_teacher_horizon_stacked",
                "shared_teacher_horizon_stacked_triton",
            }
        ),
        fused_horizon_kl=(
            config.kl_implementation
            == "shared_teacher_horizon_stacked_triton"
        ),
    )
    smooth_l1_by_horizon = torch.stack(smooth_by_horizon)
    transition_positions_by_horizon = torch.stack(transition_counts)
    smooth_l1 = smooth_l1_by_horizon.mean()
    total = (
        projection_losses["ce"]
        + config.lambda_h * smooth_l1
        + config.lambda_kl * projection_losses["kl"]
    )
    return {
        "loss": total,
        "ce": projection_losses["ce"],
        "smooth_l1": smooth_l1,
        "kl": projection_losses["kl"],
        "predicted_next_states": predicted_by_horizon[0],
        "predicted_states_by_horizon": predicted_by_horizon,
        "smooth_l1_by_horizon": smooth_l1_by_horizon,
        "kl_by_horizon": projection_losses["kl_by_horizon"],
        "transition_positions": transition_positions_by_horizon.sum(),
        "kl_positions": projection_losses["kl_positions"],
        "transition_positions_by_horizon": transition_positions_by_horizon,
        "kl_positions_by_horizon": projection_losses[
            "kl_positions_by_horizon"
        ],
        "ce_positions": projection_losses["ce_positions"],
    }
