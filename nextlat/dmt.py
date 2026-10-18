"""Detached memory training for a frozen NextLat target.

This module is intentionally separate from the joint NextLat objective.  A DMT
batch is constructed in two phases:

1. Roll the dynamics model forward under ``torch.no_grad`` using either corpus
   tokens or tokens sampled from the recurrent state.
2. Exact-replay the resulting token sequence through the frozen target and
   train every ``(detached recurrent state, token) -> exact next state`` pair as
   an independent one-step example.

The second phase is dense in time.  There is no BPTT graph: recurrent inputs,
token embeddings, exact targets, and the vocabulary head are all detached.
Only the explicitly selected repair module receives gradients.  The original
variants train the dynamics; ``counterfactual_readout`` instead freezes both
the target and dynamics and trains a residual observation adapter used solely
before the frozen vocabulary head.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Literal, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import (
    NextLatLossConfig,
    compute_nextlat_auxiliary_losses,
    nextlat_position_masks,
)


RolloutMode = Literal["teacher_forced", "recurrent_sampled"]
StateInputMode = Literal["recurrent", "exact"]
LossVariant = Literal[
    "hybrid_smooth_l1_kl",
    "official_detached_mse",
    "official_detached_mse_exact_control",
    "anchor_head_repair",
    "counterfactual_readout",
]

HYBRID_LOSS_VARIANT = "hybrid_smooth_l1_kl"
OFFICIAL_DETACHED_MSE_VARIANT = "official_detached_mse"
OFFICIAL_DETACHED_MSE_EXACT_CONTROL_VARIANT = (
    "official_detached_mse_exact_control"
)
OFFICIAL_DETACHED_MSE_VARIANTS = (
    OFFICIAL_DETACHED_MSE_VARIANT,
    OFFICIAL_DETACHED_MSE_EXACT_CONTROL_VARIANT,
)
ANCHOR_HEAD_REPAIR_VARIANT = "anchor_head_repair"
COUNTERFACTUAL_READOUT_VARIANT = "counterfactual_readout"
OFFICIAL_DETACHED_MSE_COEFFICIENT = 9.375


@dataclass(frozen=True)
class DetachedDMTConfig:
    rollout_mode: RolloutMode = "recurrent_sampled"
    state_input_mode: StateInputMode = "recurrent"
    loss_variant: LossVariant = HYBRID_LOSS_VARIANT
    rollout_horizon: int = 64
    loss_stride: int = 1
    lambda_h: float = 1.0
    lambda_kl: float = 1.0
    lambda_overlap: float = 0.0
    lambda_identity: float = 0.0
    official_mse_coefficient: float = OFFICIAL_DETACHED_MSE_COEFFICIENT
    vocab_chunk_tokens: int = 128
    temperature: float = 1.0
    top_k: int = 32
    max_anchor_position: Optional[int] = 1024

    def validate(self) -> None:
        if self.rollout_mode not in ("teacher_forced", "recurrent_sampled"):
            raise ValueError(f"unsupported rollout_mode: {self.rollout_mode}")
        if self.state_input_mode not in ("recurrent", "exact"):
            raise ValueError(f"unsupported state_input_mode: {self.state_input_mode}")
        if self.loss_variant not in (
            HYBRID_LOSS_VARIANT,
            *OFFICIAL_DETACHED_MSE_VARIANTS,
            ANCHOR_HEAD_REPAIR_VARIANT,
            COUNTERFACTUAL_READOUT_VARIANT,
        ):
            raise ValueError(f"unsupported loss_variant: {self.loss_variant}")
        if self.rollout_mode != "teacher_forced" and self.state_input_mode != "recurrent":
            raise ValueError("exact state inputs are only defined for teacher-forced rollouts")
        if self.rollout_horizon < 1:
            raise ValueError("rollout_horizon must be positive")
        if self.loss_stride < 1 or self.loss_stride > self.rollout_horizon:
            raise ValueError("loss_stride must be in [1, rollout_horizon]")
        if (
            self.lambda_h < 0
            or self.lambda_kl < 0
            or self.lambda_overlap < 0
            or self.lambda_identity < 0
        ):
            raise ValueError("loss coefficients must be non-negative")
        if self.loss_variant == HYBRID_LOSS_VARIANT:
            if self.lambda_h == 0 and self.lambda_kl == 0:
                raise ValueError("at least one hybrid DMT loss coefficient must be positive")
        elif self.loss_variant in OFFICIAL_DETACHED_MSE_VARIANTS:
            required_state_input = (
                "exact"
                if self.loss_variant
                == OFFICIAL_DETACHED_MSE_EXACT_CONTROL_VARIANT
                else "recurrent"
            )
            if (
                self.rollout_mode != "teacher_forced"
                or self.state_input_mode != required_state_input
            ):
                raise ValueError(
                    f"{self.loss_variant} requires teacher-forced "
                    f"{required_state_input} states"
                )
            if self.rollout_horizon != 256 or self.loss_stride != 16:
                raise ValueError("official detached DMT requires horizon 256 and stride 16")
            if self.lambda_h != 0 or self.lambda_kl != 0:
                raise ValueError(
                    "official detached DMT excludes SmoothL1 and vocabulary-head KL"
                )
            if self.official_mse_coefficient != OFFICIAL_DETACHED_MSE_COEFFICIENT:
                raise ValueError(
                    "official detached DMT requires effective MSE coefficient 9.375"
                )
        elif self.loss_variant == ANCHOR_HEAD_REPAIR_VARIANT:
            if self.rollout_mode != "teacher_forced" or self.state_input_mode != "recurrent":
                raise ValueError(
                    "anchor-head repair requires teacher-forced recurrent states"
                )
            if self.rollout_horizon != 256 or self.loss_stride != 16:
                raise ValueError("anchor-head repair requires horizon 256 and stride 16")
            if self.lambda_h != 0 or self.lambda_kl != 1:
                raise ValueError(
                    "anchor-head repair requires lambda_h=0 and lambda_kl=1"
                )
            if self.official_mse_coefficient != OFFICIAL_DETACHED_MSE_COEFFICIENT:
                raise ValueError(
                    "anchor-head repair requires effective MSE coefficient 9.375"
                )
        else:
            if self.rollout_mode != "recurrent_sampled" or self.state_input_mode != "recurrent":
                raise ValueError(
                    "counterfactual readout requires recurrent-sampled recurrent states"
                )
            if self.rollout_horizon < 2 or self.loss_stride != 1:
                raise ValueError(
                    "counterfactual readout requires horizon >=2 and stride 1"
                )
            if self.lambda_h != 0 or self.lambda_kl <= 0:
                raise ValueError(
                    "counterfactual readout requires lambda_h=0 and positive lambda_kl"
                )
            if self.lambda_overlap != 0 or self.lambda_identity != 0:
                raise ValueError(
                    "counterfactual readout Stage 0 optimizes pure target||readout KL"
                )
        if self.vocab_chunk_tokens < 1:
            raise ValueError("vocab_chunk_tokens must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")
        if self.max_anchor_position is not None and self.max_anchor_position < 0:
            raise ValueError("max_anchor_position must be non-negative or null")


@dataclass(frozen=True)
class NoGradRollout:
    current_states: torch.Tensor
    token_ids: torch.Tensor
    valid_mask: torch.Tensor
    next_states: torch.Tensor


@dataclass(frozen=True)
class DetachedDMTBatch:
    current_states: torch.Tensor
    token_ids: torch.Tensor
    token_embeddings: torch.Tensor
    exact_next_states: torch.Tensor
    exact_current_states: torch.Tensor
    valid_mask: torch.Tensor
    anchor_positions: torch.Tensor
    replay_input_ids: torch.Tensor
    replay_attention_mask: torch.Tensor
    target_forward_tokens: int
    target_padded_forward_tokens: int

    @property
    def transition_positions(self) -> int:
        return int(self.valid_mask.sum().item())


class RecurrentReadoutAdapter(nn.Module):
    """Identity-initialized residual observation map for recurrent states.

    This module is deliberately not part of :class:`NextLatDynamics`.  Its
    output may be passed to the frozen LM head, but it must never be fed back
    into the recurrence or applied to exact target states.  Zero-initializing
    the up projection makes a fresh adapter exactly preserve the parent
    proposal distribution.
    """

    def __init__(self, hidden_size: int, bottleneck_size: int = 256) -> None:
        super().__init__()
        if hidden_size < 1 or bottleneck_size < 1:
            raise ValueError("readout widths must be positive")
        self.hidden_size = int(hidden_size)
        self.bottleneck_size = int(bottleneck_size)
        self.norm_weight = nn.Parameter(torch.ones(hidden_size))
        self.down = nn.Linear(hidden_size, bottleneck_size, bias=False)
        self.up = nn.Linear(bottleneck_size, hidden_size, bias=False)
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        if states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"readout expected hidden width {self.hidden_size}, got {states.shape[-1]}"
            )
        normalized = F.rms_norm(
            states, (self.hidden_size,), self.norm_weight, 1e-5
        )
        return states + self.up(F.silu(self.down(normalized)))


def recurrent_readout_state(
    states: torch.Tensor,
    readout_adapter: Optional[nn.Module],
    *,
    recurrent_depth: int,
) -> torch.Tensor:
    """Route only non-anchor recurrent states through the observation adapter."""

    if recurrent_depth < 0:
        raise ValueError("recurrent_depth must be non-negative")
    if readout_adapter is None or recurrent_depth == 0:
        return states
    return readout_adapter(states)


def freeze_target_model(model: nn.Module) -> None:
    """Freeze the complete target, including tied embeddings and LM head."""

    model.requires_grad_(False)
    model.eval()
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if trainable:
        raise RuntimeError(f"target still has trainable parameters: {trainable[:4]}")


def assert_target_frozen(model: nn.Module) -> None:
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if trainable:
        raise RuntimeError(f"DMT requires a frozen target: {trainable[:4]}")
    if model.training:
        raise RuntimeError("DMT target must remain in eval mode")


def _validate_state_inputs(
    anchor_states: torch.Tensor,
    embedding_weight: torch.Tensor,
    head_weight: torch.Tensor,
) -> None:
    if anchor_states.ndim != 2:
        raise ValueError("anchor_states must be (B,D)")
    if embedding_weight.ndim != 2 or head_weight.ndim != 2:
        raise ValueError("embedding and head weights must be matrices")
    if anchor_states.shape[-1] != embedding_weight.shape[-1]:
        raise ValueError("anchor and embedding widths differ")
    if anchor_states.shape[-1] != head_weight.shape[-1]:
        raise ValueError("anchor and head widths differ")
    if embedding_weight.shape[0] != head_weight.shape[0]:
        raise ValueError("embedding and head vocabulary sizes differ")


def _sample_tokens(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    scaled = logits.float() / temperature
    if top_k:
        k = min(top_k, scaled.shape[-1])
        values, indices = torch.topk(scaled, k=k, dim=-1)
        draw = torch.multinomial(
            F.softmax(values, dim=-1), num_samples=1, generator=generator
        )
        return indices.gather(-1, draw).squeeze(-1)
    return torch.multinomial(
        F.softmax(scaled, dim=-1), num_samples=1, generator=generator
    ).squeeze(-1)


@torch.no_grad()
def recurrent_sample_rollout(
    *,
    dynamics: nn.Module,
    anchor_states: torch.Tensor,
    embedding_weight: torch.Tensor,
    head_weight: torch.Tensor,
    horizon: int,
    temperature: float = 1.0,
    top_k: int = 32,
    valid_vocab_size: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    eos_token_ids: Optional[Sequence[int]] = None,
    pad_token_id: int = 0,
    readout_adapter: Optional[nn.Module] = None,
    generator: Optional[torch.Generator] = None,
) -> NoGradRollout:
    """Sample a fixed-capacity recurrent rollout without building a graph."""

    _validate_state_inputs(anchor_states, embedding_weight, head_weight)
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if temperature <= 0 or top_k < 0:
        raise ValueError("invalid sampling parameters")

    vocabulary = head_weight.shape[0] if valid_vocab_size is None else int(valid_vocab_size)
    if vocabulary < 1 or vocabulary > head_weight.shape[0]:
        raise ValueError("valid_vocab_size is outside the LM-head roster")
    terminal_ids = {int(value) for value in (eos_token_ids or ())}
    if eos_token_id is not None:
        terminal_ids.add(int(eos_token_id))

    state = anchor_states.detach()
    alive = torch.ones(state.shape[0], dtype=torch.bool, device=state.device)
    current_states = []
    next_states = []
    tokens = []
    masks = []
    for depth in range(horizon):
        current_states.append(state)
        masks.append(alive.clone())
        observed_state = recurrent_readout_state(
            state, readout_adapter, recurrent_depth=depth
        )
        sampled = _sample_tokens(
            F.linear(observed_state, head_weight[:vocabulary].detach()),
            temperature=temperature,
            top_k=top_k,
            generator=generator,
        )
        sampled = torch.where(
            alive, sampled, torch.full_like(sampled, int(pad_token_id))
        )
        tokens.append(sampled)
        token_embeddings = F.embedding(sampled, embedding_weight.detach())
        updated = dynamics(state.detach(), token_embeddings.detach())
        state = torch.where(alive.unsqueeze(-1), updated, state)
        next_states.append(state)
        for terminal_id in terminal_ids:
            alive = alive & sampled.ne(terminal_id)

    return NoGradRollout(
        current_states=torch.stack(current_states, dim=1).detach(),
        token_ids=torch.stack(tokens, dim=1).detach(),
        valid_mask=torch.stack(masks, dim=1).detach(),
        next_states=torch.stack(next_states, dim=1).detach(),
    )


@torch.no_grad()
def teacher_forced_rollout(
    *,
    dynamics: nn.Module,
    anchor_states: torch.Tensor,
    token_ids: torch.Tensor,
    embedding_weight: torch.Tensor,
) -> NoGradRollout:
    """Roll recurrent states over corpus tokens without building a BPTT graph."""

    if anchor_states.ndim != 2 or token_ids.ndim != 2:
        raise ValueError("anchor_states and token_ids must be rank two")
    if anchor_states.shape[0] != token_ids.shape[0]:
        raise ValueError("anchor_states and token_ids batch sizes differ")
    if anchor_states.shape[-1] != embedding_weight.shape[-1]:
        raise ValueError("state and embedding widths differ")
    if token_ids.shape[1] < 1:
        raise ValueError("teacher-forced rollout must contain tokens")

    state = anchor_states.detach()
    current_states = []
    next_states = []
    for step in range(token_ids.shape[1]):
        current_states.append(state)
        embedding = F.embedding(token_ids[:, step], embedding_weight.detach())
        state = dynamics(state.detach(), embedding.detach())
        next_states.append(state)
    mask = torch.ones_like(token_ids, dtype=torch.bool)
    return NoGradRollout(
        current_states=torch.stack(current_states, dim=1).detach(),
        token_ids=token_ids.detach(),
        valid_mask=mask,
        next_states=torch.stack(next_states, dim=1).detach(),
    )


def compute_detached_dmt_losses(
    *,
    dynamics: nn.Module,
    current_states: torch.Tensor,
    token_embeddings: torch.Tensor,
    exact_next_states: torch.Tensor,
    valid_mask: torch.Tensor,
    detached_head_weight: torch.Tensor,
    config: DetachedDMTConfig,
    valid_vocab_size: Optional[int] = None,
) -> dict[str, Any]:
    """Train independent one-step transitions; only ``dynamics`` gets gradients."""

    config.validate()
    if config.loss_variant == COUNTERFACTUAL_READOUT_VARIANT:
        raise ValueError(
            "counterfactual_readout must use compute_counterfactual_readout_losses"
        )
    if current_states.ndim != 3:
        raise ValueError("DMT states must be (B,S,D)")
    if token_embeddings.shape != current_states.shape:
        raise ValueError("token_embeddings must match current_states")
    if exact_next_states.shape != current_states.shape:
        raise ValueError("exact_next_states must match current_states")
    if valid_mask.shape != current_states.shape[:2]:
        raise ValueError("valid_mask must match DMT batch and time dimensions")
    valid = int(valid_mask.sum().item())
    if valid == 0:
        raise ValueError("DMT batch has zero valid transitions")

    # These detaches are the core contract. Even graph-bearing recurrent states
    # passed by a caller cannot create a cross-step gradient path here.
    predicted = dynamics(current_states.detach(), token_embeddings.detach())
    target = exact_next_states.detach()
    selected_prediction = predicted[valid_mask]
    selected_target = target[valid_mask]

    zero = selected_prediction.new_zeros((), dtype=torch.float32)
    recurrent_mse = zero
    recurrent_kl = zero
    anchor_mse = zero
    anchor_kl = zero
    if config.loss_variant in OFFICIAL_DETACHED_MSE_VARIANTS:
        # The official Td=256,k=16 code averages 15 useful state losses plus
        # the identically-zero exact anchor. Averaging only the useful labels
        # therefore maps its coefficient 10 to 10 * 15/16 = 9.375.
        mse = F.mse_loss(
            selected_prediction.float(), selected_target.float(), reduction="mean"
        )
        smooth_l1 = zero
        kl = zero
        loss = config.official_mse_coefficient * mse
    else:
        if config.loss_variant == ANCHOR_HEAD_REPAIR_VARIANT:
            if valid_vocab_size is None:
                raise ValueError("anchor-head repair requires valid_vocab_size")
            if valid_vocab_size < 1 or valid_vocab_size > detached_head_weight.shape[0]:
                raise ValueError("valid_vocab_size is outside the LM-head roster")
            anchor_mask = torch.zeros_like(valid_mask)
            anchor_mask[:, 0] = valid_mask[:, 0]
            recurrent_mask = valid_mask & ~anchor_mask
            expected_anchor_count = current_states.shape[0]
            expected_recurrent_count = 15 * current_states.shape[0]
            if (
                int(anchor_mask.sum().item()) != expected_anchor_count
                or int(recurrent_mask.sum().item()) != expected_recurrent_count
            ):
                raise ValueError(
                    "anchor-head repair requires one depth-1 anchor and "
                    "15 recurrent depths per row"
                )
            recurrent_prediction = predicted[recurrent_mask]
            recurrent_target = target[recurrent_mask]
            anchor_prediction = predicted[anchor_mask]
            anchor_target = target[anchor_mask]
            recurrent_mse = F.mse_loss(
                recurrent_prediction.float(), recurrent_target.float(), reduction="mean"
            )
            anchor_mse = F.mse_loss(
                anchor_prediction.float(), anchor_target.float(), reduction="mean"
            )
            mse = recurrent_mse + anchor_mse
            smooth_l1 = zero
        else:
            mse = zero
            smooth_l1 = F.smooth_l1_loss(
                selected_prediction.float(), selected_target.float(), reduction="mean"
            )
        head = detached_head_weight.detach()
        if valid_vocab_size is not None:
            if valid_vocab_size < 1 or valid_vocab_size > head.shape[0]:
                raise ValueError("valid_vocab_size is outside the LM-head roster")
            head = head[:valid_vocab_size]
        elif config.loss_variant == ANCHOR_HEAD_REPAIR_VARIANT:
            raise ValueError("anchor-head repair requires valid_vocab_size")
        def mean_teacher_student_kl(
            prediction: torch.Tensor, exact_target: torch.Tensor
        ) -> torch.Tensor:
            count = prediction.shape[0]
            total = torch.zeros((), device=predicted.device, dtype=torch.float32)
            for start in range(0, count, config.vocab_chunk_tokens):
                stop = min(start + config.vocab_chunk_tokens, count)
                teacher_logits = F.linear(exact_target[start:stop], head)
                student_logits = F.linear(prediction[start:stop], head)
                teacher_log_probs = F.log_softmax(teacher_logits.float(), dim=-1)
                student_log_probs = F.log_softmax(student_logits.float(), dim=-1)
                total = total + F.kl_div(
                    student_log_probs,
                    teacher_log_probs,
                    log_target=True,
                    reduction="sum",
                )
            return total / count

        if config.loss_variant == ANCHOR_HEAD_REPAIR_VARIANT:
            recurrent_kl = mean_teacher_student_kl(
                recurrent_prediction, recurrent_target
            )
            anchor_kl = mean_teacher_student_kl(anchor_prediction, anchor_target)
            kl = recurrent_kl + anchor_kl
        else:
            kl = mean_teacher_student_kl(selected_prediction, selected_target)
        loss = (
            config.official_mse_coefficient * mse + config.lambda_kl * kl
            if config.loss_variant == ANCHOR_HEAD_REPAIR_VARIANT
            else config.lambda_h * smooth_l1 + config.lambda_kl * kl
        )
    return {
        "loss": loss,
        "mse": mse,
        "smooth_l1": smooth_l1,
        "kl": kl,
        "recurrent_mse": recurrent_mse,
        "recurrent_kl": recurrent_kl,
        "anchor_mse": anchor_mse,
        "anchor_kl": anchor_kl,
        "predicted_next_states": predicted,
        "transition_positions": valid_mask.sum(),
    }


def iter_deep_detached_repair_losses(
    *,
    dynamics: nn.Module,
    connected_states: torch.Tensor,
    exact_hidden_states: torch.Tensor,
    token_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    assistant_loss_mask: torch.Tensor,
    connected_depth: int = 4,
    repair_horizon: int = 16,
) -> Iterator[dict[str, Any]]:
    """Dense state-only repair beyond a short connected NextLat horizon.

    ``connected_states`` is the final recurrent state from the gradient-bearing
    short objective.  It, the exact states, and token embeddings are detached.
    Each subsequent transition therefore gives ``dynamics`` one-step credit,
    while detaching the prediction before it becomes the following recurrent
    input prevents BPTT beyond ``connected_depth``.
    """

    if connected_depth < 1 or repair_horizon <= connected_depth:
        raise ValueError("repair_horizon must exceed a positive connected_depth")
    if exact_hidden_states.ndim != 3:
        raise ValueError("exact_hidden_states must be (B,T,D)")
    if token_embeddings.shape != exact_hidden_states.shape:
        raise ValueError("token_embeddings must match exact_hidden_states")
    if attention_mask.shape != exact_hidden_states.shape[:2]:
        raise ValueError("attention_mask must match exact states")
    if assistant_loss_mask.shape != attention_mask.shape:
        raise ValueError("assistant_loss_mask must match attention_mask")
    expected_connected_shape = (
        exact_hidden_states.shape[0],
        exact_hidden_states.shape[1] - connected_depth,
        exact_hidden_states.shape[2],
    )
    if connected_states.shape != expected_connected_shape:
        raise ValueError(
            "connected_states do not match the final short-horizon state: "
            f"{tuple(connected_states.shape)} != {expected_connected_shape}"
        )

    _, transition_masks, _ = nextlat_position_masks(
        attention_mask, assistant_loss_mask, repair_horizon
    )
    current = connected_states.detach()
    for depth in range(connected_depth + 1, repair_horizon + 1):
        prediction = dynamics(
            current[:, :-1].detach(), token_embeddings[:, depth:].detach()
        )
        target = exact_hidden_states[:, depth:].detach()
        mask = transition_masks[depth - 1]
        valid = int(mask.sum().detach().item())
        if valid == 0:
            raise ValueError(f"deep repair depth {depth} has zero valid positions")
        elementwise = F.smooth_l1_loss(
            prediction.float(), target.float(), reduction="none"
        )
        weights = mask.unsqueeze(-1).to(dtype=elementwise.dtype)
        smooth_l1 = (elementwise * weights).sum() / (
            valid * prediction.shape[-1]
        )
        yield {
            "depth": depth,
            "loss": smooth_l1,
            "smooth_l1": smooth_l1,
            "transition_positions": mask.sum(),
        }
        # The caller backpropagates the yielded one-step loss before advancing
        # this iterator. Only the detached numerical prediction recurs.
        current = prediction.detach()
        del prediction, target, mask, elementwise, weights, smooth_l1


def compute_teacher_forced_bptt_losses(
    *,
    dynamics: nn.Module,
    exact_hidden_states: torch.Tensor,
    token_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    assistant_loss_mask: torch.Tensor,
    detached_head_weight: torch.Tensor,
    config: DetachedDMTConfig,
) -> dict[str, Any]:
    """Train only ``dynamics`` with the official short-horizon objective.

    Every corpus position is an overlapping exact anchor.  The target hidden
    states, corpus-token embeddings, and vocabulary head must come from a
    frozen target under ``no_grad``.  Unlike detached replay, recurrent states
    remain connected across all horizon steps, so depth-d losses backpropagate
    through d applications of ``dynamics``.
    """

    config.validate()
    if (
        config.loss_variant != HYBRID_LOSS_VARIANT
        or config.rollout_mode != "teacher_forced"
        or config.state_input_mode != "recurrent"
        or config.loss_stride != 1
    ):
        raise ValueError(
            "teacher-forced BPTT requires hybrid SmoothL1+KL, recurrent states, "
            "and dense stride 1"
        )
    frozen_inputs = {
        "exact_hidden_states": exact_hidden_states,
        "token_embeddings": token_embeddings,
        "detached_head_weight": detached_head_weight,
    }
    graph_bearing = [name for name, value in frozen_inputs.items() if value.requires_grad]
    if graph_bearing:
        raise ValueError(
            "teacher-forced BPTT requires detached frozen-target inputs: "
            f"{graph_bearing}"
        )

    loss_config = NextLatLossConfig(
        lambda_h=config.lambda_h,
        lambda_kl=config.lambda_kl,
        vocab_chunk_tokens=config.vocab_chunk_tokens,
        horizon=config.rollout_horizon,
    )
    outputs = compute_nextlat_auxiliary_losses(
        hidden_states=exact_hidden_states,
        token_embeddings=token_embeddings,
        attention_mask=attention_mask,
        assistant_loss_mask=assistant_loss_mask,
        head_weight=detached_head_weight,
        dynamics=dynamics,
        config=loss_config,
    )
    outputs["loss"] = (
        config.lambda_h * outputs["smooth_l1"]
        + config.lambda_kl * outputs["kl"]
    )
    return outputs


def compute_counterfactual_readout_losses(
    *,
    readout_adapter: nn.Module,
    recurrent_states: torch.Tensor,
    exact_current_states: torch.Tensor,
    valid_mask: torch.Tensor,
    detached_head_weight: torch.Tensor,
    config: DetachedDMTConfig,
    valid_vocab_size: int,
) -> dict[str, Any]:
    """Fit an observation map on exact-replayed sampled prefixes.

    The recurrence and both state populations are detached.  The target
    distribution comes from the raw exact Qwen state; only the recurrent state
    passes through ``readout_adapter`` before the same frozen valid-vocabulary
    head.  Probability overlap is ``1 - total variation`` and is reported as a
    direct bounded diagnostic as well as an optional auxiliary loss.
    """

    config.validate()
    if config.loss_variant != COUNTERFACTUAL_READOUT_VARIANT:
        raise ValueError("readout loss requires counterfactual_readout variant")
    if recurrent_states.ndim != 3 or exact_current_states.shape != recurrent_states.shape:
        raise ValueError("readout recurrent/exact states must be matching (B,S,D)")
    if valid_mask.shape != recurrent_states.shape[:2]:
        raise ValueError("readout valid_mask must match batch and time dimensions")
    valid = int(valid_mask.sum().item())
    if valid == 0:
        raise ValueError("readout batch has zero valid recurrent states")
    if valid_vocab_size < 1 or valid_vocab_size > detached_head_weight.shape[0]:
        raise ValueError("valid_vocab_size is outside the LM-head roster")

    recurrent = recurrent_states.detach()[valid_mask]
    exact = exact_current_states.detach()[valid_mask]
    adapted = readout_adapter(recurrent)
    head = detached_head_weight.detach()[:valid_vocab_size]
    kl_total = torch.zeros((), device=recurrent.device, dtype=torch.float32)
    overlap_total = torch.zeros_like(kl_total)
    for start in range(0, valid, config.vocab_chunk_tokens):
        stop = min(start + config.vocab_chunk_tokens, valid)
        teacher_logits = F.linear(exact[start:stop], head)
        student_logits = F.linear(adapted[start:stop], head)
        teacher_log_probs = F.log_softmax(teacher_logits.float(), dim=-1)
        student_log_probs = F.log_softmax(student_logits.float(), dim=-1)
        teacher_probs = teacher_log_probs.exp()
        student_probs = student_log_probs.exp()
        kl_total = kl_total + F.kl_div(
            student_log_probs,
            teacher_log_probs,
            log_target=True,
            reduction="sum",
        )
        overlap_total = overlap_total + torch.minimum(
            teacher_probs, student_probs
        ).sum()
    kl = kl_total / valid
    probability_overlap = overlap_total / valid
    overlap_loss = 1.0 - probability_overlap
    identity = (
        1.0
        - F.cosine_similarity(adapted.float(), recurrent.float(), dim=-1).mean()
    )
    # Stage 0 is deliberately a pure distribution-matching intervention.
    # Overlap and normalized identity remain detached reporting diagnostics;
    # validation above forbids assigning either an optimization coefficient.
    loss = config.lambda_kl * kl
    zero = loss.new_zeros(())
    return {
        "loss": loss,
        "kl": kl,
        "probability_overlap": probability_overlap.detach(),
        "overlap_loss": overlap_loss.detach(),
        "identity": identity.detach(),
        "mse": zero,
        "smooth_l1": zero,
        "recurrent_mse": zero,
        "recurrent_kl": kl,
        "anchor_mse": zero,
        "anchor_kl": zero,
        "adapted_states": adapted,
        "transition_positions": valid_mask.sum(),
    }


def _prefix_lengths(attention_mask: torch.Tensor) -> torch.Tensor:
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must be rank two")
    mask = attention_mask.bool()
    lengths = mask.sum(dim=1)
    expected = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0) < lengths.unsqueeze(1)
    if not torch.equal(mask, expected):
        raise ValueError("DMT requires right-padded contiguous sequences")
    return lengths


def _select_anchor_positions(
    *,
    attention_mask: torch.Tensor,
    assistant_loss_mask: torch.Tensor,
    config: DetachedDMTConfig,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    if attention_mask.shape != assistant_loss_mask.shape:
        raise ValueError("attention and assistant masks must match")
    _prefix_lengths(attention_mask)
    anchors = []
    for row in range(attention_mask.shape[0]):
        candidate = attention_mask[row].bool() & assistant_loss_mask[row].bool()
        if config.max_anchor_position is not None:
            positions = torch.arange(candidate.shape[0], device=candidate.device)
            candidate = candidate & positions.le(config.max_anchor_position)
        if config.rollout_mode == "teacher_forced":
            run = attention_mask[row].bool() & assistant_loss_mask[row].bool()
            width = config.rollout_horizon + 1
            required = torch.zeros_like(candidate)
            if run.numel() >= width:
                required[: run.numel() - width + 1] = run.unfold(
                    dimension=0, size=width, step=1
                ).all(dim=-1)
            candidate = candidate & required
        choices = torch.nonzero(candidate, as_tuple=False).flatten()
        if choices.numel() == 0:
            raise RuntimeError(
                f"row {row} has no valid {config.rollout_mode} DMT anchor"
            )
        draw = torch.randint(
            choices.numel(),
            (),
            device=choices.device,
            generator=generator,
        )
        anchors.append(choices[draw])
    return torch.stack(anchors)


def _pad_rows(
    rows: Sequence[torch.Tensor], pad_token_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if not rows:
        raise ValueError("cannot pad zero rows")
    width = max(int(row.numel()) for row in rows)
    output = torch.full(
        (len(rows), width),
        int(pad_token_id),
        dtype=rows[0].dtype,
        device=rows[0].device,
    )
    mask = torch.zeros((len(rows), width), dtype=torch.bool, device=rows[0].device)
    for index, row in enumerate(rows):
        output[index, : row.numel()] = row
        mask[index, : row.numel()] = True
    return output, mask


def _target_hidden(model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    backbone = getattr(model, "model", None)
    if backbone is None:
        raise TypeError("target model must expose its decoder backbone as model.model")
    outputs = backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    hidden = getattr(outputs, "last_hidden_state", None)
    if hidden is None:
        raise RuntimeError("target backbone returned no last_hidden_state")
    return hidden


def _target_prefix_with_cache(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, Any]:
    """Run one frozen prefix prefill and retain its KV cache for exact replay."""

    backbone = getattr(model, "model", None)
    if backbone is None:
        raise TypeError("target model must expose its decoder backbone as model.model")
    outputs = backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
    )
    hidden = getattr(outputs, "last_hidden_state", None)
    cache = getattr(outputs, "past_key_values", None)
    if hidden is None or cache is None:
        raise RuntimeError("target backbone returned no hidden state or KV cache")
    return hidden, cache


def _target_continuation_hidden(
    model: nn.Module,
    *,
    input_ids: torch.Tensor,
    prefix_attention_mask: torch.Tensor,
    continuation_attention_mask: torch.Tensor,
    prefix_lengths: torch.Tensor,
    past_key_values: Any,
) -> torch.Tensor:
    """Exact-replay a dense continuation against a previously computed prefix."""

    backbone = getattr(model, "model", None)
    if backbone is None:
        raise TypeError("target model must expose its decoder backbone as model.model")
    if input_ids.shape != continuation_attention_mask.shape:
        raise ValueError("continuation ids and mask must match")
    if prefix_attention_mask.shape[0] != input_ids.shape[0]:
        raise ValueError("prefix and continuation batch sizes differ")
    horizon = input_ids.shape[1]
    continuation_positions = torch.arange(
        horizon, device=input_ids.device, dtype=torch.long
    ).unsqueeze(0)
    position_ids = prefix_lengths.to(dtype=torch.long).unsqueeze(1) + continuation_positions
    prefix_width = prefix_attention_mask.shape[1]
    cache_position = torch.arange(
        prefix_width,
        prefix_width + horizon,
        device=input_ids.device,
        dtype=torch.long,
    )
    full_attention_mask = torch.cat(
        (prefix_attention_mask, continuation_attention_mask), dim=1
    )
    outputs = backbone(
        input_ids=input_ids,
        attention_mask=full_attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        cache_position=cache_position,
        use_cache=False,
        return_dict=True,
    )
    hidden = getattr(outputs, "last_hidden_state", None)
    if hidden is None:
        raise RuntimeError("target continuation replay returned no last_hidden_state")
    return hidden


@torch.no_grad()
def build_detached_dmt_batch(
    *,
    model: nn.Module,
    dynamics: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    assistant_loss_mask: torch.Tensor,
    pad_token_id: int,
    eos_token_id: Optional[int],
    valid_vocab_size: Optional[int] = None,
    eos_token_ids: Optional[Sequence[int]] = None,
    config: DetachedDMTConfig,
    readout_adapter: Optional[nn.Module] = None,
    generator: Optional[torch.Generator] = None,
    branches_per_anchor: int = 1,
    cached_exact_replay: bool = False,
) -> DetachedDMTBatch:
    """Create teacher-forced or counterfactual exact-replay DMT labels."""

    config.validate()
    if branches_per_anchor < 1:
        raise ValueError("branches_per_anchor must be positive")
    if config.rollout_mode != "recurrent_sampled" and (
        branches_per_anchor != 1 or cached_exact_replay
    ):
        raise ValueError(
            "branch replication and cached exact replay require recurrent sampling"
        )
    if branches_per_anchor != 1 and not cached_exact_replay:
        raise ValueError("multiple branches require cached exact replay")
    assert_target_frozen(model)
    if config.loss_variant == COUNTERFACTUAL_READOUT_VARIANT:
        if readout_adapter is None:
            raise ValueError("counterfactual readout batch requires readout_adapter")
    elif readout_adapter is not None:
        raise ValueError("readout_adapter is only valid for counterfactual readout")
    if input_ids.shape != attention_mask.shape or input_ids.shape != assistant_loss_mask.shape:
        raise ValueError("ids and masks must have identical shapes")
    anchors = _select_anchor_positions(
        attention_mask=attention_mask,
        assistant_loss_mask=assistant_loss_mask,
        config=config,
        generator=generator,
    )
    embedding_weight = model.get_input_embeddings().weight.detach()
    head_weight = model.lm_head.weight.detach()

    if config.rollout_mode == "teacher_forced":
        rows = [
            input_ids[row, : int(anchor.item()) + config.rollout_horizon + 1]
            for row, anchor in enumerate(anchors)
        ]
        replay_ids, replay_mask = _pad_rows(rows, pad_token_id)
        exact_hidden = _target_hidden(model, replay_ids, replay_mask)
        anchor_states = torch.stack(
            [exact_hidden[row, int(anchor.item())] for row, anchor in enumerate(anchors)]
        )
        token_ids = torch.stack(
            [
                input_ids[
                    row,
                    int(anchor.item()) + 1 : int(anchor.item()) + config.rollout_horizon + 1,
                ]
                for row, anchor in enumerate(anchors)
            ]
        )
        rollout = teacher_forced_rollout(
            dynamics=dynamics,
            anchor_states=anchor_states,
            token_ids=token_ids,
            embedding_weight=embedding_weight,
        )
        exact_next = torch.stack(
            [
                exact_hidden[
                    row,
                    int(anchor.item()) + 1 : int(anchor.item()) + config.rollout_horizon + 1,
                ]
                for row, anchor in enumerate(anchors)
            ]
        )
        if config.state_input_mode == "exact":
            training_current_states = torch.stack(
                [
                    exact_hidden[
                        row,
                        int(anchor.item()) : int(anchor.item()) + config.rollout_horizon,
                    ]
                    for row, anchor in enumerate(anchors)
                ]
            )
        else:
            training_current_states = rollout.current_states
        exact_current = torch.stack(
            [
                exact_hidden[
                    row,
                    int(anchor.item()) : int(anchor.item()) + config.rollout_horizon,
                ]
                for row, anchor in enumerate(anchors)
            ]
        )
        target_forward_tokens = int(replay_mask.sum().item())
        target_padded_forward_tokens = int(replay_ids.numel())
    else:
        prefix_rows = [
            input_ids[row, : int(anchor.item()) + 1]
            for row, anchor in enumerate(anchors)
        ]
        prefix_ids, prefix_mask = _pad_rows(prefix_rows, pad_token_id)
        if cached_exact_replay:
            prefix_hidden, prefix_cache = _target_prefix_with_cache(
                model, prefix_ids, prefix_mask
            )
        else:
            prefix_hidden = _target_hidden(model, prefix_ids, prefix_mask)
            prefix_cache = None
        anchor_states = torch.stack(
            [
                prefix_hidden[row, int(prefix_rows[row].numel()) - 1]
                for row in range(len(prefix_rows))
            ]
        )
        if branches_per_anchor > 1:
            anchor_states = anchor_states.repeat_interleave(
                branches_per_anchor, dim=0
            )
            anchors = anchors.repeat_interleave(branches_per_anchor, dim=0)
        rollout = recurrent_sample_rollout(
            dynamics=dynamics,
            anchor_states=anchor_states,
            embedding_weight=embedding_weight,
            head_weight=head_weight,
            horizon=config.rollout_horizon,
            temperature=config.temperature,
            top_k=config.top_k,
            valid_vocab_size=valid_vocab_size,
            eos_token_id=eos_token_id,
            eos_token_ids=eos_token_ids,
            pad_token_id=pad_token_id,
            readout_adapter=readout_adapter,
            generator=generator,
        )
        if cached_exact_replay:
            expanded_prefix_ids = prefix_ids.repeat_interleave(
                branches_per_anchor, dim=0
            )
            expanded_prefix_mask = prefix_mask.repeat_interleave(
                branches_per_anchor, dim=0
            )
            prefix_lengths = prefix_mask.sum(dim=1).repeat_interleave(
                branches_per_anchor, dim=0
            )
            if branches_per_anchor > 1:
                repeat_cache = getattr(prefix_cache, "batch_repeat_interleave", None)
                if repeat_cache is None:
                    raise RuntimeError(
                        "target KV cache cannot repeat branches in the batch dimension"
                    )
                repeat_cache(branches_per_anchor)
            exact_hidden = _target_continuation_hidden(
                model,
                input_ids=rollout.token_ids,
                prefix_attention_mask=expanded_prefix_mask,
                continuation_attention_mask=rollout.valid_mask,
                prefix_lengths=prefix_lengths,
                past_key_values=prefix_cache,
            )
            exact_next = exact_hidden
            exact_current = torch.cat(
                (anchor_states.unsqueeze(1), exact_hidden[:, :-1]), dim=1
            )
            replay_ids = torch.cat(
                (expanded_prefix_ids, rollout.token_ids), dim=1
            )
            replay_mask = torch.cat(
                (expanded_prefix_mask, rollout.valid_mask), dim=1
            )
            target_forward_tokens = int(
                prefix_mask.sum().item() + rollout.valid_mask.sum().item()
            )
            target_padded_forward_tokens = int(
                prefix_ids.numel() + rollout.token_ids.numel()
            )
        else:
            replay_rows = []
            valid_lengths = rollout.valid_mask.sum(dim=1)
            for row, prefix in enumerate(prefix_rows):
                continuation = rollout.token_ids[row, : int(valid_lengths[row].item())]
                replay_rows.append(torch.cat((prefix, continuation), dim=0))
            replay_ids, replay_mask = _pad_rows(replay_rows, pad_token_id)
            exact_hidden = _target_hidden(model, replay_ids, replay_mask)
            exact_next = torch.zeros_like(rollout.current_states)
            exact_current = torch.zeros_like(rollout.current_states)
            for row, prefix in enumerate(prefix_rows):
                count = int(valid_lengths[row].item())
                start = int(prefix.numel())
                exact_next[row, :count] = exact_hidden[row, start : start + count]
                exact_current[row, 0] = prefix_hidden[row, int(prefix.numel()) - 1]
                if count > 1:
                    exact_current[row, 1:count] = exact_hidden[
                        row, start : start + count - 1
                    ]
            target_forward_tokens = int(
                prefix_mask.sum().item() + replay_mask.sum().item()
            )
            target_padded_forward_tokens = int(
                prefix_ids.numel() + replay_ids.numel()
            )
        training_current_states = rollout.current_states

    token_embeddings = F.embedding(rollout.token_ids, embedding_weight).detach()
    # Official DMT records the pre-update state at t=0,k,2k. The t=0 anchor is
    # exact and supplies no useful dynamics update. In one-step label form, the
    # corresponding useful labels are the transitions that produce states
    # k,2k,...: transition indices k-1,2k-1,... .
    stride_mask = torch.zeros_like(rollout.valid_mask)
    if config.loss_variant in (
        *OFFICIAL_DETACHED_MSE_VARIANTS,
        ANCHOR_HEAD_REPAIR_VARIANT,
    ):
        # Literal useful labels from the published Td=256,k=16 run are states
        # 16,32,...,240. State 0 is the exact anchor and state 256 is not in the
        # official subsampled roster.
        first_transition = config.loss_stride - 1
        transition_stop = config.rollout_horizon - config.loss_stride
        stride_mask[:, first_transition:transition_stop:config.loss_stride] = True
        if config.loss_variant == ANCHOR_HEAD_REPAIR_VARIANT:
            # Protect the exact-anchor behavior with one ordinary one-step
            # transition per row in addition to recurrent depths 16,...,240.
            stride_mask[:, 0] = True
    else:
        stride_mask[:, config.loss_stride - 1 :: config.loss_stride] = True
        if config.loss_variant == COUNTERFACTUAL_READOUT_VARIANT:
            # Depth zero is the exact anchor.  It is decoded through the raw
            # head and is neither adapted nor used as a readout training label.
            stride_mask[:, 0] = False
    return DetachedDMTBatch(
        current_states=training_current_states.detach(),
        token_ids=rollout.token_ids.detach(),
        token_embeddings=token_embeddings,
        exact_next_states=exact_next.detach(),
        exact_current_states=exact_current.detach(),
        valid_mask=(rollout.valid_mask & stride_mask).detach(),
        anchor_positions=anchors.detach(),
        replay_input_ids=replay_ids.detach(),
        replay_attention_mask=replay_mask.detach(),
        target_forward_tokens=target_forward_tokens,
        target_padded_forward_tokens=target_padded_forward_tokens,
    )
