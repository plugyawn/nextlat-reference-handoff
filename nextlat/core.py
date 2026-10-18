"""Exact recurrent NextLat objective for the Qwen3-1.7B experiment.

Indexing follows the pinned official implementation at commit 3770be6:

* h[t] and embedding(x[t+1]) predict h_hat[t+1]
* SmoothL1 targets h[t+1] with stop-gradient
* KL compares the detached teacher distribution at h[t+1] with the student
  distribution at h_hat[t+1], both through a detached output-head weight
* ordinary causal CE remains trainable through the output head

Token-level CE and KL are assistant-masked. SmoothL1 covers prompt and response
transitions, as specified in Appendix C of the paper. This primary path does not
pack documents, so the attention mask is also the document-boundary mask.

For a horizon greater than one, the predicted state is fed back recursively and
the hidden/KL losses are averaged uniformly over horizons, matching the pinned
official executable. The executable upstream drops the final predicted state at
each horizon from KL because there is no observed next-token target for it.
Literal paper pseudocode can be read as keeping that terminal state; this
implementation deliberately treats the pinned upstream code as the parity
authority and records the distinction in the experiment plan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class NextLatLossConfig:
    lambda_h: float = 1.0
    lambda_kl: float = 1.0
    vocab_chunk_tokens: int = 32
    horizon: int = 1
    kl_implementation: str = "autograd"
    anchor_noise_relative: float = 0.0


def noisy_training_anchor(
    hidden_states: torch.Tensor, *, dynamics: nn.Module, relative: float
) -> torch.Tensor:
    """Perturb the initial recurrent state once; leave exact teachers clean.

    Uniform half-width is relative times each token's detached hidden RMS.
    Noise acts on the complete state, including the residual path. The exact
    anchor remains gradient-connected with identity derivative. Evaluation
    and the zero-noise control neither change values nor consume RNG state.
    CUDA RNG is owned and checkpointed by the existing trainer; data order
    uses its independent seeded DistributedSampler.
    """
    if not 0.0 <= relative <= 0.1:
        raise ValueError("anchor_noise_relative must be in [0, 0.1]")
    if relative == 0.0 or not dynamics.training:
        return hidden_states
    rms = hidden_states.detach().float().square().mean(-1, keepdim=True).sqrt()
    perturbation = torch.empty_like(hidden_states, dtype=torch.float32).uniform_(-1, 1)
    return hidden_states + (perturbation * (relative * rms)).to(hidden_states.dtype)


class BiaslessRMSNorm(nn.Module):
    """The official ``LayerNorm(..., bias=False)`` path is actually RMSNorm."""

    def __init__(self, width: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.width = width
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (self.width,), self.weight, self.eps)


class NextLatDynamics(nn.Module):
    """Official residual three-linear GELU transition p_psi."""

    def __init__(
        self,
        hidden_size: int,
        projection_factor: float = 1.6,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        input_size = 2 * hidden_size
        projection_size = 128 * round(projection_factor * input_size / 128)
        if projection_size <= 0:
            raise ValueError("projection_size must be positive")

        self.hidden_size = hidden_size
        self.projection_factor = projection_factor
        self.projection_size = projection_size
        self.hidden_state_dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.norm = BiaslessRMSNorm(input_size)
        self.mlp = nn.Sequential(
            nn.Linear(input_size, projection_size, bias=False),
            nn.GELU(),
            nn.Linear(projection_size, projection_size, bias=False),
            nn.GELU(),
            nn.Linear(projection_size, hidden_size, bias=False),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        current_states: torch.Tensor,
        next_token_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        if current_states.shape != next_token_embeddings.shape:
            raise ValueError(
                "current_states and next_token_embeddings must have identical shapes; "
                f"got {current_states.shape} and {next_token_embeddings.shape}"
            )
        if current_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"last dimension must be hidden_size={self.hidden_size}, "
                f"got {current_states.shape[-1]}"
            )
        residual = current_states
        states = self.hidden_state_dropout(current_states)
        joined = torch.cat((next_token_embeddings, states), dim=-1)
        return residual + self.mlp(self.norm(joined))


def _require_nonempty(mask: torch.Tensor, name: str) -> int:
    count = int(mask.sum().detach().item())
    if count == 0:
        raise ValueError(f"{name} has zero valid positions")
    return count


def _masked_cross_entropy(
    states: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    head_weight: torch.Tensor,
    chunk_tokens: int,
) -> torch.Tensor:
    valid = _require_nonempty(mask, "CE mask")
    selected_states = states[mask]
    selected_targets = targets[mask]
    total = torch.zeros((), device=states.device, dtype=torch.float32)
    for start in range(0, valid, chunk_tokens):
        stop = min(start + chunk_tokens, valid)
        logits = F.linear(selected_states[start:stop], head_weight)
        total = total + F.cross_entropy(
            logits.float(), selected_targets[start:stop], reduction="sum"
        )
    return total / valid


def _masked_teacher_student_kl(
    teacher_states: torch.Tensor,
    student_states: torch.Tensor,
    mask: torch.Tensor,
    detached_head_weight: torch.Tensor,
    chunk_tokens: int,
) -> torch.Tensor:
    """Mean tokenwise KL(teacher || student), vocabulary summed exactly."""

    valid = _require_nonempty(mask, "KL mask")
    teacher_selected = teacher_states[mask]
    student_selected = student_states[mask]
    total = torch.zeros((), device=student_states.device, dtype=torch.float32)
    for start in range(0, valid, chunk_tokens):
        stop = min(start + chunk_tokens, valid)
        teacher_logits = F.linear(
            teacher_selected[start:stop].detach(), detached_head_weight
        )
        student_logits = F.linear(
            student_selected[start:stop], detached_head_weight
        )
        teacher_log_probs = F.log_softmax(teacher_logits.float(), dim=-1)
        student_log_probs = F.log_softmax(student_logits.float(), dim=-1)
        total = total + F.kl_div(
            student_log_probs,
            teacher_log_probs,
            log_target=True,
            reduction="sum",
        )
    return total / valid


class _TeacherStudentKLForwardGradient(torch.autograd.Function):
    """Exact first-order KL without vocabulary-sized saved activations."""

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        teacher_states: torch.Tensor,
        student_states: torch.Tensor,
        detached_head_weight: torch.Tensor,
    ) -> torch.Tensor:
        teacher_logits = F.linear(teacher_states, detached_head_weight)
        student_logits = F.linear(student_states, detached_head_weight)
        teacher_log_probs = F.log_softmax(teacher_logits.float(), dim=-1)
        student_log_probs = F.log_softmax(student_logits.float(), dim=-1)
        total = F.kl_div(
            student_log_probs,
            teacher_log_probs,
            log_target=True,
            reduction="sum",
        )

        # Match log_softmax backward even when finite-precision teacher
        # probabilities do not sum to exactly one.
        teacher_probs = teacher_log_probs.exp()
        probability_delta = (
            student_log_probs.exp() * teacher_probs.sum(dim=-1, keepdim=True)
            - teacher_probs
        )
        student_gradient = probability_delta.to(student_logits.dtype).matmul(
            detached_head_weight
        )
        ctx.save_for_backward(student_gradient.to(student_states.dtype))
        return total

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output: torch.Tensor):
        if torch.is_grad_enabled():
            raise RuntimeError(
                "forward-gradient KL supports first-order gradients only; "
                "higher-order backward would require saved vocabulary probabilities"
            )
        (student_gradient,) = ctx.saved_tensors
        return None, (student_gradient * grad_output).to(student_gradient.dtype), None


def _masked_teacher_student_kl_forward_gradient(
    teacher_states: torch.Tensor,
    student_states: torch.Tensor,
    mask: torch.Tensor,
    detached_head_weight: torch.Tensor,
    chunk_tokens: int,
) -> torch.Tensor:
    """Mean tokenwise KL with an analytical hidden-state backward."""

    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    valid = _require_nonempty(mask, "KL mask")
    teacher_selected = teacher_states[mask]
    student_selected = student_states[mask]
    head = detached_head_weight.detach()
    total = torch.zeros((), device=student_states.device, dtype=torch.float32)
    for start in range(0, valid, chunk_tokens):
        stop = min(start + chunk_tokens, valid)
        total = total + _TeacherStudentKLForwardGradient.apply(
            teacher_selected[start:stop].detach(),
            student_selected[start:stop],
            head,
        )
    return total / valid


def _masked_smooth_l1(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    valid = _require_nonempty(mask, "SmoothL1 mask")
    elementwise = F.smooth_l1_loss(
        predicted, target.detach(), reduction="none"
    )
    weights = mask.unsqueeze(-1).to(dtype=elementwise.dtype)
    return (elementwise * weights).sum() / (valid * predicted.shape[-1])


def nextlat_position_masks(
    attention_mask: torch.Tensor,
    assistant_loss_mask: torch.Tensor,
    horizon: int,
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    """Return the exact CE, transition, and KL masks used by the objective."""

    if attention_mask.shape != assistant_loss_mask.shape:
        raise ValueError("attention_mask and assistant_loss_mask must match")
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask and assistant_loss_mask must be rank two")
    if horizon < 1:
        raise ValueError("NextLat horizon must be positive")
    length = attention_mask.shape[1]
    if length < horizon + 2:
        raise ValueError(
            "NextLat CE+KL requires at least horizon+2 tokens; "
            f"got length={length} horizon={horizon}"
        )

    attention = attention_mask.bool()
    assistant = assistant_loss_mask.bool()
    ce_mask = attention[:, :-1] & attention[:, 1:] & assistant[:, 1:]
    transition_masks: list[torch.Tensor] = []
    kl_masks: list[torch.Tensor] = []
    for step in range(1, horizon + 1):
        transition_width = length - step
        transition_masks.append(
            torch.stack(
                [
                    attention[:, offset : offset + transition_width]
                    for offset in range(step + 1)
                ],
                dim=0,
            ).all(dim=0)
        )
        kl_width = length - step - 1
        kl_masks.append(
            torch.stack(
                [
                    attention[:, offset : offset + kl_width]
                    for offset in range(step + 2)
                ],
                dim=0,
            ).all(dim=0)
            & assistant[:, step + 1 :]
        )
    return ce_mask, transition_masks, kl_masks


def nextlat_position_counts(
    attention_mask: torch.Tensor,
    assistant_loss_mask: torch.Tensor,
    horizon: int,
) -> torch.Tensor:
    """Count objective positions without constructing a model forward graph."""

    ce_mask, transition_masks, kl_masks = nextlat_position_masks(
        attention_mask, assistant_loss_mask, horizon
    )
    return torch.stack(
        [
            ce_mask.sum(),
            *(mask.sum() for mask in transition_masks),
            *(mask.sum() for mask in kl_masks),
        ]
    )


def compute_nextlat_losses(
    *,
    hidden_states: torch.Tensor,
    token_embeddings: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    assistant_loss_mask: torch.Tensor,
    head_weight: torch.Tensor,
    dynamics: NextLatDynamics,
    config: NextLatLossConfig,
) -> Dict[str, Any]:
    """Compute the paper's recurrent joint loss with explicit masking and stops."""

    if hidden_states.ndim != 3 or hidden_states.shape != token_embeddings.shape:
        raise ValueError("hidden_states and token_embeddings must be matching (B,T,D)")
    if input_ids.shape != hidden_states.shape[:2]:
        raise ValueError("input_ids must match the (B,T) hidden-state prefix")
    if attention_mask.shape != input_ids.shape or assistant_loss_mask.shape != input_ids.shape:
        raise ValueError("attention_mask and assistant_loss_mask must match input_ids")
    if config.vocab_chunk_tokens <= 0:
        raise ValueError("vocab_chunk_tokens must be positive")

    # The shared-teacher backend needs the CE projection and all recurrent
    # horizons in one schedule.  Keep this import local so the default
    # authority module has no import cycle and remains the unchanged path.
    if config.kl_implementation in (
        "shared_teacher_forward_gradient",
        "shared_teacher_horizon_stacked",
        "shared_teacher_horizon_stacked_triton",
    ):
        from .shared_teacher_loss import compute_nextlat_losses_shared_teacher_d4

        return compute_nextlat_losses_shared_teacher_d4(
            hidden_states=hidden_states,
            token_embeddings=token_embeddings,
            input_ids=input_ids,
            attention_mask=attention_mask,
            assistant_loss_mask=assistant_loss_mask,
            head_weight=head_weight,
            dynamics=dynamics,
            config=config,
        )

    ce_mask, _, _ = nextlat_position_masks(
        attention_mask, assistant_loss_mask, config.horizon
    )

    # Ordinary next-token CE: h[t] -> x[t+1].
    ce = _masked_cross_entropy(
        hidden_states[:, :-1],
        input_ids[:, 1:],
        ce_mask,
        head_weight,
        config.vocab_chunk_tokens,
    )

    auxiliary = compute_nextlat_auxiliary_losses(
        hidden_states=hidden_states,
        token_embeddings=token_embeddings,
        attention_mask=attention_mask,
        assistant_loss_mask=assistant_loss_mask,
        head_weight=head_weight,
        dynamics=dynamics,
        config=config,
    )
    smooth_l1 = auxiliary["smooth_l1"]
    kl = auxiliary["kl"]

    total = ce + config.lambda_h * smooth_l1 + config.lambda_kl * kl
    return {
        "loss": total,
        "ce": ce,
        **auxiliary,
        "ce_positions": ce_mask.sum(),
    }


def compute_nextlat_auxiliary_losses(
    *,
    hidden_states: torch.Tensor,
    token_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    assistant_loss_mask: torch.Tensor,
    head_weight: torch.Tensor,
    dynamics: NextLatDynamics,
    config: NextLatLossConfig,
) -> Dict[str, Any]:
    """Compute the official recursive SmoothL1+KL terms without causal CE.

    This is the same recurrence and reduction used by
    :func:`compute_nextlat_losses`.  Keeping it as a first-class entry point
    lets a frozen-target sidecar run reuse the official objective without
    paying for, or accidentally optimizing, an irrelevant CE term.
    """

    if hidden_states.ndim != 3 or hidden_states.shape != token_embeddings.shape:
        raise ValueError("hidden_states and token_embeddings must be matching (B,T,D)")
    if attention_mask.shape != hidden_states.shape[:2]:
        raise ValueError("attention_mask must match hidden-state batch and time")
    if assistant_loss_mask.shape != hidden_states.shape[:2]:
        raise ValueError("assistant_loss_mask must match hidden-state batch and time")
    if config.vocab_chunk_tokens <= 0:
        raise ValueError("vocab_chunk_tokens must be positive")
    if config.kl_implementation not in ("autograd", "forward_gradient"):
        raise ValueError(
            "kl_implementation must be 'autograd' or 'forward_gradient'"
        )

    _, transition_masks, kl_masks = nextlat_position_masks(
        attention_mask, assistant_loss_mask, config.horizon
    )
    predicted_by_horizon = []
    smooth_by_horizon = []
    kl_by_horizon = []
    transition_counts = []
    kl_counts = []
    recurrent = noisy_training_anchor(
        hidden_states, dynamics=dynamics, relative=config.anchor_noise_relative
    )

    # Pinned upstream recursive path:
    #   (h[t], x[t+1]) -> h_hat[t+1]
    #   (h_hat[t+1], x[t+2]) -> h_hat[t+2]
    # Recurrent is intentionally not detached between loop iterations: depth
    # d receives temporal credit through all d applications of p_psi.
    for step in range(1, config.horizon + 1):
        recurrent = dynamics(recurrent[:, :-1], token_embeddings[:, step:])
        predicted_by_horizon.append(recurrent)

        transition_mask = transition_masks[step - 1]
        smooth_by_horizon.append(
            _masked_smooth_l1(
                recurrent,
                hidden_states[:, step:],
                transition_mask,
            )
        )
        transition_counts.append(transition_mask.sum())

        # A state at t+step predicts x[t+step+1], so the final recurrent state
        # at every horizon has no KL target. Teacher states and output head are
        # detached inside the helper.
        kl_mask = kl_masks[step - 1]
        kl_helper = (
            _masked_teacher_student_kl_forward_gradient
            if config.kl_implementation == "forward_gradient"
            else _masked_teacher_student_kl
        )
        kl_by_horizon.append(
            kl_helper(
                hidden_states[:, step:-1],
                recurrent[:, :-1],
                kl_mask,
                head_weight.detach(),
                config.vocab_chunk_tokens,
            )
        )
        kl_counts.append(kl_mask.sum())

    smooth_l1_by_horizon = torch.stack(smooth_by_horizon)
    kl_by_horizon_tensor = torch.stack(kl_by_horizon)
    transition_positions_by_horizon = torch.stack(transition_counts)
    kl_positions_by_horizon = torch.stack(kl_counts)
    smooth_l1 = smooth_l1_by_horizon.mean()
    kl = kl_by_horizon_tensor.mean()
    return {
        "smooth_l1": smooth_l1,
        "kl": kl,
        # Preserve the d=1 compatibility key used by smoke/evaluation code.
        "predicted_next_states": predicted_by_horizon[0],
        "predicted_states_by_horizon": predicted_by_horizon,
        "smooth_l1_by_horizon": smooth_l1_by_horizon,
        "kl_by_horizon": kl_by_horizon_tensor,
        "transition_positions": transition_positions_by_horizon.sum(),
        "kl_positions": kl_positions_by_horizon.sum(),
        "transition_positions_by_horizon": transition_positions_by_horizon,
        "kl_positions_by_horizon": kl_positions_by_horizon,
    }


class NextLatQwen(nn.Module):
    """Thin joint wrapper around one Hugging Face Qwen3 causal LM."""

    def __init__(
        self,
        model: nn.Module,
        dynamics: NextLatDynamics,
        loss_config: NextLatLossConfig,
    ) -> None:
        super().__init__()
        hidden_size = int(model.config.hidden_size)
        if hidden_size != dynamics.hidden_size:
            raise ValueError(
                f"model hidden_size={hidden_size} != dynamics hidden_size={dynamics.hidden_size}"
            )
        self.model = model
        self.dynamics = dynamics
        self.loss_config = loss_config

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        assistant_loss_mask: torch.Tensor,
        *,
        return_training_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        token_embeddings = self.model.get_input_embeddings()(input_ids)
        outputs = self.model.model(
            inputs_embeds=token_embeddings,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        losses = compute_nextlat_losses(
            hidden_states=outputs.last_hidden_state,
            token_embeddings=token_embeddings,
            input_ids=input_ids,
            attention_mask=attention_mask,
            assistant_loss_mask=assistant_loss_mask,
            head_weight=self.model.lm_head.weight,
            dynamics=self.dynamics,
            config=self.loss_config,
        )
        if return_training_states:
            # The hybrid correction path detaches these tensors before using
            # them. Returning the tensors from this invocation lets it reuse
            # the exact states without a second Qwen forward.
            losses["exact_hidden_states"] = outputs.last_hidden_state
            losses["token_embeddings"] = token_embeddings
        return losses
