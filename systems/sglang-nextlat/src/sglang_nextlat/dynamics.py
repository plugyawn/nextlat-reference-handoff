"""Exact residual NextLat transition used by the Qwen3-1.7B experiment."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


def projection_size(hidden_size: int, projection_factor: float = 1.6) -> int:
    """Match the official 128-wide rounded projection rule exactly."""

    if hidden_size <= 0:
        raise ValueError(f"hidden_size must be positive, got {hidden_size}")
    if projection_factor <= 0:
        raise ValueError(
            f"projection_factor must be positive, got {projection_factor}"
        )
    input_size = 2 * hidden_size
    width = 128 * round(projection_factor * input_size / 128)
    if width <= 0:
        raise ValueError("projection_size must be positive")
    return width


def expected_state_shapes(
    hidden_size: int, projection_factor: float = 1.6
) -> dict[str, tuple[int, ...]]:
    width = projection_size(hidden_size, projection_factor)
    input_size = 2 * hidden_size
    return {
        "norm.weight": (input_size,),
        "mlp.0.weight": (width, input_size),
        "mlp.2.weight": (width, width),
        "mlp.4.weight": (hidden_size, width),
    }


class BiaslessRMSNorm(nn.Module):
    """The official ``LayerNorm(..., bias=False)`` path is RMSNorm."""

    def __init__(self, width: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.width = width
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(inputs, (self.width,), self.weight, self.eps)


class NextLatDynamics(nn.Module):
    """Official residual three-linear GELU transition ``p_psi``."""

    def __init__(
        self,
        hidden_size: int,
        projection_factor: float = 1.6,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        input_size = 2 * hidden_size
        projected = projection_size(hidden_size, projection_factor)

        self.hidden_size = hidden_size
        self.projection_factor = projection_factor
        self.projection_size = projected
        self.hidden_state_dropout = (
            nn.Dropout(dropout) if dropout else nn.Identity()
        )
        self.norm = BiaslessRMSNorm(input_size)
        self.mlp = nn.Sequential(
            nn.Linear(input_size, projected, bias=False),
            nn.GELU(),
            nn.Linear(projected, projected, bias=False),
            nn.GELU(),
            nn.Linear(projected, hidden_size, bias=False),
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
                "current_states and next_token_embeddings must have identical "
                f"shapes; got {current_states.shape} and "
                f"{next_token_embeddings.shape}"
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


def validate_state_dict(
    state: Mapping[str, torch.Tensor],
    *,
    hidden_size: int,
    projection_factor: float,
    expected_dtype: torch.dtype,
) -> None:
    """Reject every key, shape, or storage-dtype drift before loading."""

    expected = expected_state_shapes(hidden_size, projection_factor)
    observed_keys = set(state)
    expected_keys = set(expected)
    if observed_keys != expected_keys:
        missing = sorted(expected_keys - observed_keys)
        extra = sorted(observed_keys - expected_keys)
        raise ValueError(
            f"dynamics tensor keys do not match: missing={missing}, extra={extra}"
        )
    for name, shape in expected.items():
        tensor = state[name]
        if tuple(tensor.shape) != shape:
            raise ValueError(
                f"dynamics tensor {name} has shape {tuple(tensor.shape)}, "
                f"expected {shape}"
            )
        if tensor.dtype != expected_dtype:
            raise ValueError(
                f"dynamics tensor {name} has dtype {tensor.dtype}, "
                f"expected {expected_dtype}"
            )

