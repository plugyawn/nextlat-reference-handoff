"""Diagnostic-only LM-head precision comparisons for parity reduction."""

from __future__ import annotations

from typing import Any

import torch


NEAR_TIE_LOGIT_GAP = 0.25
_TOP_K = 4


def _dtype_name(value: torch.Tensor) -> str:
    return str(value.dtype).removeprefix("torch.")


@torch.no_grad()
def near_tie_lm_head_probe(
    *,
    raw_logits: torch.Tensor,
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    vocab_size: int,
) -> dict[str, Any]:
    """Compare batched logits with isolated BF16 and candidate-only FP32 scores.

    This intentionally computes no full-vocabulary FP32 projection.  It casts only
    the four candidates selected by each raw near-tie row, so the diagnostic does
    not materialize an FP32 copy of the full Qwen LM head.
    """

    if raw_logits.ndim != 2 or hidden_states.ndim != 2 or lm_head_weight.ndim != 2:
        raise ValueError("LM-head precision probe expects rank-two tensors")
    if raw_logits.shape[0] != hidden_states.shape[0]:
        raise ValueError("raw logits and hidden states must have the same row count")
    if hidden_states.shape[1] != lm_head_weight.shape[1]:
        raise ValueError("hidden width differs from the LM-head weight")
    if not _TOP_K <= vocab_size <= raw_logits.shape[1]:
        raise ValueError("vocab_size is outside the raw-logit width")
    if vocab_size > lm_head_weight.shape[0]:
        raise ValueError("vocab_size is outside the LM-head weight height")
    if not (raw_logits.device == hidden_states.device == lm_head_weight.device):
        raise ValueError("LM-head precision probe tensors must share one device")
    if hidden_states.dtype != torch.bfloat16 or lm_head_weight.dtype != torch.bfloat16:
        raise ValueError(
            "isolated LM-head replay requires BF16 hidden states and weights"
        )

    raw_top_values, raw_top_ids = torch.topk(
        raw_logits[:, :vocab_size].float(), k=_TOP_K, dim=-1
    )
    raw_margins = raw_top_values[:, 0] - raw_top_values[:, 1]
    selected_rows = torch.nonzero(
        raw_margins <= NEAR_TIE_LOGIT_GAP, as_tuple=False
    ).flatten()

    rows: list[dict[str, Any]] = []
    for compact_row_index in selected_rows.detach().cpu().tolist():
        candidate_ids = raw_top_ids[compact_row_index]
        row_hidden = hidden_states[compact_row_index]
        candidate_weights = lm_head_weight.index_select(0, candidate_ids)
        fp32_candidate_scores = torch.mv(
            candidate_weights.float(), row_hidden.float()
        )

        # This is deliberately an isolated M=1 BF16 GEMM over the entire admitted
        # vocabulary.  It tests whether changing only the LM-head row geometry
        # changes top-token ordering relative to the existing batched projection.
        isolated_logits = torch.matmul(
            row_hidden[None, :], lm_head_weight[:vocab_size].T
        )
        if isolated_logits.dtype != torch.bfloat16:
            raise RuntimeError(
                "isolated BF16 LM-head replay returned a non-BF16 tensor"
            )
        isolated_values, isolated_ids = torch.topk(
            isolated_logits.float(), k=_TOP_K, dim=-1
        )

        rows.append(
            {
                "compact_target_row_index": int(compact_row_index),
                "raw_batched_projection": {
                    "dtype": _dtype_name(raw_logits),
                    "argmax_token_id": int(
                        torch.argmax(
                            raw_logits[compact_row_index, :vocab_size]
                        ).item()
                    ),
                    "top_token_ids": raw_top_ids[compact_row_index]
                    .detach()
                    .cpu()
                    .tolist(),
                    "top_logits": raw_top_values[compact_row_index]
                    .detach()
                    .cpu()
                    .tolist(),
                    "top1_minus_top2_logit": float(
                        raw_margins[compact_row_index].item()
                    ),
                },
                "fp32_candidate_dot_products": {
                    "hidden_source_dtype": _dtype_name(hidden_states),
                    "weight_source_dtype": _dtype_name(lm_head_weight),
                    "token_ids": candidate_ids.detach().cpu().tolist(),
                    "scores": fp32_candidate_scores.detach().cpu().tolist(),
                },
                "isolated_row_bf16_full_lm_head": {
                    "dtype": _dtype_name(isolated_logits),
                    "argmax_token_id": int(torch.argmax(isolated_logits[0]).item()),
                    "top_token_ids": isolated_ids[0].detach().cpu().tolist(),
                    "top_logits": isolated_values[0].detach().cpu().tolist(),
                    "top1_minus_top2_logit": float(
                        (isolated_values[0, 0] - isolated_values[0, 1]).item()
                    ),
                },
            }
        )

    return {
        "selection": "raw_batched_top1_minus_top2_logit_le_threshold",
        "threshold": NEAR_TIE_LOGIT_GAP,
        "target_row_count": int(raw_logits.shape[0]),
        "selected_row_count": len(rows),
        "rows": rows,
    }
