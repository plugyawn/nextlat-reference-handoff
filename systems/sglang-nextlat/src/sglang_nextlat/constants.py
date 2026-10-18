"""The single supported model and serving contract."""

from __future__ import annotations

from dataclasses import dataclass
import math

AUDITED_SGLANG_COMMIT = "baf09f39544d3cd812c77ed78132eedefaee5c5d"
TARGET_MODEL_ID = "Qwen/Qwen3-1.7B"
TARGET_BASE_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
HIDDEN_STATE_CONVENTION = "post_final_norm"

FIXED_TOPK = 1
HOT_VOCAB_SIZE = 32_768
# Qwen3-1.7B exposes 151,936 embedding/head rows, but its tokenizer can emit
# only IDs [0, 151,669). The final 267 rows are embedding padding and must have
# zero probability in lossless sampled decoding.
QWEN3_1_7B_TOKENIZER_VOCAB_SIZE = 151_669
DEFAULT_MAX_DRAFT_TOKENS = 8
MAX_DRAFT_TOKENS = 32
SERVING_ATTENTION_BACKENDS = frozenset({"flashinfer", "fa3"})


@dataclass(frozen=True)
class NextLatContract:
    """Immutable shape and provenance contract for one target model."""

    target_model_id: str
    target_base_revision: str
    hidden_size: int
    vocab_size: int
    projection_factor: float = 1.6
    dropout: float = 0.0
    hidden_state_convention: str = HIDDEN_STATE_CONVENTION
    dynamics_dtype: str = "float32"
    target_source_dtype: str = "float32"
    target_runtime_dtype: str = "bfloat16"
    sglang_commit: str = AUDITED_SGLANG_COMMIT


QWEN3_1_7B_CONTRACT = NextLatContract(
    target_model_id=TARGET_MODEL_ID,
    target_base_revision=TARGET_BASE_REVISION,
    hidden_size=2048,
    vocab_size=151_936,
)


def scaling_contract(
    target_model_id: str, target_base_revision: str, hidden_size: int,
    vocab_size: int, projection_factor: float,
) -> NextLatContract:
    """Admit only the three pinned Qwen3 targets in the existing scaling grid."""
    target = (target_model_id, target_base_revision, hidden_size, vocab_size)
    if target not in {
        ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca", 1024, 151936),
        ("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e", 2048, 151936),
        ("Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c", 2560, 151936),
    }:
        raise ValueError(f"target is outside the pinned Qwen3 scaling grid: {target!r}")
    if isinstance(projection_factor, bool) or not isinstance(projection_factor, (int, float)):
        raise ValueError("projection_factor must be a finite positive number")
    if not math.isfinite(projection_factor) or projection_factor <= 0:
        raise ValueError("projection_factor must be a finite positive number")
    return NextLatContract(
        target_model_id=target_model_id, target_base_revision=target_base_revision,
        hidden_size=hidden_size, vocab_size=vocab_size,
        projection_factor=float(projection_factor),
    )
