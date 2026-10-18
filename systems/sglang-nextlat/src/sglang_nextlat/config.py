"""Fail-closed validation for the narrow SGLang integration tier."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .beam_tree import TREE_HORIZON, TREE_NODE_BUDGET
from .checkpoint import read_sidecar_metadata
from .constants import (
    FIXED_TOPK,
    MAX_DRAFT_TOKENS,
    SERVING_ATTENTION_BACKENDS,
)
from .diagonal_decode_verify import (
    diagonal_decode_max_rows,
    diagonal_decode_verify_enabled,
    fixed_b1_variable_verify_enabled,
    fixed_b32_variable_verify_enabled,
)
from .native_ragged_verify import (
    BATCH_SIZE as NATIVE_RAGGED_BATCH_SIZE,
    native_ragged_verify_enabled,
)


TARGET_VERIFY_GRAPH_DIAGNOSTIC_ENV = "NEXTLAT_TARGET_VERIFY_GRAPH_DIAGNOSTIC"
TARGET_VERIFY_GRAPH_BATCH_SIZE_ENV = "NEXTLAT_TARGET_VERIFY_GRAPH_BATCH_SIZE"
EXPERIMENTAL_ALLOW_FP8_TARGET_ENV = "NEXTLAT_EXPERIMENTAL_ALLOW_FP8_TARGET"
TOP2_TREE_VERIFY_ENV = "NEXTLAT_TOP2_TREE_VERIFY"
TOP2_TREE_OPTIMIZED_DRAFT_ENV = "NEXTLAT_TOP2_TREE_OPTIMIZED_DRAFT"
TARGET_VERIFY_GRAPH_BATCH_SIZE = 8
TARGET_VERIFY_GRAPH_MAX_DRAFT_TOKENS = 8
TARGET_VERIFY_GRAPH_VERIFY_WIDTH = 9
TARGET_VERIFY_GRAPH_ROWS = 72


def _value(server_args: Any, name: str, default: Any = None) -> Any:
    return getattr(server_args, name, default)


def _normalized(value: Any) -> str:
    if hasattr(value, "value"):
        value = value.value
    return str(value).strip().lower()


def _phase_graph_backend(config: Any, phase: str) -> Any:
    if config is None:
        return None
    if isinstance(config, dict):
        phase_config = config.get(phase)
    else:
        phase_config = getattr(config, phase, None)
    if isinstance(phase_config, dict):
        return phase_config.get("backend")
    return getattr(phase_config, "backend", None)


def _phase_graph_value(config: Any, phase: str, name: str) -> Any:
    if config is None:
        return None
    phase_config = config.get(phase) if isinstance(config, dict) else getattr(config, phase, None)
    if isinstance(phase_config, dict):
        return phase_config.get(name)
    return getattr(phase_config, name, None)


def target_verify_graph_diagnostic_enabled() -> bool:
    value = os.environ.get(TARGET_VERIFY_GRAPH_DIAGNOSTIC_ENV)
    if value is None:
        return False
    if value != "1":
        raise ValueError(f"{TARGET_VERIFY_GRAPH_DIAGNOSTIC_ENV} must be exactly '1'")
    return True


def target_verify_graph_batch_size() -> int:
    value = os.environ.get(TARGET_VERIFY_GRAPH_BATCH_SIZE_ENV)
    if value is None:
        return TARGET_VERIFY_GRAPH_BATCH_SIZE
    if value not in {"1", "8"}:
        raise ValueError(
            f"{TARGET_VERIFY_GRAPH_BATCH_SIZE_ENV} must be exactly '1' or '8'"
        )
    return int(value)


def experimental_allow_fp8_target_enabled() -> bool:
    value = os.environ.get(EXPERIMENTAL_ALLOW_FP8_TARGET_ENV)
    if value is None:
        return False
    if value != "1":
        raise ValueError(f"{EXPERIMENTAL_ALLOW_FP8_TARGET_ENV} must be exactly '1'")
    return True


def top2_tree_verify_enabled() -> bool:
    """Whether the bounded B1 irregular-tree experiment is selected."""

    value = os.environ.get(TOP2_TREE_VERIFY_ENV)
    if value is None:
        return False
    if value != "1":
        raise ValueError(f"{TOP2_TREE_VERIFY_ENV} must be exactly '1'")
    return True


def top2_tree_optimized_draft_enabled() -> bool:
    value = os.environ.get(TOP2_TREE_OPTIMIZED_DRAFT_ENV)
    if value is None:
        return False
    if value != "1":
        raise ValueError(f"{TOP2_TREE_OPTIMIZED_DRAFT_ENV} must be exactly '1'")
    return True


def validate_server_args(server_args: Any) -> None:
    """Validate the audited greedy tier or the fixed B1/M4 sampled tier."""

    errors: list[str] = []
    tree_verify = top2_tree_verify_enabled()
    tree_optimized_draft = top2_tree_optimized_draft_enabled()
    if tree_optimized_draft and not tree_verify:
        errors.append("optimized tree draft requires top2 tree verification")
    if tree_optimized_draft and os.environ.get("NEXTLAT_DRAFT_CUDA_GRAPH") != "1":
        errors.append("optimized tree draft requires NEXTLAT_DRAFT_CUDA_GRAPH=1")

    algorithm = _value(server_args, "speculative_algorithm")
    if not isinstance(algorithm, str) or algorithm.upper() != "NEXTLAT":
        errors.append("speculative_algorithm must be NEXTLAT")

    draft_path = _value(server_args, "speculative_draft_model_path")
    if not isinstance(draft_path, str) or not draft_path.strip():
        errors.append("speculative_draft_model_path must name a local sidecar")

    model_path = _value(server_args, "model_path")
    if not isinstance(model_path, str) or not model_path.strip():
        errors.append("model_path must name a local target checkpoint directory")

    exact_values = {
        "speculative_eagle_topk": 2 if tree_verify else FIXED_TOPK,
        "tp_size": 1,
        "pp_size": 1,
        "dp_size": 1,
        "num_continuous_decode_steps": 1,
        "chunked_prefill_size": -1,
    }
    for name, expected in exact_values.items():
        observed = _value(server_args, name)
        if observed != expected:
            errors.append(f"{name} must be {expected}, got {observed!r}")

    max_drafts = _value(server_args, "speculative_num_steps")
    if (
        not isinstance(max_drafts, int)
        or isinstance(max_drafts, bool)
        or not 1 <= max_drafts <= MAX_DRAFT_TOKENS
    ):
        errors.append(
            "speculative_num_steps is the configured max draft length and must "
            f"be an integer in [1, {MAX_DRAFT_TOKENS}], got {max_drafts!r}"
        )
    verify_width = _value(server_args, "speculative_num_draft_tokens")
    expected_width = (
        TREE_NODE_BUDGET
        if tree_verify
        else max_drafts + 1
        if isinstance(max_drafts, int)
        else None
    )
    if (
        not isinstance(verify_width, int)
        or isinstance(verify_width, bool)
        or verify_width != expected_width
    ):
        if tree_verify:
            errors.append(
                "top2 tree speculative_num_draft_tokens must equal the fixed "
                f"node budget {TREE_NODE_BUDGET}, got {verify_width!r}"
            )
        else:
            errors.append(
                "speculative_num_draft_tokens must equal "
                f"speculative_num_steps + 1 ({expected_width!r}), got {verify_width!r}"
            )
    if tree_verify and max_drafts != TREE_HORIZON:
        errors.append(
            f"top2 tree speculative_num_steps must be {TREE_HORIZON}, "
            f"got {max_drafts!r}"
        )

    required_false = (
        "enable_dp_attention",
        "enable_mixed_chunk",
        "enable_multi_layer_eagle",
        "speculative_adaptive",
        "trust_remote_code",
    )
    for name in required_false:
        observed = _value(server_args, name)
        if observed is not False:
            errors.append(f"{name} must be disabled")

    rejection_sampling = _value(server_args, "speculative_use_rejection_sampling")
    if not isinstance(rejection_sampling, bool):
        errors.append("speculative_use_rejection_sampling must be a boolean")
    else:
        if rejection_sampling and (max_drafts != 4 or verify_width != 5):
            errors.append(
                "NEXTLAT rejection sampling currently requires fixed M=4 and "
                "speculative_num_draft_tokens=5"
            )
        if tree_verify and rejection_sampling:
            errors.append("top2 tree verification is greedy-only")

    overlap_disabled = _value(server_args, "disable_overlap_schedule")
    if not isinstance(overlap_disabled, bool):
        errors.append("disable_overlap_schedule must be a boolean")
    elif tree_verify and not overlap_disabled:
        errors.append("top2 tree verification initially requires overlap disabled")
    if tree_verify:
        expected_running_requests = (
            TREE_NODE_BUDGET + 1 if diagonal_decode_verify_enabled() else 1
        )
        if _value(server_args, "max_running_requests") != expected_running_requests:
            errors.append(
                "top2 tree verification requires max_running_requests="
                f"{expected_running_requests} for its selected target rail"
            )
    if tree_verify and not tree_optimized_draft:
        if os.environ.get("NEXTLAT_DRAFT_CUDA_GRAPH") is not None:
            errors.append(
                "top2 tree verification requires eager recurrent beam unless "
                f"{TOP2_TREE_OPTIMIZED_DRAFT_ENV}=1"
            )
        if os.environ.get("NEXTLAT_DRAFT_FP8") is not None:
            errors.append(
                "top2 tree verification requires BF16 recurrent beam unless "
                f"{TOP2_TREE_OPTIMIZED_DRAFT_ENV}=1"
            )

    dtype = _normalized(_value(server_args, "dtype"))
    if dtype not in {"bfloat16", "bf16", "torch.bfloat16"}:
        errors.append(f"dtype must be bfloat16, got {dtype!r}")

    device = _normalized(_value(server_args, "device"))
    if device != "cuda" and not device.startswith("cuda:"):
        errors.append(f"device must explicitly select CUDA, got {device!r}")

    target_quantization = _value(server_args, "quantization")
    experimental_fp8_target = experimental_allow_fp8_target_enabled()
    if target_quantization is not None:
        if not experimental_fp8_target:
            errors.append(
                "target quantization is not supported without "
                f"{EXPERIMENTAL_ALLOW_FP8_TARGET_ENV}=1"
            )
        elif target_quantization != "fp8":
            errors.append(
                "experimental target quantization admits only literal 'fp8', "
                f"got {target_quantization!r}"
            )
    if _normalized(_value(server_args, "load_format")) != "safetensors":
        errors.append("load_format must be safetensors")

    attention_backend = _normalized(_value(server_args, "attention_backend"))
    if attention_backend not in SERVING_ATTENTION_BACKENDS:
        errors.append(
            "attention_backend must be one of "
            f"{sorted(SERVING_ATTENTION_BACKENDS)!r}, got {attention_backend!r}"
        )
    for name in ("decode_attention_backend", "prefill_attention_backend"):
        observed = _value(server_args, name)
        if observed is not None and _normalized(observed) != attention_backend:
            errors.append(
                f"{name} must be unset or match attention_backend "
                f"{attention_backend!r}"
            )

    graph_config = _value(server_args, "cuda_graph_config")
    tree_decode_backend = (
        _phase_graph_backend(graph_config, "decode")
        if graph_config is not None
        else _value(server_args, "cuda_graph_backend_decode")
    )
    tree_graph_requested = bool(
        tree_verify and _normalized(tree_decode_backend) == "full"
    )
    legacy_target_verify_graph = target_verify_graph_diagnostic_enabled()
    diagonal_graph = diagonal_decode_verify_enabled()
    native_ragged_graph = native_ragged_verify_enabled()
    tree_diagonal_graph = tree_verify and diagonal_graph
    variable_b1_graph = fixed_b1_variable_verify_enabled()
    variable_b32_graph = fixed_b32_variable_verify_enabled()
    selected_graph_modes = sum(
        (legacy_target_verify_graph, diagonal_graph, native_ragged_graph)
    )
    if selected_graph_modes > 1:
        errors.append(
            "legacy target-verify, diagonal, and native ragged verifier graphs "
            "are mutually exclusive"
        )
    if variable_b32_graph and not diagonal_graph:
        errors.append(
            "variable B32 verifier geometry requires the diagonal DECODE graph"
        )
    if variable_b1_graph and not diagonal_graph:
        errors.append(
            "variable B1 verifier geometry requires the diagonal DECODE graph"
        )
    if variable_b1_graph and variable_b32_graph:
        errors.append("variable B1 and B32 verifier geometries are mutually exclusive")
    graph_diagnostic = (
        legacy_target_verify_graph or diagonal_graph or native_ragged_graph
    )
    if tree_verify and (legacy_target_verify_graph or native_ragged_graph):
        errors.append(
            "top2 tree verification admits only its native TARGET_VERIFY graph "
            "or the exact B14 diagonal DECODE graph"
        )
    if tree_verify and (variable_b1_graph or variable_b32_graph):
        errors.append(
            "top2 tree diagonal verification owns the fixed B1/H8/N14/B14 geometry"
        )
    graph_diagnostic = tree_graph_requested or graph_diagnostic
    if graph_diagnostic:
        if tree_diagonal_graph:
            graph_batch_size = diagonal_decode_max_rows()
            graph_label = "B1/H8/N14 tree diagonal DECODE verifier"
            if _value(server_args, "disable_cuda_graph") is True:
                errors.append(
                    "B1/H8/N14 tree diagonal DECODE requires CUDA graphs enabled"
                )
        elif tree_graph_requested:
            graph_batch_size = 1
            graph_label = "native B1/N14 tree TARGET_VERIFY"
            if _value(server_args, "disable_cuda_graph") is True:
                errors.append(
                    "native B1/N14 tree TARGET_VERIFY requires CUDA graphs enabled"
                )
        elif legacy_target_verify_graph:
            graph_batch_size = target_verify_graph_batch_size()
            graph_label = "legacy TARGET_VERIFY graph"
        elif native_ragged_graph:
            graph_batch_size = NATIVE_RAGGED_BATCH_SIZE
            graph_label = "native ragged TARGET_VERIFY"
        else:
            graph_batch_size = diagonal_decode_max_rows()
            graph_label = "diagonal DECODE verifier"
        if graph_config is None:
            expected_raw_graph_values = {
                "cuda_graph_backend_decode": "full",
                "cuda_graph_max_bs_decode": graph_batch_size,
                "cuda_graph_bs_decode": [graph_batch_size],
                "cuda_graph_backend_prefill": "disabled",
            }
            for name, expected in expected_raw_graph_values.items():
                observed = _value(server_args, name)
                if name.endswith("backend_decode") or name.endswith(
                    "backend_prefill"
                ):
                    observed = _normalized(observed)
                if observed != expected:
                    errors.append(
                        f"{graph_label} requires {name} to be {expected!r}, "
                        f"got {observed!r}"
                    )
        else:
            expected_graph_values = {
                ("decode", "backend"): "full",
                ("decode", "max_bs"): graph_batch_size,
                ("decode", "bs"): [graph_batch_size],
                ("prefill", "backend"): "disabled",
            }
            for (phase, name), expected in expected_graph_values.items():
                observed = _phase_graph_value(graph_config, phase, name)
                if name == "backend":
                    observed = _normalized(observed)
                if observed != expected:
                    errors.append(
                        f"{graph_label} requires cuda_graph_config.{phase}.{name} "
                        f"to be {expected!r}, got {observed!r}"
                    )
        if tree_diagonal_graph:
            admitted_graph_geometry = (
                max_drafts,
                verify_width,
                _value(server_args, "speculative_eagle_topk"),
                graph_batch_size,
            ) == (TREE_HORIZON, TREE_NODE_BUDGET, 2, TREE_NODE_BUDGET)
        elif tree_graph_requested:
            admitted_graph_geometry = (
                max_drafts,
                verify_width,
                _value(server_args, "speculative_eagle_topk"),
                graph_batch_size,
            ) == (TREE_HORIZON, TREE_NODE_BUDGET, 2, 1)
        elif legacy_target_verify_graph:
            admitted_graph_geometry = (
                max_drafts,
                verify_width,
            ) == (
                TARGET_VERIFY_GRAPH_MAX_DRAFT_TOKENS,
                TARGET_VERIFY_GRAPH_VERIFY_WIDTH,
            )
        elif native_ragged_graph:
            admitted_graph_geometry = (
                max_drafts,
                verify_width,
                graph_batch_size,
            ) == (4, 5, 32)
        elif variable_b1_graph:
            admitted_graph_geometry = bool(
                isinstance(max_drafts, int)
                and not isinstance(max_drafts, bool)
                and 1 <= max_drafts <= 16
                and verify_width == max_drafts + 1
                and graph_batch_size == verify_width
            )
        elif variable_b32_graph:
            admitted_graph_geometry = bool(
                isinstance(max_drafts, int)
                and not isinstance(max_drafts, bool)
                and 1 <= max_drafts <= 16
                and verify_width == max_drafts + 1
                and graph_batch_size == 32 * verify_width
            )
        else:
            admitted_graph_geometry = (
                max_drafts,
                verify_width,
                graph_batch_size,
            ) in {
                (4, 5, 5),
                (4, 5, 40),
                (4, 5, 160),
                (8, 9, 9),
                (16, 17, 17),
                (16, 17, 544),
                (8, 9, 72),
                (8, 9, 288),
                (8, 9, 576),
                (8, 9, 1152),
            }
        if not admitted_graph_geometry:
            errors.append(f"{graph_label} received an unregistered graph geometry")
    elif graph_config is not None:
        for phase in ("decode", "prefill"):
            backend = _normalized(_phase_graph_backend(graph_config, phase))
            if backend != "disabled":
                errors.append(
                    f"cuda_graph_config.{phase}.backend must be disabled, got {backend!r}"
                )
    else:
        legacy_disabled = _value(server_args, "disable_cuda_graph") is True
        phase_backends = {
            phase: _value(server_args, f"cuda_graph_backend_{phase}")
            for phase in ("decode", "prefill")
        }
        for phase, backend in phase_backends.items():
            if backend is not None and _normalized(backend) != "disabled":
                errors.append(
                    f"cuda_graph_backend_{phase} must be unset or disabled"
                )
        explicitly_disabled = all(
            _normalized(backend) == "disabled"
            for backend in phase_backends.values()
        )
        if not legacy_disabled and not explicitly_disabled:
            errors.append(
                "decode and prefill CUDA graphs must both be disabled"
            )

    enable_lora = _value(server_args, "enable_lora")
    if enable_lora not in (None, False):
        errors.append("enable_lora must be unset or disabled")
    if _value(server_args, "lora_paths"):
        errors.append("lora_paths must be empty")

    if _normalized(_value(server_args, "disaggregation_mode")) != "null":
        errors.append("disaggregation_mode must be null")
    if _value(server_args, "speculative_draft_load_format") is not None:
        errors.append("speculative_draft_load_format must be unset")
    token_map = _value(server_args, "speculative_token_map")
    if not isinstance(token_map, str) or not token_map.strip():
        errors.append(
            "speculative_token_map must name the 32,768-token Qwen hot-vocabulary table"
        )

    if errors:
        raise ValueError("invalid NEXTLAT server arguments:\n- " + "\n- ".join(errors))

    # The registry hook runs before worker construction.  Verify the local
    # receipt/config now, but leave the large safetensors load to the worker.
    read_sidecar_metadata(Path(draft_path))
