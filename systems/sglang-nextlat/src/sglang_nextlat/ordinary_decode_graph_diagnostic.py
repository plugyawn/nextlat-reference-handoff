"""Canary-only B1 ordinary-DECODE graph capture for NextLat correction.

The sole target decode runner is captured while the model runner temporarily
identifies as non-speculative.  Runtime identity is restored to NEXTLAT before
serving.  Only a one-row ordinary DECODE with no ``spec_info`` may replay;
ragged target verification must remain eager.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import torch

from .config import target_verify_graph_diagnostic_enabled


GRAPH_BATCH_SIZE = 1
GRAPH_TOKEN_ROWS = 1
_PATCHED = False


def _algorithm_name(value: Any) -> str | None:
    return getattr(value, "name", None)


def _is_nextlat_model_runner(model_runner: Any) -> bool:
    return _algorithm_name(getattr(model_runner, "spec_algorithm", None)) == "NEXTLAT"


def _is_nextlat_runner(runner: Any) -> bool:
    return _is_nextlat_model_runner(getattr(runner, "model_runner", None))


@contextmanager
def temporary_spec_algorithm(
    model_runner: Any, *, replacement: Any
) -> Iterator[None]:
    """Temporarily replace model-runner identity and always restore by identity."""

    if not hasattr(model_runner, "spec_algorithm"):
        raise ValueError("model runner lacks spec_algorithm")
    original = model_runner.spec_algorithm
    try:
        model_runner.spec_algorithm = replacement
        yield
    finally:
        model_runner.spec_algorithm = original
        if model_runner.spec_algorithm is not original:
            raise RuntimeError("model runner speculative identity was not restored")


@contextmanager
def temporary_attribute(owner: Any, name: str, replacement: Any) -> Iterator[None]:
    """Temporarily replace one existing source hook and restore by identity."""

    if not hasattr(owner, name):
        raise ValueError(f"temporary hook owner lacks {name}")
    original = getattr(owner, name)
    try:
        setattr(owner, name, replacement)
        yield
    finally:
        setattr(owner, name, original)
        if getattr(owner, name) is not original:
            raise RuntimeError(f"temporary source hook {name} was not restored")


def is_exact_ordinary_decode_geometry(forward_batch: Any) -> bool:
    """Return the only runtime geometry authorized to replay the graph."""

    mode = getattr(forward_batch, "forward_mode", None)
    input_ids = getattr(forward_batch, "input_ids", None)
    out_cache_loc = getattr(forward_batch, "out_cache_loc", None)
    positions = getattr(forward_batch, "positions", None)
    return bool(
        getattr(mode, "name", None) == "DECODE"
        and getattr(forward_batch, "spec_info", None) is None
        and getattr(forward_batch, "batch_size", None) == GRAPH_BATCH_SIZE
        and isinstance(input_ids, torch.Tensor)
        and input_ids.numel() == GRAPH_TOKEN_ROWS
        and isinstance(out_cache_loc, torch.Tensor)
        and out_cache_loc.numel() == GRAPH_TOKEN_ROWS
        and isinstance(positions, torch.Tensor)
        and positions.numel() == GRAPH_TOKEN_ROWS
    )


def _runner_contract(runner: Any) -> dict[str, Any]:
    capture_num_tokens = getattr(runner, "capture_num_tokens", None)
    return {
        "capture_forward_mode": getattr(
            getattr(runner, "capture_forward_mode", None), "name", None
        ),
        "capture_hidden_mode": getattr(
            getattr(runner, "capture_hidden_mode", None), "name", None
        ),
        "captured_req_width": getattr(runner, "captured_req_width", None),
        "capture_bs": list(getattr(runner, "capture_bs", [])),
        "capture_num_tokens": (
            None if capture_num_tokens is None else list(capture_num_tokens)
        ),
        "max_bs": getattr(runner, "max_bs", None),
        "ragged_verify_mode": getattr(runner, "ragged_verify_mode", None),
        "runtime_spec_algorithm": _algorithm_name(
            getattr(getattr(runner, "model_runner", None), "spec_algorithm", None)
        ),
    }


def assert_captured_runner(runner: Any) -> dict[str, Any]:
    """Prove the sole runner is B1, width-one, FULL-hidden ordinary DECODE."""

    if runner is None or not _is_nextlat_runner(runner):
        raise RuntimeError("ordinary correction graph has no restored NEXTLAT runner")
    expected = {
        "capture_forward_mode": "DECODE",
        "capture_hidden_mode": "FULL",
        "captured_req_width": GRAPH_TOKEN_ROWS,
        "capture_bs": [GRAPH_BATCH_SIZE],
        "capture_num_tokens": None,
        "max_bs": GRAPH_BATCH_SIZE,
        "ragged_verify_mode": False,
        "runtime_spec_algorithm": "NEXTLAT",
    }
    observed = _runner_contract(runner)
    if observed != expected:
        raise RuntimeError(
            f"ordinary correction graph contract differs: {observed!r} != {expected!r}"
        )
    output = getattr(getattr(runner, "buffers", None), "next_token_logits_buffer", None)
    if not isinstance(output, torch.Tensor) or output.ndim != 2 or output.shape[0] < 1:
        raise RuntimeError("ordinary correction graph lacks a static logits buffer")
    return {**observed, "static_logits_buffer_shape": list(output.shape)}


def ordinary_decode_replay_snapshot(runner: Any) -> dict[str, Any]:
    """Prove the latest accepted replay used the exact unpadded B1 graph."""

    contract = assert_captured_runner(runner)
    key = getattr(runner, "_replay_graph_key", None)
    replay = {
        "raw_bs": getattr(runner, "raw_bs", None),
        "padded_bs": getattr(runner, "bs", None),
        "raw_rows": getattr(runner, "raw_num_token", None),
        "graph_key_size": getattr(key, "size", None),
        "ordinary_decode_admission_count": getattr(
            runner, "_nextlat_ordinary_decode_admission_count", None
        ),
        "target_verify_eager_count": getattr(
            runner, "_nextlat_target_verify_eager_count", None
        ),
    }
    expected_geometry = {
        "raw_bs": 1,
        "padded_bs": 1,
        "raw_rows": 1,
        "graph_key_size": 1,
    }
    if {name: replay[name] for name in expected_geometry} != expected_geometry:
        raise RuntimeError(f"ordinary correction graph replay drifted: {replay!r}")
    if not isinstance(replay["ordinary_decode_admission_count"], int) or replay[
        "ordinary_decode_admission_count"
    ] <= 0:
        raise RuntimeError("ordinary correction graph recorded no admitted replay")
    if not isinstance(replay["target_verify_eager_count"], int) or replay[
        "target_verify_eager_count"
    ] <= 0:
        raise RuntimeError("ordinary correction graph lacks eager verify evidence")
    return {"capture_contract": contract, "latest_replay": replay}


def install_ordinary_decode_graph_overlay() -> None:
    """Patch the pinned runner constructor and admission for one canary shape."""

    global _PATCHED
    if not target_verify_graph_diagnostic_enabled() or _PATCHED:
        return

    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
        DecodeCudaGraphRunner,
    )
    from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
    from sglang.srt.model_executor.runner import base_runner as base_runner_module
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    _install_runner_overlay(
        DecodeCudaGraphRunner,
        no_spec_algorithm=SpeculativeAlgorithm.NONE,
        full_hidden_mode=CaptureHiddenMode.FULL,
        base_runner_module=base_runner_module,
    )
    _PATCHED = True


def _install_runner_overlay(
    runner_class: Any,
    *,
    no_spec_algorithm: Any,
    full_hidden_mode: Any,
    base_runner_module: Any,
) -> None:
    """Install the ctor/admission wrappers; split out for source-level tests."""

    original_init = runner_class.__init__
    original_can_run_graph = runner_class.can_run_graph

    def init(
        runner: Any,
        model_runner: Any,
        *,
        attn_backend=None,
        speculative_num_steps=None,
        speculative_num_draft_tokens=None,
    ) -> None:
        if not _is_nextlat_model_runner(model_runner):
            original_init(
                runner,
                model_runner,
                attn_backend=attn_backend,
                speculative_num_steps=speculative_num_steps,
                speculative_num_draft_tokens=speculative_num_draft_tokens,
            )
            return
        if model_runner.is_draft_worker:
            raise RuntimeError("ordinary correction graph cannot capture a draft worker")
        with (
            temporary_spec_algorithm(
                model_runner, replacement=no_spec_algorithm
            ),
            temporary_attribute(
                base_runner_module,
                "get_server_return_hidden_states_mode",
                lambda: full_hidden_mode,
            ),
        ):
            original_init(
                runner,
                model_runner,
                attn_backend=attn_backend,
                speculative_num_steps=0,
                speculative_num_draft_tokens=GRAPH_TOKEN_ROWS,
            )
        runner._nextlat_ordinary_decode_admission_count = 0
        runner._nextlat_target_verify_eager_count = 0
        assert_captured_runner(runner)

    def can_run_graph(runner: Any, forward_batch: Any):
        if not _is_nextlat_runner(runner):
            return original_can_run_graph(runner, forward_batch)
        expected = is_exact_ordinary_decode_geometry(forward_batch)
        if expected:
            observed = bool(original_can_run_graph(runner, forward_batch))
            if not observed:
                raise RuntimeError(
                    "base graph runner rejected exact B1 ordinary DECODE"
                )
            runner._nextlat_ordinary_decode_admission_count += 1
            return True
        if getattr(
            getattr(forward_batch, "forward_mode", None), "name", None
        ) == "TARGET_VERIFY":
            runner._nextlat_target_verify_eager_count += 1
        # Pinned base admission does not itself require forward-mode equality.
        # Every NEXTLAT geometry outside exact B1 ordinary DECODE is explicitly
        # eager, including ragged TARGET_VERIFY and completion-tail variants.
        return False

    runner_class.__init__ = init
    runner_class.can_run_graph = can_run_graph
