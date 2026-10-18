"""Diagnostic ordinary-DECODE rail for linear speculative verification.

The target verifier normally presents ``[anchor, draft_1, ...]`` as one
``TARGET_VERIFY`` extend row.  FA3 then uses its multi-query extend kernel,
which is not numerically identical to the ordinary one-token DECODE graph.

This module provides the smallest throughput-preserving falsifier: keep the
same compact target tokens and scratch KV slots, but reinterpret every token as
one q_len=1 DECODE row.  Rows for a request repeat its request-pool index and
use monotonically increasing sequence lengths, so row ``j`` sees the canonical
prefix plus scratch rows ``0..j``.  One target-model forward still processes
the whole verification window; only the attention execution rail changes.

The implementation is diagnostic until the H100 semantic canaries pass.  It
has no eager fallback and does not silently admit ordinary TARGET_VERIFY.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import torch

from .beam_tree import TREE_NODE_BUDGET


MAX_ROWS_ENV = "NEXTLAT_DIAGONAL_VERIFY_MAX_ROWS"
ENABLED_ENV = "NEXTLAT_DIAGONAL_VERIFY_DIAGNOSTIC"
VARIABLE_B32_ENABLED_ENV = "NEXTLAT_FIXED_B32_VARIABLE_VERIFY_GEOMETRY"
VARIABLE_B1_ENABLED_ENV = "NEXTLAT_FIXED_B1_VARIABLE_VERIFY_GEOMETRY"
TARGET_BF16_LOGITS_BUFFER_ENV = "NEXTLAT_TARGET_BF16_LOGITS_BUFFER"
_PATCHED = False
_LEGACY_ADMITTED_MAX_ROWS = (5, 9, 17, 40, 72, 160, 288, 544, 576, 1152)
_B1_VARIABLE_MAX_ROWS = tuple(draft_tokens + 1 for draft_tokens in range(1, 17))
_B32_VARIABLE_MAX_ROWS = tuple(32 * (draft_tokens + 1) for draft_tokens in range(1, 17))
_ADMITTED_MAX_ROWS = tuple(
    sorted(
        set(
            _LEGACY_ADMITTED_MAX_ROWS
            + _B1_VARIABLE_MAX_ROWS
            + _B32_VARIABLE_MAX_ROWS
        )
    )
)


def _algorithm_name(value: Any) -> str | None:
    return getattr(value, "name", None)


def _is_nextlat_model_runner(model_runner: Any) -> bool:
    return _algorithm_name(getattr(model_runner, "spec_algorithm", None)) == "NEXTLAT"


def _is_nextlat_runner(runner: Any) -> bool:
    return _is_nextlat_model_runner(getattr(runner, "model_runner", None))


def diagonal_decode_verify_enabled() -> bool:
    value = os.environ.get(ENABLED_ENV)
    if value is None:
        return False
    if value != "1":
        raise RuntimeError(f"{ENABLED_ENV} must be exactly '1' when present")
    return True


def fixed_b32_variable_verify_enabled() -> bool:
    value = os.environ.get(VARIABLE_B32_ENABLED_ENV)
    if value is None:
        return False
    if value != "1":
        raise RuntimeError(
            f"{VARIABLE_B32_ENABLED_ENV} must be exactly '1' when present"
        )
    return True


def fixed_b1_variable_verify_enabled() -> bool:
    value = os.environ.get(VARIABLE_B1_ENABLED_ENV)
    if value is None:
        return False
    if value != "1":
        raise RuntimeError(
            f"{VARIABLE_B1_ENABLED_ENV} must be exactly '1' when present"
        )
    return True


def target_bf16_logits_buffer_enabled() -> bool:
    value = os.environ.get(TARGET_BF16_LOGITS_BUFFER_ENV)
    if value is None:
        return False
    if value != "1":
        raise RuntimeError(
            f"{TARGET_BF16_LOGITS_BUFFER_ENV} must be exactly '1' when present"
        )
    return True


def diagonal_decode_max_rows() -> int:
    value = os.environ.get(MAX_ROWS_ENV)
    if value is None:
        raise RuntimeError(f"{MAX_ROWS_ENV} is required by the diagonal verifier")
    try:
        rows = int(value)
    except ValueError as exc:
        raise RuntimeError(f"{MAX_ROWS_ENV} must be a base-10 integer") from exc
    if rows not in _ADMITTED_MAX_ROWS or str(rows) != value:
        admitted = ", ".join(repr(str(row)) for row in _ADMITTED_MAX_ROWS)
        raise RuntimeError(
            f"{MAX_ROWS_ENV} must be exactly one of {admitted}"
        )
    return rows


@dataclass(frozen=True)
class DiagonalGeometry:
    verify_lens: tuple[int, ...]
    prefix_seq_lens: tuple[int, ...]
    row_to_request: tuple[int, ...]
    within_request: tuple[int, ...]
    positions: tuple[int, ...]
    seq_lens: tuple[int, ...]

    @property
    def batch_size(self) -> int:
        return len(self.verify_lens)

    @property
    def total_rows(self) -> int:
        return len(self.row_to_request)


def build_diagonal_geometry(
    *, verify_lens: Sequence[int], prefix_seq_lens: Sequence[int]
) -> DiagonalGeometry:
    """Build the exact q_len=1 row mapping without touching runtime state."""

    verify = tuple(int(value) for value in verify_lens)
    prefixes = tuple(int(value) for value in prefix_seq_lens)
    if not verify or len(verify) != len(prefixes):
        raise ValueError("diagonal verify needs one non-empty length per request")
    if any(length < 1 or length > 17 for length in verify):
        raise ValueError("diagonal verify lengths must be in [1, 17]")
    if any(prefix < 1 for prefix in prefixes):
        raise ValueError("diagonal verify prefixes must be positive")

    row_to_request: list[int] = []
    within_request: list[int] = []
    positions: list[int] = []
    seq_lens: list[int] = []
    for request_index, (prefix, length) in enumerate(zip(prefixes, verify)):
        for within in range(length):
            row_to_request.append(request_index)
            within_request.append(within)
            positions.append(prefix + within)
            seq_lens.append(prefix + within + 1)
    return DiagonalGeometry(
        verify_lens=verify,
        prefix_seq_lens=prefixes,
        row_to_request=tuple(row_to_request),
        within_request=tuple(within_request),
        positions=tuple(positions),
        seq_lens=tuple(seq_lens),
    )


def _expand_optional_rows(values: Any, row_to_request: torch.Tensor) -> Any:
    if values is None:
        return None
    if isinstance(values, torch.Tensor):
        return values[row_to_request]
    if isinstance(values, list):
        indices = row_to_request.detach().cpu().tolist()
        return [values[index] for index in indices]
    raise RuntimeError(f"unsupported per-request diagonal field: {type(values)!r}")


def _repeat_single_optional_row(values: Any, row_count: int) -> Any:
    """Repeat one request-shaped field without a device-to-host index copy."""

    if values is None:
        return None
    if isinstance(values, torch.Tensor):
        if values.ndim < 1 or values.shape[0] != 1:
            raise RuntimeError("tree diagonal tensor field must contain one request")
        return values.expand((row_count,) + tuple(values.shape[1:])).contiguous()
    if isinstance(values, list):
        if len(values) != 1:
            raise RuntimeError("tree diagonal list field must contain one request")
        return values * row_count
    raise RuntimeError(
        f"unsupported per-request tree diagonal field: {type(values)!r}"
    )


def diagonalize_verify_forward_batch(
    forward_batch: Any,
    *,
    verify_lens_cpu: Sequence[int],
    prefix_seq_lens: torch.Tensor,
    prefix_seq_lens_cpu: Sequence[int],
) -> dict[str, Any]:
    """Mutate one freshly-built TARGET_VERIFY ForwardBatch into DECODE rows.

    The caller must invoke this after SGLang has allocated/written the compact
    verify scratch locations and before the target model forward.  No accepted
    tokens or request lengths are committed here.
    """

    from sglang.srt.model_executor.forward_batch_info import (
        CaptureHiddenMode,
        ForwardMode,
    )

    if not diagonal_decode_verify_enabled():
        raise RuntimeError("diagonal verify mutation requires its explicit diagnostic")
    if getattr(forward_batch.forward_mode, "name", None) != "TARGET_VERIFY":
        raise RuntimeError("diagonal verifier did not receive TARGET_VERIFY")
    if forward_batch.spec_info is None:
        raise RuntimeError("diagonal verifier lost the original verify spec_info")
    if forward_batch.capture_hidden_mode != CaptureHiddenMode.FULL:
        raise RuntimeError("diagonal verifier requires FULL target hidden states")
    if bool(getattr(forward_batch, "forward_metadata_ready", False)):
        raise RuntimeError("diagonal verifier refuses a pre-planned ForwardBatch")
    if getattr(getattr(forward_batch, "spec_algorithm", None), "name", None) != "NEXTLAT":
        raise RuntimeError("diagonal verifier requires NEXTLAT batch identity")
    if bool(getattr(forward_batch, "return_logprob", False)):
        raise RuntimeError("diagonal verifier does not admit logprob requests")
    if bool(getattr(forward_batch, "is_prefill_only", False)):
        raise RuntimeError("diagonal verifier does not admit prefill-only batches")

    # These fields would need their own row-wise semantic definition.  The
    # admitted Qwen3 text-only TP1 lane leaves every one unset; fail rather
    # than silently reuse request-shaped metadata for token-shaped rows.
    unsupported_fields = (
        "input_embeds",
        "replace_embeds",
        "replace_positions",
        "token_type_ids",
        "encoder_lens",
        "encoder_out_cache_loc",
        "mamba_track_indices",
        "mamba_track_mask",
        "mamba_track_seqlens",
        "mamba_cow_src_indices",
        "mamba_cow_dst_indices",
        "mamba_clear_indices",
        "mrope_positions",
        "rids_int",
        "bootstrap_room_ids_int",
        "req_all_ids_flat",
        "req_all_ids_lens",
        "attn_cp_metadata",
        "attn_dcp_metadata",
        "dcp_kv_mask",
        "ngram_embedding_info",
    )
    populated_unsupported = [
        name for name in unsupported_fields if getattr(forward_batch, name, None) is not None
    ]
    if populated_unsupported:
        raise RuntimeError(
            "diagonal verifier found unsupported ForwardBatch fields: "
            + ", ".join(populated_unsupported)
        )

    prefix_cpu = tuple(int(value) for value in prefix_seq_lens_cpu)
    geometry = build_diagonal_geometry(
        verify_lens=verify_lens_cpu, prefix_seq_lens=prefix_cpu
    )
    max_rows = diagonal_decode_max_rows()
    if fixed_b1_variable_verify_enabled():
        if not 2 <= max_rows <= 17:
            raise RuntimeError(
                "variable B1 diagonal verification requires exactly M+1 rows "
                "for M in [1,16]"
            )
        verify_width = max_rows
    elif fixed_b32_variable_verify_enabled():
        if max_rows % 32 != 0 or not 2 <= max_rows // 32 <= 17:
            raise RuntimeError(
                "variable B32 diagonal verification requires exactly 32 x (M+1) rows"
            )
        verify_width = max_rows // 32
    else:
        # M4 uses width five for one, eight, or 32 requests. M16 uses width 17
        # for B1 or B32; all remaining admitted lanes use M8 width nine.
        verify_width = (
            5 if max_rows in (5, 40, 160) else 17 if max_rows in (17, 544) else 9
        )
    max_requests = max_rows // verify_width
    if not 1 <= geometry.batch_size <= max_requests:
        raise RuntimeError(
            "diagonal verify request count exceeds the admitted graph lane"
        )
    if geometry.total_rows > max_rows:
        raise RuntimeError(
            f"diagonal verify has {geometry.total_rows} rows, graph cap is {max_rows}"
        )
    if geometry.batch_size != int(forward_batch.batch_size):
        raise RuntimeError("diagonal verify request count differs from ForwardBatch")

    required_tensors = {
        "input_ids": forward_batch.input_ids,
        "out_cache_loc": forward_batch.out_cache_loc,
        "positions": forward_batch.positions,
        "req_pool_indices": forward_batch.req_pool_indices,
        "prefix_seq_lens": prefix_seq_lens,
    }
    for name, value in required_tensors.items():
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(f"diagonal verify {name} is not a tensor")
    if forward_batch.input_ids.numel() != geometry.total_rows:
        raise RuntimeError("diagonal verify input row count differs")
    if forward_batch.out_cache_loc.numel() != geometry.total_rows:
        raise RuntimeError("diagonal verify KV scratch row count differs")
    if forward_batch.positions.numel() != geometry.total_rows:
        raise RuntimeError("diagonal verify position row count differs")
    if forward_batch.req_pool_indices.numel() != geometry.batch_size:
        raise RuntimeError("diagonal verify request-pool row count differs")
    if prefix_seq_lens.numel() != geometry.batch_size:
        raise RuntimeError("diagonal verify prefix tensor count differs")

    device = forward_batch.input_ids.device
    row_to_request = torch.tensor(
        geometry.row_to_request, dtype=torch.int64, device=device
    )
    within_request = torch.tensor(
        geometry.within_request, dtype=prefix_seq_lens.dtype, device=device
    )
    diagonal_seq_lens = (
        prefix_seq_lens[row_to_request] + within_request + 1
    ).contiguous()
    expected_positions = diagonal_seq_lens.to(forward_batch.positions.dtype) - 1
    if not torch.equal(forward_batch.positions, expected_positions):
        raise RuntimeError("diagonal verify positions differ from prefix+offset")

    original_req_pool_indices = forward_batch.req_pool_indices
    diagonal_req_pool_indices = original_req_pool_indices[row_to_request].contiguous()
    diagonal_seq_lens_cpu = torch.tensor(
        geometry.seq_lens, dtype=torch.int64, device="cpu"
    )

    original_rids = getattr(forward_batch, "rids", None)
    original_lora_ids = getattr(forward_batch, "lora_ids", None)
    original_orig_seq_lens = getattr(forward_batch, "orig_seq_lens", None)
    original_mm_inputs = getattr(forward_batch, "mm_inputs", None)
    forward_batch.forward_mode = ForwardMode.DECODE
    forward_batch.global_forward_mode = ForwardMode.DECODE
    forward_batch.batch_size = geometry.total_rows
    forward_batch.req_pool_indices = diagonal_req_pool_indices
    forward_batch.seq_lens = diagonal_seq_lens
    forward_batch.seq_lens_cpu = diagonal_seq_lens_cpu
    forward_batch.seq_lens_sum = sum(geometry.seq_lens)
    forward_batch.spec_info = None
    forward_batch.rids = _expand_optional_rows(original_rids, row_to_request)
    forward_batch.lora_ids = _expand_optional_rows(original_lora_ids, row_to_request)
    forward_batch.orig_seq_lens = _expand_optional_rows(
        original_orig_seq_lens, row_to_request
    )
    forward_batch.mm_inputs = _expand_optional_rows(original_mm_inputs, row_to_request)
    forward_batch.capture_hidden_mode = CaptureHiddenMode.FULL
    # The graph admission hook requires this diagnostic-only provenance bit.
    # It is deliberately not inferred from DECODE geometry alone.
    forward_batch._nextlat_diagonal_verify = True

    # Fields used only by extend/TARGET_VERIFY must not survive the mode switch.
    forward_batch.extend_num_tokens = None
    forward_batch.extend_seq_lens = None
    forward_batch.extend_prefix_lens = None
    forward_batch.extend_start_loc = None
    forward_batch.extend_prefix_lens_cpu = None
    forward_batch.extend_seq_lens_cpu = None
    forward_batch.extend_logprob_start_lens_cpu = None

    return {
        "original_request_count": geometry.batch_size,
        "active_decode_rows": geometry.total_rows,
        "graph_max_rows": max_rows,
        "verify_lens": list(geometry.verify_lens),
        "prefix_seq_lens": list(geometry.prefix_seq_lens),
        "row_to_request": list(geometry.row_to_request),
        "within_request": list(geometry.within_request),
        "positions": list(geometry.positions),
        "seq_lens": list(geometry.seq_lens),
        "req_pool_indices": diagonal_req_pool_indices.detach().cpu().tolist(),
        "out_cache_locs": forward_batch.out_cache_loc.detach().cpu().tolist(),
        "input_ids": forward_batch.input_ids.detach().cpu().tolist(),
        "capture_hidden_mode": "FULL",
        "forward_mode": "DECODE",
        "spec_info_cleared": forward_batch.spec_info is None,
        "diagnostic_marker": bool(forward_batch._nextlat_diagonal_verify),
    }


def diagonalize_tree_verify_forward_batch(
    forward_batch: Any,
    *,
    ancestor_mask: torch.Tensor,
    canonical_req_pool_indices: torch.Tensor,
    virtual_req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    prefix_seq_len: int,
    page_size: int,
) -> dict[str, Any]:
    """Turn one tree ``TARGET_VERIFY`` batch into ordinary DECODE rows.

    Every tree node becomes one q_len=1 decode lane.  Its virtual request-table
    row aliases the exact request prefix and then lists only the scratch KV
    locations on that node's root-to-node path.  The target therefore evaluates
    the complete causal tree in one ordinary-DECODE forward without using the
    top-k TARGET_VERIFY cascade/merge rail.

    ``eagle_prepare_for_verify`` must run first: it owns allocation of the
    node-order scratch locations in ``forward_batch.out_cache_loc``.  The
    virtual request rows are caller-owned reservations and must not include the
    canonical request row or padding row zero.

    Only page-size one is admitted.  With larger physical KV pages, arbitrary
    branch paths cannot in general be represented by aliasing individual token
    locations into an ordinary paged-DECODE row.
    """

    from sglang.srt.model_executor.forward_batch_info import (
        CaptureHiddenMode,
        ForwardMode,
    )

    if not diagonal_decode_verify_enabled():
        raise RuntimeError(
            "tree diagonal verify mutation requires its explicit diagnostic"
        )
    if getattr(forward_batch.forward_mode, "name", None) != "TARGET_VERIFY":
        raise RuntimeError("tree diagonal verifier did not receive TARGET_VERIFY")
    if forward_batch.spec_info is None:
        raise RuntimeError("tree diagonal verifier lost the original verify spec_info")
    if forward_batch.capture_hidden_mode != CaptureHiddenMode.FULL:
        raise RuntimeError("tree diagonal verifier requires FULL target hidden states")
    if bool(getattr(forward_batch, "forward_metadata_ready", False)):
        raise RuntimeError("tree diagonal verifier refuses a pre-planned ForwardBatch")
    if (
        getattr(getattr(forward_batch, "spec_algorithm", None), "name", None)
        != "NEXTLAT"
    ):
        raise RuntimeError("tree diagonal verifier requires NEXTLAT batch identity")
    if bool(getattr(forward_batch, "return_logprob", False)):
        raise RuntimeError("tree diagonal verifier does not admit logprob requests")
    if bool(getattr(forward_batch, "is_prefill_only", False)):
        raise RuntimeError("tree diagonal verifier does not admit prefill-only batches")
    if page_size != 1:
        raise RuntimeError("tree diagonal verifier requires page_size=1")
    if not isinstance(prefix_seq_len, int) or isinstance(prefix_seq_len, bool):
        raise TypeError("tree diagonal verifier prefix length must be an integer")
    if prefix_seq_len < 1:
        raise RuntimeError("tree diagonal verifier prefix length must be positive")

    required_tensors = {
        "input_ids": getattr(forward_batch, "input_ids", None),
        "out_cache_loc": getattr(forward_batch, "out_cache_loc", None),
        "positions": getattr(forward_batch, "positions", None),
        "req_pool_indices": getattr(forward_batch, "req_pool_indices", None),
        "seq_lens": getattr(forward_batch, "seq_lens", None),
        "ancestor_mask": ancestor_mask,
        "canonical_req_pool_indices": canonical_req_pool_indices,
        "virtual_req_pool_indices": virtual_req_pool_indices,
        "req_to_token": req_to_token,
    }
    for name, value in required_tensors.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"tree diagonal verify {name} is not a tensor")

    if (
        ancestor_mask.ndim != 2
        or ancestor_mask.shape[0] != ancestor_mask.shape[1]
    ):
        raise RuntimeError("tree diagonal ancestor mask must be square")
    if ancestor_mask.dtype != torch.bool:
        raise RuntimeError("tree diagonal ancestor mask must be boolean")
    node_count = int(ancestor_mask.shape[0])
    if node_count < 1:
        raise RuntimeError("tree diagonal verifier requires at least one node")
    if node_count > diagonal_decode_max_rows():
        raise RuntimeError("tree diagonal node count exceeds the graph row cap")
    if any(
        tensor.device != forward_batch.input_ids.device
        for tensor in (
            ancestor_mask,
            canonical_req_pool_indices,
            virtual_req_pool_indices,
            req_to_token,
            forward_batch.out_cache_loc,
            forward_batch.positions,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
        )
    ):
        raise RuntimeError("tree diagonal tensors must share one device")
    if req_to_token.ndim != 2 or req_to_token.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise RuntimeError("tree diagonal req_to_token must be a 2D integer tensor")
    if prefix_seq_len + node_count > req_to_token.shape[1]:
        raise RuntimeError("tree diagonal path rows exceed req_to_token width")
    if forward_batch.batch_size != 1:
        raise RuntimeError("tree diagonal verifier requires one exact request")
    if canonical_req_pool_indices.numel() != 1:
        raise RuntimeError("tree diagonal verifier requires one canonical request row")
    if canonical_req_pool_indices.dtype not in (torch.int32, torch.int64):
        raise RuntimeError("tree diagonal canonical request row must be integer")
    canonical_row_value = canonical_req_pool_indices.reshape(-1)[0]
    if forward_batch.req_pool_indices.numel() != 1:
        raise RuntimeError("tree diagonal ForwardBatch must have one request row")
    if (
        virtual_req_pool_indices.ndim != 1
        or virtual_req_pool_indices.numel() != node_count
    ):
        raise RuntimeError("tree diagonal verifier requires one virtual row per node")
    if virtual_req_pool_indices.dtype not in (torch.int32, torch.int64):
        raise RuntimeError("tree diagonal virtual request rows must be integer")
    if any(
        tensor.ndim != 1 or tensor.numel() != node_count
        for tensor in (
            forward_batch.input_ids,
            forward_batch.out_cache_loc,
            forward_batch.positions,
        )
    ):
        raise RuntimeError("tree diagonal target rows differ from the node count")
    if forward_batch.seq_lens.numel() != 1:
        raise RuntimeError("tree diagonal verifier lost the exact prefix length")
    # These dynamic guards are useful in CPU unit tests, but each CUDA
    # assertion is a host-visible synchronization in this per-verify hot path.
    # CUDA inputs come exclusively from the validated fixed tree packer and
    # the host-reserved row roster, so keep the production rail asynchronous.
    if forward_batch.input_ids.device.type == "cpu":
        torch._assert_async(
            (canonical_row_value > 0)
            & (canonical_row_value < req_to_token.shape[0]),
            "tree diagonal canonical request row is out of range",
        )
        torch._assert_async(
            torch.all(
                forward_batch.req_pool_indices.reshape(-1)
                == canonical_req_pool_indices.reshape(-1).to(
                    forward_batch.req_pool_indices.dtype
                )
            ),
            "tree diagonal canonical request row differs from ForwardBatch",
        )
        duplicate_count = (
            virtual_req_pool_indices[:, None] == virtual_req_pool_indices[None, :]
        ).sum()
        torch._assert_async(
            duplicate_count == node_count,
            "tree diagonal virtual request rows must be unique",
        )
        torch._assert_async(
            torch.all(virtual_req_pool_indices != 0),
            "tree diagonal virtual request rows cannot use padding row zero",
        )
        torch._assert_async(
            torch.all(virtual_req_pool_indices != canonical_row_value),
            "tree diagonal virtual rows cannot alias the canonical row",
        )
        torch._assert_async(
            torch.all(
                (virtual_req_pool_indices > 0)
                & (virtual_req_pool_indices < req_to_token.shape[0])
            ),
            "tree diagonal virtual request row is out of range",
        )
        torch._assert_async(
            torch.all(forward_batch.seq_lens == prefix_seq_len),
            "tree diagonal ForwardBatch prefix length differs",
        )
        torch._assert_async(
            torch.all(torch.diagonal(ancestor_mask)),
            "tree diagonal every node must see itself",
        )
        torch._assert_async(
            ~torch.any(torch.triu(ancestor_mask, diagonal=1)),
            "tree diagonal nodes cannot see future rows",
        )
        node_depths = ancestor_mask.sum(dim=1).to(forward_batch.positions.dtype)
        expected_positions = prefix_seq_len + node_depths - 1
        torch._assert_async(
            torch.all(forward_batch.positions == expected_positions),
            "tree diagonal positions differ from prefix plus path depth",
        )

    unsupported_fields = (
        "input_embeds",
        "replace_embeds",
        "replace_positions",
        "token_type_ids",
        "encoder_lens",
        "encoder_out_cache_loc",
        "mamba_track_indices",
        "mamba_track_mask",
        "mamba_track_seqlens",
        "mamba_cow_src_indices",
        "mamba_cow_dst_indices",
        "mamba_clear_indices",
        "mrope_positions",
        "rids_int",
        "bootstrap_room_ids_int",
        "req_all_ids_flat",
        "req_all_ids_lens",
        "attn_cp_metadata",
        "attn_dcp_metadata",
        "dcp_kv_mask",
        "ngram_embedding_info",
    )
    populated_unsupported = [
        name
        for name in unsupported_fields
        if getattr(forward_batch, name, None) is not None
    ]
    if populated_unsupported:
        raise RuntimeError(
            "tree diagonal verifier found unsupported ForwardBatch fields: "
            + ", ".join(populated_unsupported)
        )

    expanded_rids = _repeat_single_optional_row(
        getattr(forward_batch, "rids", None), node_count
    )
    expanded_lora_ids = _repeat_single_optional_row(
        getattr(forward_batch, "lora_ids", None), node_count
    )
    expanded_orig_seq_lens = _repeat_single_optional_row(
        getattr(forward_batch, "orig_seq_lens", None), node_count
    )
    expanded_mm_inputs = _repeat_single_optional_row(
        getattr(forward_batch, "mm_inputs", None), node_count
    )

    # Populate all virtual page-table rows before changing the ForwardBatch.
    # Advanced-index assignment writes through to req_to_token; using copy_ on
    # an advanced-index result would instead modify a temporary tensor.
    prefix_columns = torch.arange(
        prefix_seq_len, dtype=torch.int64, device=req_to_token.device
    )
    canonical_rows = canonical_req_pool_indices.reshape(-1).to(torch.int64)
    virtual_rows = virtual_req_pool_indices.to(torch.int64)
    canonical_prefix = req_to_token[:, :prefix_seq_len].index_select(
        0, canonical_rows
    )
    req_to_token[
        virtual_rows[:, None],
        prefix_columns[None, :],
    ] = canonical_prefix

    node_columns = torch.arange(
        node_count, dtype=torch.int64, device=req_to_token.device
    )
    # Sort fixed-size keys so visible ancestors precede padding, preserving
    # node order within each path. Unlike nonzero(), this never needs the host
    # to learn a data-dependent output size. Sequence lengths hide zero padding.
    ancestor_order = torch.argsort(
        torch.where(ancestor_mask, node_columns, node_columns + node_count), dim=1
    )
    compact_path = torch.where(
        ancestor_mask.gather(1, ancestor_order),
        forward_batch.out_cache_loc[ancestor_order].to(req_to_token.dtype),
        0,
    )
    req_to_token[
        virtual_rows[:, None],
        (prefix_seq_len + node_columns)[None, :],
    ] = compact_path

    tree_seq_lens = (
        (forward_batch.positions + 1).to(forward_batch.seq_lens.dtype).contiguous()
    )

    forward_batch.forward_mode = ForwardMode.DECODE
    forward_batch.global_forward_mode = ForwardMode.DECODE
    forward_batch.batch_size = node_count
    forward_batch.req_pool_indices = virtual_req_pool_indices.contiguous()
    forward_batch.seq_lens = tree_seq_lens
    # The pinned ordinary-DECODE graph uses the device lengths.  Clearing the
    # one-request host mirror avoids presenting stale request-shaped metadata as
    # fourteen node-shaped rows and avoids a new device-to-host synchronization.
    forward_batch.seq_lens_cpu = None
    forward_batch.seq_lens_sum = None
    forward_batch.spec_info = None
    forward_batch.rids = expanded_rids
    forward_batch.lora_ids = expanded_lora_ids
    forward_batch.orig_seq_lens = expanded_orig_seq_lens
    forward_batch.mm_inputs = expanded_mm_inputs
    forward_batch.capture_hidden_mode = CaptureHiddenMode.FULL
    forward_batch._nextlat_diagonal_verify = True
    forward_batch._nextlat_tree_diagonal_verify = True

    forward_batch.extend_num_tokens = None
    forward_batch.extend_seq_lens = None
    forward_batch.extend_prefix_lens = None
    forward_batch.extend_start_loc = None
    forward_batch.extend_prefix_lens_cpu = None
    forward_batch.extend_seq_lens_cpu = None
    forward_batch.extend_logprob_start_lens_cpu = None

    return {
        "original_request_count": 1,
        "active_decode_rows": node_count,
        "graph_max_rows": diagonal_decode_max_rows(),
        "prefix_seq_len": prefix_seq_len,
        "page_size": page_size,
        "capture_hidden_mode": "FULL",
        "forward_mode": "DECODE",
        "spec_info_cleared": forward_batch.spec_info is None,
        "diagnostic_marker": bool(forward_batch._nextlat_diagonal_verify),
        "tree_marker": bool(forward_batch._nextlat_tree_diagonal_verify),
    }


@contextmanager
def _temporary_spec_algorithm(
    model_runner: Any, *, replacement: Any
) -> Iterator[None]:
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
def _temporary_attribute(owner: Any, name: str, replacement: Any) -> Iterator[None]:
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


@contextmanager
def _temporary_bf16_logits_buffer(model_runner: Any, runner: Any) -> Iterator[None]:
    """Give only the NEXTLAT target graph a graph-stable BF16 logits buffer."""

    if not target_bf16_logits_buffer_enabled():
        yield
        return
    shared = getattr(model_runner, "graph_shared_output", None)
    if shared is None:
        raise RuntimeError("NEXTLAT BF16 target logits require graph_shared_output")
    private_buffers: dict[int, torch.Tensor] = {}

    def get_logits_buffer(vocab_size: int, *, rows: int) -> torch.Tensor:
        if rows > int(shared.max_rows):
            raise RuntimeError(
                "NEXTLAT BF16 target logits exceed the shared graph row cap"
            )
        buffer = private_buffers.get(vocab_size)
        if buffer is None:
            buffer = torch.zeros(
                (int(shared.max_rows), vocab_size),
                dtype=torch.bfloat16,
                device=shared.device,
            )
            private_buffers[vocab_size] = buffer
        return buffer[:rows]

    runner._nextlat_bf16_logits_buffers = private_buffers
    with _temporary_attribute(shared, "get_logits_buffer", get_logits_buffer):
        yield


def _install_bf16_logits_copy_overlay(logits_processor_class: Any) -> None:
    """Preserve BF16 graph logits instead of widening them to FP32."""

    original_copy = logits_processor_class._copy_logits_to_buffer

    def copy_logits_to_buffer(
        processor: Any,
        logits: torch.Tensor,
        logits_metadata: Any,
        use_buffer: bool = True,
    ) -> torch.Tensor:
        buffer = (
            logits_metadata.next_token_logits_buffer if use_buffer else None
        )
        if buffer is None or buffer.dtype != torch.bfloat16:
            return original_copy(
                processor, logits, logits_metadata, use_buffer=use_buffer
            )
        if logits.shape[-1] > processor.vocab_size:
            logits = logits[:, : processor.vocab_size]
        if tuple(buffer.shape) != tuple(logits.shape):
            raise RuntimeError(
                "NEXTLAT BF16 graph logits buffer shape differs from LM-head output"
            )
        if logits.dtype != torch.bfloat16:
            raise RuntimeError(
                "NEXTLAT BF16 graph logits expected a BF16 LM-head output"
            )
        buffer.copy_(logits)
        return buffer

    logits_processor_class._copy_logits_to_buffer = copy_logits_to_buffer


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
    if runner is None or not _is_nextlat_runner(runner):
        raise RuntimeError("diagonal verifier has no restored NEXTLAT graph runner")
    rows = diagonal_decode_max_rows()
    expected = {
        "capture_forward_mode": "DECODE",
        "capture_hidden_mode": "FULL",
        "captured_req_width": 1,
        "capture_bs": [rows],
        "capture_num_tokens": None,
        "max_bs": rows,
        "ragged_verify_mode": False,
        "runtime_spec_algorithm": "NEXTLAT",
    }
    observed = _runner_contract(runner)
    if observed != expected:
        raise RuntimeError(
            f"diagonal verifier graph contract differs: {observed!r} != {expected!r}"
        )
    output = getattr(getattr(runner, "buffers", None), "next_token_logits_buffer", None)
    if (
        not isinstance(output, torch.Tensor)
        or output.ndim != 2
        or output.shape[0] < rows
    ):
        raise RuntimeError("diagonal verifier lacks its static logits buffer")
    return {**observed, "static_logits_buffer_shape": list(output.shape)}


def _is_admitted_diagonal_geometry(forward_batch: Any, *, max_rows: int) -> bool:
    mode = getattr(forward_batch, "forward_mode", None)
    input_ids = getattr(forward_batch, "input_ids", None)
    out_cache_loc = getattr(forward_batch, "out_cache_loc", None)
    positions = getattr(forward_batch, "positions", None)
    seq_lens = getattr(forward_batch, "seq_lens", None)
    req_pool_indices = getattr(forward_batch, "req_pool_indices", None)
    batch_size = getattr(forward_batch, "batch_size", None)
    fixed_dense = bool(
        getattr(forward_batch, "_nextlat_fixed_dense_m8_verify", False)
        and batch_size == max_rows
        and max_rows in _ADMITTED_MAX_ROWS
    )
    fixed_cuda_tree = bool(
        getattr(forward_batch, "_nextlat_tree_diagonal_verify", False)
        and isinstance(input_ids, torch.Tensor)
        and input_ids.device.type == "cuda"
        and batch_size == max_rows == TREE_NODE_BUDGET
    )
    return bool(
        getattr(mode, "name", None) == "DECODE"
        and getattr(forward_batch, "_nextlat_diagonal_verify", False) is True
        and getattr(forward_batch, "spec_info", None) is None
        and isinstance(batch_size, int)
        and 1 <= batch_size <= max_rows
        and all(
            isinstance(value, torch.Tensor) and value.numel() == batch_size
            for value in (
                input_ids,
                out_cache_loc,
                positions,
                seq_lens,
                req_pool_indices,
            )
        )
        # Both fixed producers derive lengths directly from positions. Avoid
        # re-proving that relation with a host synchronization on every replay.
        # CPU tree inputs retain the value check, as does M8's initial bootstrap.
        and (
            fixed_dense
            or fixed_cuda_tree
            or torch.equal(positions, seq_lens.to(positions.dtype) - 1)
        )
    )


def diagonal_decode_replay_snapshot(runner: Any) -> dict[str, Any]:
    contract = assert_captured_runner(runner)
    rows = diagonal_decode_max_rows()
    key = getattr(runner, "_replay_graph_key", None)
    replay = {
        "raw_bs": getattr(runner, "raw_bs", None),
        "padded_bs": getattr(runner, "bs", None),
        "raw_rows": getattr(runner, "raw_num_token", None),
        "graph_key_size": getattr(key, "size", None),
        "diagonal_decode_admission_count": getattr(
            runner, "_nextlat_diagonal_decode_admission_count", None
        ),
        "target_verify_rejection_count": getattr(
            runner, "_nextlat_target_verify_rejection_count", None
        ),
    }
    raw = replay["raw_bs"]
    if (
        not isinstance(raw, int)
        or not 1 <= raw <= rows
        or replay["raw_rows"] != raw
        or replay["padded_bs"] != rows
        or replay["graph_key_size"] != rows
    ):
        raise RuntimeError(f"diagonal verifier graph replay drifted: {replay!r}")
    if (
        not isinstance(replay["diagonal_decode_admission_count"], int)
        or replay["diagonal_decode_admission_count"] <= 0
    ):
        raise RuntimeError("diagonal verifier recorded no admitted replay")
    if (
        not isinstance(replay["target_verify_rejection_count"], int)
        or replay["target_verify_rejection_count"] <= 0
    ):
        raise RuntimeError("diagonal verifier lacks rejected TARGET_VERIFY evidence")
    return {"capture_contract": contract, "latest_replay": replay}


# Keep the worker overlay's pre-existing import name small.
replay_snapshot = diagonal_decode_replay_snapshot


def install_diagonal_decode_graph_overlay() -> None:
    global _PATCHED
    if not diagonal_decode_verify_enabled() or _PATCHED:
        return

    from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
    from sglang.srt.model_executor.runner import base_runner as base_runner_module
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
        DecodeCudaGraphRunner,
    )
    from sglang.srt.layers.logits_processor import LogitsProcessor
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    if target_bf16_logits_buffer_enabled():
        _install_bf16_logits_copy_overlay(LogitsProcessor)

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
            raise RuntimeError("diagonal verifier cannot capture a draft worker")
        with (
            _temporary_spec_algorithm(model_runner, replacement=no_spec_algorithm),
            _temporary_attribute(
                base_runner_module,
                "get_server_return_hidden_states_mode",
                lambda: full_hidden_mode,
            ),
            _temporary_bf16_logits_buffer(model_runner, runner),
        ):
            original_init(
                runner,
                model_runner,
                attn_backend=attn_backend,
                speculative_num_steps=0,
                speculative_num_draft_tokens=1,
            )
        runner._nextlat_diagonal_decode_admission_count = 0
        runner._nextlat_target_verify_rejection_count = 0
        assert_captured_runner(runner)

    def can_run_graph(runner: Any, forward_batch: Any):
        if not _is_nextlat_runner(runner):
            return original_can_run_graph(runner, forward_batch)
        rows = diagonal_decode_max_rows()
        if _is_admitted_diagonal_geometry(forward_batch, max_rows=rows):
            observed = bool(original_can_run_graph(runner, forward_batch))
            if not observed:
                raise RuntimeError("base graph runner rejected diagonal DECODE")
            runner._nextlat_diagonal_decode_admission_count += 1
            return True
        if getattr(
            getattr(forward_batch, "forward_mode", None), "name", None
        ) == "TARGET_VERIFY":
            runner._nextlat_target_verify_rejection_count += 1
        return False

    runner_class.__init__ = init
    runner_class.can_run_graph = can_run_graph
