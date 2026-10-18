"""Allocation-free verifier geometry for a fixed dense M8 throughput lane.

The ordinary-DECODE verifier graph already owns pointer-stable input
buffers.  After one canonical DFlash bootstrap, this module reuses that
ForwardBatch and writes the changing row values straight into those buffers
with one device launch.  Tail and non-dense batches remain on the canonical path.
"""

from __future__ import annotations

import os
from typing import Any

import torch
import triton
import triton.language as tl


ENABLED_ENV = "NEXTLAT_FIXED_B8_M8_VERIFY_GEOMETRY"
M4_ENABLED_ENV = "NEXTLAT_FIXED_B1_M4_VERIFY_GEOMETRY"
M4_B32_ENABLED_ENV = "NEXTLAT_FIXED_B32_M4_VERIFY_GEOMETRY"
M16_ENABLED_ENV = "NEXTLAT_FIXED_M16_VERIFY_GEOMETRY"
DRAFT_TOKENS = 8
VERIFY_WIDTH = DRAFT_TOKENS + 1
ADMITTED_VERIFY_ROWS = (9, 72, 288, 576, 1152)
M4_ADMITTED_VERIFY_ROWS = (5, 160)
M16_ADMITTED_VERIFY_ROWS = (17, 544)


def fixed_b8_m8_verify_enabled() -> bool:
    value = os.environ.get(ENABLED_ENV)
    if value is None:
        return False
    if value != "1":
        raise RuntimeError(f"{ENABLED_ENV} must be exactly '1' when present")
    return True


def fixed_b1_m4_verify_enabled() -> bool:
    value = os.environ.get(M4_ENABLED_ENV)
    if value is None:
        return False
    if value != "1":
        raise RuntimeError(f"{M4_ENABLED_ENV} must be exactly '1' when present")
    return True


def fixed_b32_m4_verify_enabled() -> bool:
    value = os.environ.get(M4_B32_ENABLED_ENV)
    if value is None:
        return False
    if value != "1":
        raise RuntimeError(
            f"{M4_B32_ENABLED_ENV} must be exactly '1' when present"
        )
    return True


def fixed_m16_verify_enabled() -> bool:
    value = os.environ.get(M16_ENABLED_ENV)
    if value is None:
        return False
    if value != "1":
        raise RuntimeError(f"{M16_ENABLED_ENV} must be exactly '1' when present")
    return True


@triton.jit
def _populate_fixed_dense_m8_verify_kernel(
    anchors,
    drafts,
    prefix_seq_lens,
    batch_req_pool_indices,
    req_to_token,
    input_ids_out,
    positions_out,
    seq_lens_out,
    req_pool_indices_out,
    out_cache_loc_out,
    req_to_token_stride,
    VERIFY_ROWS_CONST: tl.constexpr,
    VERIFY_WIDTH_CONST: tl.constexpr,
    DRAFT_TOKENS_CONST: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.arange(0, BLOCK)
    mask = row < VERIFY_ROWS_CONST
    request = row // VERIFY_WIDTH_CONST
    within = row - request * VERIFY_WIDTH_CONST

    prefix = tl.load(prefix_seq_lens + request, mask=mask, other=0).to(tl.int64)
    req_pool_index = tl.load(
        batch_req_pool_indices + request, mask=mask, other=0
    ).to(tl.int64)
    draft_column = tl.maximum(within - 1, 0)
    anchor = tl.load(anchors + request, mask=mask, other=0)
    draft = tl.load(
        drafts + request * DRAFT_TOKENS_CONST + draft_column, mask=mask, other=0
    )
    token = tl.where(within == 0, anchor, draft)
    position = prefix + within
    cache_loc = tl.load(
        req_to_token + req_pool_index * req_to_token_stride + position,
        mask=mask,
        other=0,
    )

    tl.store(input_ids_out + row, token, mask=mask)
    tl.store(positions_out + row, position, mask=mask)
    tl.store(seq_lens_out + row, position + 1, mask=mask)
    tl.store(req_pool_indices_out + row, req_pool_index, mask=mask)
    tl.store(out_cache_loc_out + row, cache_loc, mask=mask)


class FixedB8M8VerifyGeometry:
    """One-time DFlash bootstrap followed by static-buffer device refreshes."""

    def __init__(self, *, verify_rows: int, draft_tokens: int = DRAFT_TOKENS) -> None:
        admitted = (
            draft_tokens == DRAFT_TOKENS and verify_rows in ADMITTED_VERIFY_ROWS
        ) or (
            draft_tokens == 4 and verify_rows in M4_ADMITTED_VERIFY_ROWS
        ) or (
            draft_tokens == 16 and verify_rows in M16_ADMITTED_VERIFY_ROWS
        ) or (
            1 <= draft_tokens <= 16
            and verify_rows == draft_tokens + 1
        ) or (
            1 <= draft_tokens <= 16
            and verify_rows == 32 * (draft_tokens + 1)
        )
        if not admitted:
            raise RuntimeError(
                "fixed dense verifier requires B1 or B32 with M in [1,16], "
                "or an admitted wider M8 row count"
            )
        self.draft_tokens = draft_tokens
        self.verify_width = draft_tokens + 1
        self.verify_rows = verify_rows
        self.batch_size = verify_rows // self.verify_width
        self.verify_lens_cpu = (self.verify_width,) * self.batch_size
        self.total_verify_tokens = verify_rows
        self._runner: Any | None = None
        self._forward_batch: Any | None = None
        self._verify_input: Any | None = None
        self._layout: Any | None = None

    @property
    def is_bound(self) -> bool:
        return self._forward_batch is not None

    def bind(
        self,
        *,
        runner: Any,
        forward_batch: Any,
        verify_input: Any,
        layout: Any,
    ) -> None:
        """Bind a canonical first-round batch to captured graph storage."""

        from sglang.srt.model_executor.forward_batch_info import (
            CaptureHiddenMode,
            ForwardMode,
        )

        if self.is_bound:
            raise RuntimeError("fixed dense M8 verifier geometry was bound twice")
        if runner is None or getattr(runner, "max_bs", None) != self.verify_rows:
            raise RuntimeError(
                f"fixed dense M8 verifier requires the {self.verify_rows}-row graph"
            )
        if getattr(runner, "captured_req_width", None) != 1:
            raise RuntimeError("fixed dense M8 verifier requires q_len=1 capture")
        if getattr(forward_batch.forward_mode, "name", None) != "DECODE":
            raise RuntimeError("fixed dense M8 bootstrap did not reach DECODE")
        if int(forward_batch.batch_size) != self.verify_rows:
            raise RuntimeError(
                f"fixed dense M8 bootstrap did not produce {self.verify_rows} rows"
            )
        if forward_batch.capture_hidden_mode != CaptureHiddenMode.FULL:
            raise RuntimeError("fixed dense M8 verifier requires FULL hidden states")

        buffers = runner.buffers
        required = (
            "input_ids",
            "positions",
            "seq_lens",
            "req_pool_indices",
            "out_cache_loc",
        )
        for name in required:
            tensor = getattr(buffers, name, None)
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.numel() < self.verify_rows
            ):
                raise RuntimeError(
                    f"{self.verify_rows}-row graph lacks static {name} storage"
                )

        forward_batch.forward_mode = ForwardMode.DECODE
        forward_batch.global_forward_mode = ForwardMode.DECODE
        forward_batch.batch_size = self.verify_rows
        forward_batch.input_ids = buffers.input_ids[: self.verify_rows]
        forward_batch.positions = buffers.positions[: self.verify_rows]
        forward_batch.seq_lens = buffers.seq_lens[: self.verify_rows]
        forward_batch.req_pool_indices = buffers.req_pool_indices[: self.verify_rows]
        forward_batch.out_cache_loc = buffers.out_cache_loc[: self.verify_rows]
        forward_batch.orig_seq_lens = forward_batch.seq_lens
        forward_batch.seq_lens_cpu = None
        forward_batch.seq_lens_sum = None
        forward_batch.spec_info = None
        forward_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        forward_batch.forward_metadata_ready = False
        forward_batch.forward_metadata_planned_bs = None
        forward_batch.forward_metadata_planned_num_tokens = None
        forward_batch.forward_metadata_replan_equivalent = False
        forward_batch._nextlat_diagonal_verify = True
        forward_batch._nextlat_fixed_dense_m8_verify = True

        verify_input.draft_token = forward_batch.input_ids
        verify_input.positions = forward_batch.positions
        verify_input.ragged_verify_layout = layout
        verify_input.live_seq_lens_cpu = None
        self._runner = runner
        self._forward_batch = forward_batch
        self._verify_input = verify_input
        self._layout = layout

    def prepare(self, *, batch: Any, proposal: Any, req_to_token: torch.Tensor) -> Any:
        """Refresh all dynamic verifier rows on device and return the template.

        This runs after SGLang's canonical ``alloc_for_spec_decode`` scheduler
        step.  That allocator owns fresh-slot reservation and writes the current
        speculative window into ``req_to_token``; the kernel below only gathers
        those newly reserved locations into pointer-stable graph storage.
        """

        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        if not self.is_bound or self._runner is None or self._verify_input is None:
            raise RuntimeError("fixed dense M8 verifier geometry is not bound")
        if len(batch.reqs) != self.batch_size:
            raise RuntimeError(
                "fixed verifier received a batch outside its dense lane"
            )
        tensors = {
            "anchors": proposal.anchor_tokens,
            "drafts": proposal.draft_tokens,
            "prefix_seq_lens": batch.seq_lens,
            "batch_req_pool_indices": batch.req_pool_indices,
            "req_to_token": req_to_token,
        }
        expected_shapes = {
            "anchors": (self.batch_size,),
            "drafts": (self.batch_size, self.draft_tokens),
            "prefix_seq_lens": (self.batch_size,),
            "batch_req_pool_indices": (self.batch_size,),
        }
        for name, tensor in tensors.items():
            if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cuda":
                raise RuntimeError(f"fixed verifier {name} must be a CUDA tensor")
            if not tensor.is_contiguous():
                raise RuntimeError(f"fixed verifier {name} must be contiguous")
            expected = expected_shapes.get(name)
            if expected is not None and tuple(tensor.shape) != expected:
                raise RuntimeError(
                    f"fixed verifier {name} shape {tuple(tensor.shape)} != {expected}"
                )
        if req_to_token.ndim != 2:
            raise RuntimeError("fixed verifier req_to_token must be rank two")

        buffers = self._runner.buffers
        _populate_fixed_dense_m8_verify_kernel[(1,)](
            proposal.anchor_tokens,
            proposal.draft_tokens,
            batch.seq_lens,
            batch.req_pool_indices,
            req_to_token,
            buffers.input_ids,
            buffers.positions,
            buffers.seq_lens,
            buffers.req_pool_indices,
            buffers.out_cache_loc,
            req_to_token.stride(0),
            VERIFY_ROWS_CONST=self.verify_rows,
            VERIFY_WIDTH_CONST=self.verify_width,
            DRAFT_TOKENS_CONST=self.draft_tokens,
            BLOCK=triton.next_power_of_2(self.verify_rows),
        )

        # Preserve the ScheduleBatch mutations made by canonical DFlash verify;
        # downstream KV commit code continues to see the same logical batch.
        batch.input_ids = buffers.input_ids[: self.verify_rows]
        batch.out_cache_loc = buffers.out_cache_loc[: self.verify_rows]
        batch.spec_info = self._verify_input
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        self._verify_input.live_seq_lens_cpu = batch.seq_lens_cpu
        return self._forward_batch

    def trace_geometry(self) -> dict[str, Any]:
        return {
            "mode": f"fixed_dense_m{self.draft_tokens}_on_device",
            "original_request_count": self.batch_size,
            "active_decode_rows": self.verify_rows,
            "graph_max_rows": self.verify_rows,
            "verify_lens": list(self.verify_lens_cpu),
            "capture_hidden_mode": "FULL",
            "forward_mode": "DECODE",
            "spec_info_cleared": True,
            "diagnostic_marker": True,
        }
