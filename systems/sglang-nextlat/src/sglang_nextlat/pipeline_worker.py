"""Two-device throughput cut for fixed dense B1/B8/B32/B64/B128 NEXTLAT.

The target model and its canonical KV cache stay on ``cuda:0``.  The recurrent
sidecar, copied tied embedding/head, and draft CUDA graph live on ``cuda:1``.
Independent SGLang workers use two independent stage locks so one worker's
draft graph can run on GPU1 while another worker owns the target verifier on
GPU0.  The cohort
count is configured by ``NEXTLAT_PIPELINE_COHORTS`` (default two, maximum eight);
one admits a single-process two-GPU lane.  The dense batch is configured by
``NEXTLAT_PIPELINE_BATCH_SIZE`` (default eight; admitted values one, eight, 32,
64, and 128).
Ordinary single-worker behavior is unchanged unless
``NEXTLAT_PIPELINE_DRAFT_DEVICE`` is set.
"""

from __future__ import annotations

import fcntl
import mmap
import os
import struct
import time
from multiprocessing.connection import Client, Listener
from pathlib import Path

import torch
import torch.nn.functional as F

from .checkpoint import load_dynamics_sidecar
from .constants import QWEN3_1_7B_CONTRACT
from .worker import NextLatWorker


_MAGIC = b"NLP1"
_HEADER_BYTES = 4096
_RECORD = struct.Struct("<QQQQQQq")
_MAX_RECORDS = 8192
_PIPELINE_BATCH_SIZE_ENV = "NEXTLAT_PIPELINE_BATCH_SIZE"
_PIPELINE_TRACE_ENV = "NEXTLAT_PIPELINE_TRACE"
_PIPELINE_EVENT_HANDOFF_ENV = "NEXTLAT_PIPELINE_EVENT_HANDOFF"
_PIPELINE_TARGET_MUTEX_ENV = "NEXTLAT_PIPELINE_TARGET_MUTEX"
_SHARED_DRAFT_OWNER_ENV = "NEXTLAT_SHARED_DRAFT_OWNER_ROLE"
_SHARED_READY_OFFSET = 128
_SHARED_DONE_OFFSET = 256
_ADMITTED_PIPELINE_BATCH_SIZES = frozenset((1, 8, 32, 64, 128))


class _CohortTurnstile:
    def __init__(self, *, role: int, cohorts: int, path: Path, reset: bool) -> None:
        if not 1 <= cohorts <= 8:
            raise RuntimeError("NEXTLAT_PIPELINE_COHORTS must be in [1, 8]")
        if not 0 <= role < cohorts:
            raise RuntimeError(
                "NEXTLAT_PIPELINE_ROLE must be in "
                f"[0, NEXTLAT_PIPELINE_COHORTS), got {role}"
            )
        self.role = role
        self.cohorts = cohorts
        self.path = path
        size = _HEADER_BYTES + cohorts * _MAX_RECORDS * _RECORD.size
        path.parent.mkdir(parents=True, exist_ok=True)
        if reset:
            fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o600)
            os.ftruncate(fd, size)
            mapping = mmap.mmap(fd, size)
            mapping[0:4] = _MAGIC
            mapping[4] = 0  # draft turn
            mapping[5] = 0  # target turn
            for cohort in range(cohorts):
                struct.pack_into("<Q", mapping, _SHARED_READY_OFFSET + 8 * cohort, 0)
            struct.pack_into("<Q", mapping, _SHARED_DONE_OFFSET, 0)
            mapping.flush()
        else:
            deadline = time.monotonic() + 120.0
            while True:
                try:
                    fd = os.open(path, os.O_RDWR)
                    if os.fstat(fd).st_size == size:
                        mapping = mmap.mmap(fd, size)
                        if mapping[0:4] == _MAGIC:
                            break
                    os.close(fd)
                except FileNotFoundError:
                    pass
                if time.monotonic() >= deadline:
                    raise RuntimeError("timed out waiting for NEXTLAT pipeline state")
                time.sleep(0.01)
        self._fd = fd
        self._mapping = mapping
        self._cycle = 0
        self._draft_wait_ns = 0
        self._draft_start_ns = 0
        self._draft_end_ns = 0
        self._target_start_ns = 0
        self._target_end_ns = 0

    def _wait(self, offset: int) -> None:
        deadline = time.monotonic() + 120.0
        while self._mapping[offset] != self.role:
            if self._mapping[offset] == 255:
                raise RuntimeError("NEXTLAT multi-cohort pipeline was disabled by peer")
            if time.monotonic() >= deadline:
                raise RuntimeError("NEXTLAT multi-cohort pipeline turn timed out")
            os.sched_yield()

    def begin_draft(self) -> None:
        self._draft_wait_ns = time.perf_counter_ns()
        self._wait(4)
        self._draft_start_ns = time.perf_counter_ns()

    def finish_draft(self) -> None:
        self._draft_end_ns = time.perf_counter_ns()
        self._mapping[4] = (self.role + 1) % self.cohorts

    def acquire_target(self) -> None:
        # The verifier is the bottleneck, so admit whichever cohort is ready
        # instead of idling behind a fixed role order.  Lock byte 5 preserves
        # the existing mmap layout while providing an atomic process mutex.
        fcntl.lockf(self._fd, fcntl.LOCK_EX, 1, 5, os.SEEK_SET)
        if self._mapping[5] == 255:
            fcntl.lockf(self._fd, fcntl.LOCK_UN, 1, 5, os.SEEK_SET)
            raise RuntimeError("NEXTLAT multi-cohort pipeline was disabled by peer")
        self._target_start_ns = time.perf_counter_ns()

    def release_target(self) -> None:
        self._target_end_ns = time.perf_counter_ns()
        fcntl.lockf(self._fd, fcntl.LOCK_UN, 1, 5, os.SEEK_SET)

    def finish_cycle(self, emitted_tokens: int) -> None:
        if self._cycle >= _MAX_RECORDS:
            raise RuntimeError("NEXTLAT pipeline trace capacity exhausted")
        base = _HEADER_BYTES + self.role * _MAX_RECORDS * _RECORD.size
        offset = base + self._cycle * _RECORD.size
        _RECORD.pack_into(
            self._mapping,
            offset,
            self._cycle,
            self._draft_wait_ns,
            self._draft_start_ns,
            self._draft_end_ns,
            self._target_start_ns,
            self._target_end_ns,
            emitted_tokens,
        )
        self._cycle += 1
        struct.pack_into("<Q", self._mapping, 8 + self.role * 8, self._cycle)

    def disable(self) -> None:
        self._mapping[4] = 255
        self._mapping[5] = 255

    def shared_arrive(self, cycle: int) -> None:
        struct.pack_into(
            "<Q", self._mapping, _SHARED_READY_OFFSET + 8 * self.role, cycle
        )

    def shared_wait_all(self, cycle: int) -> None:
        deadline = time.monotonic() + 120.0
        while any(
            struct.unpack_from(
                "<Q", self._mapping, _SHARED_READY_OFFSET + 8 * role
            )[0]
            < cycle
            for role in range(self.cohorts)
        ):
            if time.monotonic() >= deadline:
                raise RuntimeError("NEXTLAT shared draft input barrier timed out")
            os.sched_yield()

    def shared_publish_done(self, cycle: int) -> None:
        struct.pack_into("<Q", self._mapping, _SHARED_DONE_OFFSET, cycle)

    def shared_wait_done(self, cycle: int) -> None:
        deadline = time.monotonic() + 120.0
        while struct.unpack_from(
            "<Q", self._mapping, _SHARED_DONE_OFFSET
        )[0] < cycle:
            if time.monotonic() >= deadline:
                raise RuntimeError("NEXTLAT shared draft output barrier timed out")
            os.sched_yield()

class PipelinedNextLatWorker(NextLatWorker):
    """Fixed dense M4/M8 worker with a dedicated recurrent GPU."""

    def __init__(self, server_args, gpu_id: int, ps, nccl_port: int, target_worker):
        super().__init__(server_args, gpu_id, ps, nccl_port, target_worker)
        device_value = os.environ.get("NEXTLAT_PIPELINE_DRAFT_DEVICE")
        if not device_value:
            raise RuntimeError("pipelined worker requires NEXTLAT_PIPELINE_DRAFT_DEVICE")
        self._pipeline_target_device = torch.device(
            "cuda",
            self.device.index
            if self.device.index is not None
            else torch.cuda.current_device(),
        )
        self._pipeline_draft_device = torch.device(device_value)
        if (
            self._pipeline_draft_device.type != "cuda"
            or self._pipeline_draft_device == self._pipeline_target_device
        ):
            raise RuntimeError("pipeline draft device must be a distinct CUDA device")
        if not torch.cuda.can_device_access_peer(
            self._pipeline_target_device.index or 0,
            self._pipeline_draft_device.index or 0,
        ):
            raise RuntimeError("target and draft GPUs lack CUDA peer access")
        if self.max_draft_tokens not in (4, 8):
            raise RuntimeError("two-GPU pipeline supports only dense M4 or M8")
        if self.verify_width != self.max_draft_tokens + 1:
            raise RuntimeError("two-GPU pipeline verify width is inconsistent")
        batch_size_value = os.environ.get(_PIPELINE_BATCH_SIZE_ENV, "8")
        if (
            batch_size_value not in {str(value) for value in _ADMITTED_PIPELINE_BATCH_SIZES}
        ):
            raise RuntimeError(
                f"{_PIPELINE_BATCH_SIZE_ENV} must be exactly '1', '8', '32', "
                "'64', or '128'"
            )
        self._pipeline_batch_size = int(batch_size_value)
        trace_value = os.environ.get(_PIPELINE_TRACE_ENV, "1")
        if trace_value not in {"0", "1"}:
            raise RuntimeError(f"{_PIPELINE_TRACE_ENV} must be exactly '0' or '1'")
        self._pipeline_trace_enabled = trace_value == "1"
        target_mutex_value = os.environ.get(_PIPELINE_TARGET_MUTEX_ENV, "1")
        if target_mutex_value not in {"0", "1"}:
            raise RuntimeError(f"{_PIPELINE_TARGET_MUTEX_ENV} must be '0' or '1'")
        self._pipeline_target_mutex = target_mutex_value == "1"
        event_handoff_value = os.environ.get(_PIPELINE_EVENT_HANDOFF_ENV)
        if event_handoff_value not in (None, "1"):
            raise RuntimeError(
                f"{_PIPELINE_EVENT_HANDOFF_ENV} must be exactly '1' when present"
            )
        self._pipeline_event_handoff = event_handoff_value == "1"
        role = int(os.environ["NEXTLAT_PIPELINE_ROLE"])
        cohorts = int(os.environ.get("NEXTLAT_PIPELINE_COHORTS", "2"))
        shared_owner_value = os.environ.get(_SHARED_DRAFT_OWNER_ENV)
        if shared_owner_value not in (None, "0"):
            raise RuntimeError(f"{_SHARED_DRAFT_OWNER_ENV} must be exactly '0'")
        self._pipeline_shared_draft = shared_owner_value == "0"
        self._pipeline_shared_owner = self._pipeline_shared_draft and role == 0
        if self._pipeline_shared_draft and self.max_draft_tokens != 8:
            raise RuntimeError("shared seven-cohort draft service remains M8-only")
        if self._pipeline_shared_draft and (
            self._pipeline_batch_size != 32
            or cohorts != 7
            or self._pipeline_event_handoff
            or self._pipeline_target_mutex
        ):
            raise RuntimeError(
                "shared draft owner path requires B32, seven cohorts, "
                "target mutex off, and event handoff off"
            )
        if (
            self._target_verify_graph_diagnostic
            and self._draft_graph_batch_size != self._pipeline_batch_size
        ):
            raise RuntimeError(
                "two-GPU dense batch must equal diagonal verifier rows / "
                "verify width"
            )
        # ``NextLatWorker._decode`` uses ``_draft_graph_batch_size`` as the
        # logical per-worker admission size.  Keep that at B32 so each target
        # lane actually enters ``_replay_dense_draft_cuda_graph``; only the
        # owner graph and its IPC buffers are widened to the seven-lane B224
        # physical batch.
        self._pipeline_draft_capture_batch_size = (
            self._pipeline_batch_size * cohorts
            if self._pipeline_shared_draft
            else self._pipeline_batch_size
        )

        sidecar_path = server_args.speculative_draft_model_path
        with torch.cuda.device(self._pipeline_draft_device):
            if not self._pipeline_shared_draft or self._pipeline_shared_owner:
                loaded = load_dynamics_sidecar(
                    sidecar_path, device=self._pipeline_draft_device
                )
                self._pipeline_dynamics = loaded.dynamics.to(dtype=torch.bfloat16)
                embeddings = self.target_model.get_input_embeddings().weight.detach()
                self._pipeline_embedding_weight = embeddings.to(
                    self._pipeline_draft_device, non_blocking=False
                ).contiguous()
                self._pipeline_hot_token_id = self.hot_token_id.to(
                    self._pipeline_draft_device, non_blocking=False
                )
                self._pipeline_hot_lm_head_weight = self.hot_lm_head_weight.to(
                    self._pipeline_draft_device, non_blocking=False
                ).contiguous()
                if self._draft_fp8_enabled:
                    self._initialize_pipeline_draft_fp8_weights()
            else:
                self._pipeline_dynamics = None
                self._pipeline_embedding_weight = None
                self._pipeline_hot_token_id = None
                self._pipeline_hot_lm_head_weight = None
            self._pipeline_draft_stream = torch.cuda.Stream(
                device=self._pipeline_draft_device
            )
        if self._pipeline_event_handoff and (
            self._pipeline_batch_size != 1 or cohorts != 1
        ):
            raise RuntimeError(
                f"{_PIPELINE_EVENT_HANDOFF_ENV}=1 is specialized to "
                f"{_PIPELINE_BATCH_SIZE_ENV}=1 and NEXTLAT_PIPELINE_COHORTS=1"
            )
        state_path = Path(
            os.environ.get("NEXTLAT_PIPELINE_STATE", "/dev/shm/nextlat-pipeline.bin")
        )
        reset = os.environ.get("NEXTLAT_PIPELINE_RESET") == "1"
        self._pipeline_turnstile = _CohortTurnstile(
            role=role, cohorts=cohorts, path=state_path, reset=reset
        )
        self._pipeline_shared_role = role
        self._pipeline_shared_cohorts = cohorts
        self._pipeline_shared_cycle = 1
        self._pipeline_shared_socket = state_path.with_suffix(".sock")
        if self._pipeline_shared_owner and reset:
            self._pipeline_shared_socket.unlink(missing_ok=True)
        self._pipeline_target_owned = False
        # In the B1 latency lane, GPU-side events carry both cross-device data
        # dependencies.  This keeps the host out of the recurrent draft/target
        # critical path without letting GPU1 read the next recurrent seed before
        # GPU0 has produced it.
        self._pipeline_draft_ready_event = torch.cuda.Event(blocking=False)
        self._pipeline_seed_ready_event = torch.cuda.Event(blocking=False)
        self._pipeline_seed_ready_recorded = False

    def _prefill(self, batch, on_publish):
        # Prefill is a real target-model stage too.  Serializing it seeds the
        # intended phase offset: each next role can overlap the prior role's
        # first draft.
        if self._pipeline_target_mutex:
            self._pipeline_turnstile.acquire_target()
            self._pipeline_target_owned = True
        try:
            result = super()._prefill(batch, on_publish)
            torch.cuda.current_stream(self._pipeline_target_device).synchronize()
            self._pipeline_seed_ready_recorded = False
            return result
        finally:
            if self._pipeline_target_owned:
                self._pipeline_turnstile.release_target()
                self._pipeline_target_owned = False

    def _initialize_pipeline_draft_fp8_weights(self) -> None:
        """Quantize the GPU1 copies instead of replaying GPU0 FP8 weights.

        ``NextLatWorker`` initializes its ordinary colocated fallback before the
        pipeline placement is known.  The dense two-device lane therefore owns
        a second, device-local FP8 set.  Keeping the sets distinct also keeps
        the non-dense fallback valid on the target device.
        """

        from sglang.kernels.ops.quantization.fp8_kernel import (
            per_token_group_quant_fp8,
        )

        def quantize_transposed(weight: torch.Tensor):
            canonical = weight.detach().contiguous()
            quantized, scale = per_token_group_quant_fp8(
                canonical, group_size=canonical.shape[-1]
            )
            return quantized.T, scale.T.contiguous()

        with torch.no_grad():
            (
                self._pipeline_fp8_mlp0_weight,
                self._pipeline_fp8_mlp0_scale,
            ) = quantize_transposed(self._pipeline_dynamics.mlp[0].weight)
            (
                self._pipeline_fp8_mlp2_weight,
                self._pipeline_fp8_mlp2_scale,
            ) = quantize_transposed(self._pipeline_dynamics.mlp[2].weight)
            (
                self._pipeline_fp8_mlp4_weight,
                self._pipeline_fp8_mlp4_scale,
            ) = quantize_transposed(self._pipeline_dynamics.mlp[4].weight)
            (
                self._pipeline_fp8_head_weight,
                self._pipeline_fp8_head_scale,
            ) = quantize_transposed(self._pipeline_hot_lm_head_weight)

    def _pipeline_fp8_dynamics(
        self, current_states: torch.Tensor, token_embeddings: torch.Tensor
    ) -> torch.Tensor:
        joined = torch.cat((token_embeddings, current_states), dim=-1)
        hidden = self._pipeline_dynamics.norm(joined)
        hidden = F.gelu(
            self._draft_fp8_linear(
                hidden,
                self._pipeline_fp8_mlp0_weight,
                self._pipeline_fp8_mlp0_scale,
            )
        )
        hidden = F.gelu(
            self._draft_fp8_linear(
                hidden,
                self._pipeline_fp8_mlp2_weight,
                self._pipeline_fp8_mlp2_scale,
            )
        )
        return current_states + self._draft_fp8_linear(
            hidden,
            self._pipeline_fp8_mlp4_weight,
            self._pipeline_fp8_mlp4_scale,
        )

    def _dense_pipeline_draft_chain(
        self, current_states: torch.Tensor, anchor_tokens: torch.Tensor
    ) -> torch.Tensor:
        state = current_states
        input_tokens = anchor_tokens
        outputs: list[torch.Tensor] = []
        for _ in range(self.max_draft_tokens):
            token_embeddings = F.embedding(
                input_tokens, self._pipeline_embedding_weight
            )
            if self._draft_fp8_enabled:
                state = self._pipeline_fp8_dynamics(state, token_embeddings)
                logits = self._draft_fp8_linear(
                    state,
                    self._pipeline_fp8_head_weight,
                    self._pipeline_fp8_head_scale,
                )
            else:
                state = self._pipeline_dynamics(state, token_embeddings)
                logits = F.linear(state, self._pipeline_hot_lm_head_weight)
            reduced_ids = torch.argmax(logits, dim=-1).to(torch.int64)
            input_tokens = self._pipeline_hot_token_id[reduced_ids]
            outputs.append(input_tokens)
        return torch.stack(outputs, dim=1)

    def _capture_dense_draft_cuda_graph(self) -> None:
        if not self._draft_cuda_graph_enabled:
            return
        if self._pipeline_shared_draft and not self._pipeline_shared_owner:
            self._receive_shared_draft_buffers()
            return
        if self._draft_cuda_graph is not None:
            raise RuntimeError("NEXTLAT dense draft CUDA graph was captured twice")
        with torch.cuda.device(self._pipeline_draft_device):
            static_state = torch.zeros(
                (
                    self._pipeline_draft_capture_batch_size,
                    QWEN3_1_7B_CONTRACT.hidden_size,
                ),
                dtype=torch.bfloat16,
                device=self._pipeline_draft_device,
            )
            static_anchor = torch.zeros(
                (self._pipeline_draft_capture_batch_size,),
                dtype=torch.int64,
                device=self._pipeline_draft_device,
            )
            stream = self._pipeline_draft_stream
            stream.wait_stream(torch.cuda.current_stream(self._pipeline_draft_device))
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self._dense_pipeline_draft_chain(static_state, static_anchor)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                static_tokens = self._dense_pipeline_draft_chain(
                    static_state, static_anchor
                )
            stream.synchronize()
        self._draft_graph_state = static_state
        self._draft_graph_anchor = static_anchor
        self._draft_graph_tokens = static_tokens
        self._draft_cuda_graph = graph
        if self._pipeline_shared_owner:
            self._publish_shared_draft_buffers()

    def _publish_shared_draft_buffers(self) -> None:
        if (
            self._draft_graph_state is None
            or self._draft_graph_anchor is None
            or self._draft_graph_tokens is None
        ):
            raise RuntimeError("shared draft owner has no captured buffers")
        listener = Listener(str(self._pipeline_shared_socket), family="AF_UNIX")
        connected: set[int] = set()
        try:
            while len(connected) < self._pipeline_shared_cohorts - 1:
                connection = listener.accept()
                try:
                    role = int(connection.recv())
                    if (
                        role <= 0
                        or role >= self._pipeline_shared_cohorts
                        or role in connected
                    ):
                        raise RuntimeError(f"invalid shared draft client role {role}")
                    connection.send(
                        (
                            self._draft_graph_state,
                            self._draft_graph_anchor,
                            self._draft_graph_tokens,
                        )
                    )
                    connected.add(role)
                finally:
                    connection.close()
        finally:
            listener.close()

    def _receive_shared_draft_buffers(self) -> None:
        deadline = time.monotonic() + 300.0
        with torch.cuda.device(self._pipeline_draft_device):
            torch.cuda.init()
            # Force the shared physical-device primary context to exist before
            # Torch opens the exported allocation in ``connection.recv``.
            torch.empty((1,), device=self._pipeline_draft_device)
            while True:
                try:
                    connection = Client(
                        str(self._pipeline_shared_socket), family="AF_UNIX"
                    )
                    break
                except (FileNotFoundError, ConnectionRefusedError):
                    if time.monotonic() >= deadline:
                        raise RuntimeError("timed out connecting to shared draft owner")
                    time.sleep(0.01)
            try:
                connection.send(self._pipeline_shared_role)
                state, anchor, tokens = connection.recv()
            finally:
                connection.close()
        expected_state = (
            self._pipeline_draft_capture_batch_size,
            QWEN3_1_7B_CONTRACT.hidden_size,
        )
        expected_tokens = (expected_state[0], self.max_draft_tokens)
        if (
            tuple(state.shape) != expected_state
            or tuple(anchor.shape) != (expected_state[0],)
            or tuple(tokens.shape) != expected_tokens
            or state.device != self._pipeline_draft_device
            or anchor.device != self._pipeline_draft_device
            or tokens.device != self._pipeline_draft_device
        ):
            raise RuntimeError("shared draft CUDA IPC buffer contract mismatch")
        self._draft_graph_state = state
        self._draft_graph_anchor = anchor
        self._draft_graph_tokens = tokens

    def _replay_dense_draft_cuda_graph(
        self, current_states: torch.Tensor, anchor_tokens: torch.Tensor
    ) -> torch.Tensor:
        if self._pipeline_shared_draft:
            return self._replay_shared_dense_draft_cuda_graph(
                current_states, anchor_tokens
            )
        if (
            self._draft_cuda_graph is None
            or self._draft_graph_state is None
            or self._draft_graph_anchor is None
            or self._draft_graph_tokens is None
        ):
            raise RuntimeError("NEXTLAT dense draft CUDA graph is not initialized")
        turnstile = self._pipeline_turnstile
        turnstile.begin_draft()
        with torch.cuda.device(self._pipeline_draft_device):
            stream = self._pipeline_draft_stream
            with torch.cuda.stream(stream):
                if (
                    self._pipeline_event_handoff
                    and self._pipeline_seed_ready_recorded
                ):
                    stream.wait_event(self._pipeline_seed_ready_event)
                self._draft_graph_state.copy_(current_states, non_blocking=True)
                self._draft_graph_anchor.copy_(anchor_tokens, non_blocking=True)
                self._draft_cuda_graph.replay()
                target_tokens = self._draft_graph_tokens.to(
                    self._pipeline_target_device, non_blocking=True
                )
                if self._pipeline_event_handoff:
                    self._pipeline_draft_ready_event.record(stream)
            if not self._pipeline_event_handoff:
                stream.synchronize()
        if self._pipeline_event_handoff:
            # cudaStreamWaitEvent supports cross-device events.  All verifier
            # setup kernels subsequently issued on the target stream therefore
            # consume complete P2P draft tokens without a host fence.
            with torch.cuda.device(self._pipeline_target_device):
                torch.cuda.current_stream(
                    self._pipeline_target_device
                ).wait_event(self._pipeline_draft_ready_event)
        turnstile.finish_draft()
        return target_tokens

    def _replay_shared_dense_draft_cuda_graph(
        self, current_states: torch.Tensor, anchor_tokens: torch.Tensor
    ) -> torch.Tensor:
        if (
            self._draft_graph_state is None
            or self._draft_graph_anchor is None
            or self._draft_graph_tokens is None
            or (self._pipeline_shared_owner and self._draft_cuda_graph is None)
        ):
            raise RuntimeError("NEXTLAT shared dense draft graph is not initialized")
        cycle = self._pipeline_shared_cycle
        start = self._pipeline_shared_role * self._pipeline_batch_size
        end = start + self._pipeline_batch_size
        with torch.cuda.device(self._pipeline_draft_device):
            with torch.cuda.stream(self._pipeline_draft_stream):
                self._draft_graph_state[start:end].copy_(
                    current_states, non_blocking=True
                )
                self._draft_graph_anchor[start:end].copy_(
                    anchor_tokens, non_blocking=True
                )
            self._pipeline_draft_stream.synchronize()
        self._pipeline_turnstile.shared_arrive(cycle)
        if self._pipeline_shared_owner:
            self._pipeline_turnstile.shared_wait_all(cycle)
            with torch.cuda.device(self._pipeline_draft_device):
                with torch.cuda.stream(self._pipeline_draft_stream):
                    self._draft_cuda_graph.replay()
                self._pipeline_draft_stream.synchronize()
            self._pipeline_turnstile.shared_publish_done(cycle)
        else:
            self._pipeline_turnstile.shared_wait_done(cycle)
        with torch.cuda.device(self._pipeline_draft_device):
            with torch.cuda.stream(self._pipeline_draft_stream):
                target_tokens = self._draft_graph_tokens[start:end].to(
                    self._pipeline_target_device, non_blocking=True
                )
            self._pipeline_draft_stream.synchronize()
        self._pipeline_shared_cycle += 1
        return target_tokens

    def _before_target_verify_forward(self) -> None:
        if self._pipeline_target_mutex:
            self._pipeline_turnstile.acquire_target()
            self._pipeline_target_owned = True

    def _after_target_verify_forward(self) -> None:
        if not self._pipeline_event_handoff:
            torch.cuda.current_stream(self._pipeline_target_device).synchronize()
        if self._pipeline_target_owned:
            self._pipeline_turnstile.release_target()
            self._pipeline_target_owned = False

    def _decode(self, batch, on_publish):
        proposal_lens = self._proposal_lens_cpu(batch)
        dense = (
            len(batch.reqs) == self._pipeline_batch_size
            and proposal_lens
            == [self.max_draft_tokens] * self._pipeline_batch_size
        )
        if not dense:
            # Serialize the infrequent completion tail under the draft turn.
            # The target hooks still own the target forward itself.  Identical
            # cohort prompts/budgets guarantee both roles traverse this tail.
            self._pipeline_turnstile.begin_draft()
            try:
                return super()._decode(batch, on_publish)
            finally:
                self._pipeline_turnstile.finish_draft()
        try:
            result = super()._decode(batch, on_publish)
            if self._pipeline_event_handoff:
                # ``super()._decode`` has enqueued the selected-hidden gather
                # that forms the next recurrent seed.  The next GPU1 replay
                # waits on this marker before its peer copies.
                with torch.cuda.device(self._pipeline_target_device):
                    self._pipeline_seed_ready_event.record(
                        torch.cuda.current_stream(self._pipeline_target_device)
                    )
                self._pipeline_seed_ready_recorded = True
            if not self._pipeline_trace_enabled:
                return result
            # The target stage has already released its turn.  Synchronize only
            # to make the accepted-token count and full-cycle trace concrete.
            torch.cuda.current_stream(self._pipeline_target_device).synchronize()
            emitted = int(result.accept_lens.sum().item())
            self._pipeline_turnstile.finish_cycle(emitted)
            return result
        except BaseException:
            if self._pipeline_target_owned:
                self._pipeline_turnstile.release_target()
                self._pipeline_target_owned = False
            raise
