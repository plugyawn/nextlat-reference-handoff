"""SGLang custom-spec registry glue for the narrow NEXTLAT algorithm."""

from __future__ import annotations

from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_registry import CustomSpecAlgo

from .config import top2_tree_verify_enabled, validate_server_args
from .diagonal_decode_verify import (
    diagonal_decode_verify_enabled,
    install_diagonal_decode_graph_overlay,
)
from .native_ragged_verify import (
    install_native_ragged_verify_graph_overlay,
    native_ragged_verify_enabled,
)
from .top2_tree_verify_graph import install_top2_tree_verify_graph_overlay

_REGISTERED = False


class NextLatSpecAlgo(CustomSpecAlgo):
    """Fill the pinned custom-registry methods used by the V2 scheduler."""

    def is_ngram(self) -> bool:
        # FutureMap's NGRAM branch relays accepted-token trie state, whereas
        # NextLat must relay its exact recurrent hidden state.  Target-only KV
        # allocation is expressed directly by has_draft_kv() below.
        return False

    def supports_ragged_verify(self) -> bool:
        # Linear NextLat uses DSpark's ragged window.  The opt-in two-line
        # experiment instead supplies an EAGLE-compatible irregular tree mask.
        return not top2_tree_verify_enabled()

    def has_draft_kv(self) -> bool:
        return False

    def carries_draft_hidden_states(self) -> bool:
        # Disaggregation is rejected by config validation.
        return False

    def need_topk(self) -> bool:
        # NextLat relays only the bonus token and recurrent hidden state.
        return False

    def create_future_map(
        self,
        device,
        req_to_token_pool,
        needs_cpu_seq_lens: bool = True,
        needs_confidence_relay: bool = False,
    ):
        from sglang.srt.managers.overlap_utils import FutureMap

        return FutureMap(
            device,
            self,
            req_to_token_pool,
            needs_cpu_seq_lens,
            needs_confidence_relay,
        )


def _worker_factory(server_args):
    del server_args
    import os

    if os.environ.get("NEXTLAT_PIPELINE_DRAFT_DEVICE"):
        from .pipeline_worker import PipelinedNextLatWorker

        return PipelinedNextLatWorker
    from .worker import NextLatWorker

    return NextLatWorker


def register() -> None:
    """Register exactly once in each SGLang process."""

    global _REGISTERED
    if _REGISTERED:
        return
    # Graph overlays are no-ops unless their explicit one-shape environment is
    # admitted.  Install before the target ModelRunner constructs/captures its
    # decode graph.
    if top2_tree_verify_enabled() and not diagonal_decode_verify_enabled():
        install_top2_tree_verify_graph_overlay()
    elif native_ragged_verify_enabled():
        install_native_ragged_verify_graph_overlay()
    else:
        install_diagonal_decode_graph_overlay()
    SpeculativeAlgorithm.register(
        "NEXTLAT",
        supports_overlap=True,
        validate_server_args=validate_server_args,
        spec_class=NextLatSpecAlgo,
    )(_worker_factory)
    _REGISTERED = True
