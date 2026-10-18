"""Width-eight recurrent beam packed into a two-direction verify tree.

This module deliberately stops at the draft/target boundary.  It produces the
fixed-width token bank and ancestor mask consumed by SGLang's existing EAGLE
tree verifier; target attention, greedy arbitration, KV movement, and state
refresh remain owned by SGLang and :mod:`sglang_nextlat.worker`.

The first systems experiment keeps the local-greedy recurrent line in full and
uses the remaining node budget for the highest-q distinct beam alternative.
The target, not q, chooses a route through the resulting tree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


TREE_BEAM_WIDTH = 8
TREE_HORIZON = 8
TREE_NODE_BUDGET = 14  # root + primary H8 line + five alternate nodes


@dataclass(frozen=True)
class Top2BeamTree:
    """One B1 irregular tree in SGLang node order.

    ``node_tokens[0]`` is the already-emitted exact anchor.  Rows ``1:H+1``
    contain the complete local-greedy line.  Remaining reachable rows contain
    a prefix of the highest-q distinct line after its shared prefix; unused
    rows are isolated dummy roots and can never be selected by the verifier.
    """

    node_tokens: torch.Tensor
    ancestor_mask: torch.Tensor
    selected_lines: torch.Tensor
    selected_log_q: torch.Tensor
    shared_prefix_length: torch.Tensor
    actual_node_count: torch.Tensor
    horizon: int
    node_budget: int

    def validate(self) -> None:
        if self.node_tokens.shape != (self.node_budget,):
            raise ValueError("node_tokens must have fixed node_budget width")
        if self.node_tokens.dtype not in (torch.int32, torch.int64):
            raise ValueError("node_tokens must contain integer token ids")
        if self.ancestor_mask.shape != (self.node_budget, self.node_budget):
            raise ValueError("ancestor_mask must be [node_budget, node_budget]")
        if self.ancestor_mask.dtype != torch.bool:
            raise ValueError("ancestor_mask must be boolean")
        if self.selected_lines.shape != (2, self.horizon):
            raise ValueError("selected_lines must contain exactly two H-token lines")
        if self.selected_log_q.shape != (2,):
            raise ValueError("selected_log_q must contain two cumulative scores")
        if any(
            tensor.device != self.node_tokens.device
            for tensor in (
                self.ancestor_mask,
                self.selected_lines,
                self.selected_log_q,
                self.shared_prefix_length,
                self.actual_node_count,
            )
        ):
            raise ValueError("all tree tensors must share one device")


def recurrent_beam_lines(
    *,
    current_state: torch.Tensor,
    anchor_token: torch.Tensor,
    dynamics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    embed_tokens: Callable[[torch.Tensor], torch.Tensor],
    project_hot_logits: Callable[[torch.Tensor], torch.Tensor],
    hot_token_ids: torch.Tensor,
    horizon: int,
    beam_width: int = TREE_BEAM_WIDTH,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return local-greedy plus the highest-q distinct beam direction.

    The local-greedy path is protected in the width-``beam_width`` frontier,
    replacing only its lowest-scoring row if ordinary global pruning would
    remove it.  The other returned line is the highest-q distinct row.  Thus a
    target tree can never lose the existing greedy proposal merely because the
    beam objective preferred two alternatives.  Scores are normalized over the
    deployed hot vocabulary.  This keeps the implementation compatible with
    the resident 32K head; it is intentionally not advertised as identical to
    the offline full-vocabulary normalization.  No host synchronization occurs
    in the recurrent loop.
    """

    if current_state.ndim != 2 or current_state.shape[0] != 1:
        raise ValueError("tree beam currently requires one recurrent state")
    if anchor_token.shape != (1,) or anchor_token.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("tree beam currently requires one integer anchor token")
    if not 0 <= horizon <= TREE_HORIZON:
        raise ValueError(f"horizon must be in [0, {TREE_HORIZON}]")
    if beam_width < 2:
        raise ValueError("tree beam requires width at least two")
    if (
        hot_token_ids.ndim != 1
        or hot_token_ids.dtype != torch.int64
        or hot_token_ids.device != current_state.device
        or hot_token_ids.numel() < beam_width
    ):
        raise ValueError("hot_token_ids must be an int64 support on the state device")

    if horizon == 0:
        return (
            torch.empty((2, 0), dtype=torch.int64, device=current_state.device),
            torch.zeros((2,), dtype=torch.float32, device=current_state.device),
        )

    states = current_state
    input_tokens = anchor_token.to(torch.int64)
    cumulative = torch.zeros((1,), dtype=torch.float32, device=current_state.device)
    paths = torch.empty((1, 0), dtype=torch.int64, device=current_state.device)
    support_size = int(hot_token_ids.numel())
    greedy_row = torch.zeros((), dtype=torch.int64, device=current_state.device)

    for _ in range(horizon):
        next_states = dynamics(states, embed_tokens(input_tokens))
        hot_logits = project_hot_logits(next_states)
        if hot_logits.ndim != 2 or tuple(hot_logits.shape) != (
            states.shape[0],
            support_size,
        ):
            raise ValueError("project_hot_logits returned the wrong beam/support shape")
        hot_logits_f32 = hot_logits.float()
        log_q = hot_logits_f32 - torch.logsumexp(
            hot_logits_f32, dim=-1, keepdim=True
        )
        expansion = cumulative[:, None] + log_q
        width = min(beam_width, int(expansion.numel()))
        _, flat = torch.topk(
            expansion.flatten(), k=width, largest=True, sorted=True
        )
        # A zero-dimensional CUDA index invokes Tensor.item() in eager
        # indexing. Keep the protected-row lookup on device and graphable.
        greedy_logits = log_q.index_select(0, greedy_row.reshape(1)).squeeze(0)
        greedy_support = torch.argmax(greedy_logits, dim=-1).to(torch.int64)
        greedy_flat = greedy_row * support_size + greedy_support
        contains_greedy = flat.eq(greedy_flat).any()
        flat = torch.cat(
            (
                flat[:-1],
                torch.where(contains_greedy, flat[-1], greedy_flat)[None],
            )
        )
        cumulative = expansion.flatten().index_select(0, flat)
        cumulative, order = torch.sort(cumulative, descending=True)
        flat = flat.index_select(0, order)
        parents = torch.div(flat, support_size, rounding_mode="floor")
        support_indices = torch.remainder(flat, support_size).to(torch.int64)
        next_tokens = hot_token_ids[support_indices]
        paths = torch.cat(
            (paths.index_select(0, parents), next_tokens[:, None]), dim=1
        )
        states = next_states.index_select(0, parents)
        input_tokens = next_tokens
        greedy_row = torch.argmax(flat.eq(greedy_flat).to(torch.int64))

    if paths.shape[0] < 2:
        raise RuntimeError("recurrent beam did not produce two final directions")
    rows = torch.arange(paths.shape[0], dtype=torch.int64, device=paths.device)
    alternate_row = torch.argmax(
        cumulative.masked_fill(rows.eq(greedy_row), -torch.inf)
    )
    selected = torch.stack((greedy_row, alternate_row))
    return (
        paths.index_select(0, selected).contiguous(),
        cumulative.index_select(0, selected).contiguous(),
    )


def pack_top2_prefix_tree(
    *,
    anchor_token: torch.Tensor,
    selected_lines: torch.Tensor,
    selected_log_q: torch.Tensor,
    node_budget: int = TREE_NODE_BUDGET,
) -> Top2BeamTree:
    """Pack two lines into a fixed-width prefix tree without a CPU decision.

    The protected local-greedy line is never truncated.  The alternate receives
    every remaining row after its common prefix.  A fixed 14-node H8 tree
    therefore carries five novel alternate nodes; that is the observed
    cost/yield knee across the registered 512-anchor, four-offset result.
    """

    if anchor_token.shape != (1,) or anchor_token.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("tree packing requires one integer anchor token")
    if selected_lines.ndim != 2 or selected_lines.shape[0] != 2:
        raise ValueError("selected_lines must have shape [2, horizon]")
    if selected_lines.dtype not in (torch.int32, torch.int64):
        raise ValueError("selected_lines must contain integer token ids")
    if selected_log_q.shape != (2,) or not selected_log_q.is_floating_point():
        raise ValueError("selected_log_q must contain two floating-point scores")
    if (
        selected_lines.device != anchor_token.device
        or selected_log_q.device != anchor_token.device
    ):
        raise ValueError("anchor, lines, and scores must share one device")
    horizon = int(selected_lines.shape[1])
    if not 0 <= horizon <= TREE_HORIZON:
        raise ValueError(f"tree horizon must be in [0, {TREE_HORIZON}]")
    if node_budget < 1 + horizon:
        raise ValueError("node budget must preserve the complete primary line")

    device = anchor_token.device
    if horizon:
        equal_prefix = selected_lines[0].eq(selected_lines[1]).to(torch.int64)
        shared = torch.cumprod(equal_prefix, dim=0).sum()
    else:
        shared = torch.zeros((), dtype=torch.int64, device=device)

    alternate_capacity = node_budget - (1 + horizon)
    alternate_slots = torch.arange(
        alternate_capacity, dtype=torch.int64, device=device
    )
    alternate_depth_zero = shared + alternate_slots
    alternate_active = alternate_depth_zero < horizon
    if alternate_capacity:
        if horizon:
            gather_at = alternate_depth_zero.clamp(max=horizon - 1)
            alternate_tokens = selected_lines[1].index_select(0, gather_at)
        else:
            alternate_tokens = anchor_token.expand(alternate_capacity)
        alternate_tokens = torch.where(
            alternate_active,
            alternate_tokens,
            anchor_token.expand(alternate_capacity),
        )
    else:
        alternate_tokens = torch.empty((0,), dtype=torch.int64, device=device)

    node_tokens = torch.cat(
        (
            anchor_token.to(torch.int64),
            selected_lines[0].to(torch.int64),
            alternate_tokens.to(torch.int64),
        )
    )
    rows = torch.arange(node_budget, dtype=torch.int64, device=device)
    columns = rows[None, :]
    ancestor_mask = torch.eye(node_budget, dtype=torch.bool, device=device)

    if horizon:
        primary_rows = torch.arange(1, horizon + 1, device=device)[:, None]
        ancestor_mask[1 : horizon + 1] = columns <= primary_rows

    if alternate_capacity:
        alternate_rows = 1 + horizon + alternate_slots
        shared_primary = (columns == 0) | (
            (columns >= 1) & (columns <= shared)
        )
        alternate_chain = (columns >= 1 + horizon) & (
            columns <= alternate_rows[:, None]
        )
        reachable_mask = shared_primary | alternate_chain
        isolated_mask = torch.nn.functional.one_hot(
            alternate_rows, num_classes=node_budget
        ).to(torch.bool)
        ancestor_mask[alternate_rows] = torch.where(
            alternate_active[:, None], reachable_mask, isolated_mask
        )

    actual_alternate = torch.minimum(
        torch.tensor(alternate_capacity, dtype=torch.int64, device=device),
        torch.tensor(horizon, dtype=torch.int64, device=device) - shared,
    )
    actual_node_count = 1 + horizon + actual_alternate
    result = Top2BeamTree(
        node_tokens=node_tokens,
        ancestor_mask=ancestor_mask,
        selected_lines=selected_lines.to(torch.int64),
        selected_log_q=selected_log_q.float(),
        shared_prefix_length=shared,
        actual_node_count=actual_node_count,
        horizon=horizon,
        node_budget=node_budget,
    )
    result.validate()
    return result


def reference_greedy_tree_acceptance(
    tree: Top2BeamTree, target_predictions: torch.Tensor
) -> dict[str, object]:
    """Small CPU/reference equivalent of SGLang's greedy tree traversal."""

    tree.validate()
    if target_predictions.shape != (
        tree.node_budget,
    ) or target_predictions.dtype not in (torch.int32, torch.int64):
        raise ValueError("target_predictions must contain one token per tree node")
    if target_predictions.device != tree.node_tokens.device:
        raise ValueError("target predictions and tree must share one device")

    mask = tree.ancestor_mask.cpu()
    tokens = tree.node_tokens.cpu().tolist()
    predictions = target_predictions.cpu().tolist()
    parents = [-1] * tree.node_budget
    for node in range(1, tree.node_budget):
        ancestors = [index for index in range(node) if bool(mask[node, index])]
        parents[node] = max(ancestors) if ancestors else -1
    children: list[list[int]] = [[] for _ in range(tree.node_budget)]
    for node, parent in enumerate(parents):
        if parent >= 0:
            children[parent].append(node)

    current = 0
    accepted_path = [0]
    outputs: list[int] = []
    while True:
        predicted = int(predictions[current])
        matching = [child for child in children[current] if tokens[child] == predicted]
        if matching:
            current = matching[0]
            accepted_path.append(current)
            outputs.append(predicted)
            continue
        outputs.append(predicted)
        break
    return {
        "accepted_drafts": len(accepted_path) - 1,
        "accept_index": accepted_path,
        "output_tokens": outputs,
        "bonus_token": outputs[-1],
        "selected_hidden_node": current,
    }
