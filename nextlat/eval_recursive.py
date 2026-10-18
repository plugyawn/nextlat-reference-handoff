"""Recursive-state diagnostics for the fixed Qwen3 NextLat path."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F

from .core import NextLatDynamics
from .dmt import recurrent_readout_state


def _valid_head_weight(
    model: torch.nn.Module, valid_vocab_size: int | None
) -> Tuple[torch.Tensor, int]:
    """Return the deployed-token rows of the (possibly padded) LM head."""

    head_weight = model.lm_head.weight.detach()
    resolved = int(head_weight.shape[0]) if valid_vocab_size is None else valid_vocab_size
    if (
        isinstance(resolved, bool)
        or not isinstance(resolved, int)
        or resolved <= 0
        or resolved > int(head_weight.shape[0])
        or resolved > int(model.get_input_embeddings().weight.shape[0])
    ):
        raise ValueError("valid_vocab_size is incompatible with model embeddings/head")
    return head_weight[:resolved], resolved


def _eos_token_set(
    model: torch.nn.Module,
    eos_token_ids: Sequence[int] | int | None,
    *,
    valid_vocab_size: int,
) -> set[int]:
    """Union caller-supplied generation EOS IDs with the model-config fallback."""

    values: List[Any] = []
    for source in (getattr(model.config, "eos_token_id", None), eos_token_ids):
        if source is None:
            continue
        if isinstance(source, (list, tuple, set)):
            values.extend(source)
        else:
            values.append(source)
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value >= valid_vocab_size
        for value in values
    ):
        raise ValueError("EOS token IDs must be valid deployed-token IDs")
    return {int(value) for value in values}


def _anchor_offset(problem_hash: str, available: int, ordinal: int) -> int:
    digest = hashlib.sha256(
        f"anchor:{ordinal}:{problem_hash}".encode("utf-8")
    ).hexdigest()
    return int(digest[:16], 16) % available


def _random_response_anchors(
    batch: Dict[str, Any],
    max_depth: int,
    anchors_per_example: int,
) -> List[Tuple[int, int]]:
    attention = batch["attention_mask"].bool()
    assistant = batch["assistant_loss_mask"].bool()
    hashes = batch.get("problem_hash") or [str(i) for i in range(attention.shape[0])]
    anchors: List[Tuple[int, int]] = []
    for row in range(attention.shape[0]):
        valid_length = int(attention[row].sum().item())
        candidates = [
            position
            for position in range(max(0, valid_length - max_depth))
            if bool(assistant[row, position + 1])
        ]
        if not candidates:
            continue
        chosen = set()
        for ordinal in range(min(anchors_per_example, len(candidates))):
            offset = _anchor_offset(str(hashes[row]), len(candidates), ordinal)
            while offset in chosen:
                offset = (offset + 1) % len(candidates)
            chosen.add(offset)
            anchors.append((row, candidates[offset]))
    return anchors


def _distribution_metrics(
    teacher_states: torch.Tensor,
    student_states: torch.Tensor,
    detached_head_weight: torch.Tensor,
) -> Tuple[float, float]:
    teacher_logits = F.linear(teacher_states, detached_head_weight)
    student_logits = F.linear(student_states, detached_head_weight)
    teacher_log_probs = F.log_softmax(teacher_logits.float(), dim=-1)
    student_log_probs = F.log_softmax(student_logits.float(), dim=-1)
    kl = F.kl_div(
        student_log_probs,
        teacher_log_probs,
        log_target=True,
        reduction="none",
    ).sum(dim=-1)
    agreement = teacher_logits.argmax(dim=-1).eq(student_logits.argmax(dim=-1))
    return float(kl.mean().item()), float(agreement.float().mean().item())


def _state_metrics(
    teacher_states: torch.Tensor, student_states: torch.Tensor
) -> Tuple[float, float, float]:
    smooth = F.smooth_l1_loss(student_states, teacher_states, reduction="mean")
    error = (student_states.float() - teacher_states.float()).pow(2).sum(dim=-1).sqrt()
    scale = teacher_states.float().pow(2).sum(dim=-1).sqrt().clamp_min(1e-12)
    relative_l2 = (error / scale).mean()
    cosine = F.cosine_similarity(
        student_states.float(), teacher_states.float(), dim=-1
    ).mean()
    return float(smooth.item()), float(relative_l2.item()), float(cosine.item())


@torch.inference_mode()
def teacher_forced_recursive_metrics(
    *,
    model: torch.nn.Module,
    dynamics: NextLatDynamics,
    batch: Dict[str, Any],
    depths: Sequence[int],
    anchors_per_example: int = 2,
    valid_vocab_size: int | None = None,
    readout_adapter: torch.nn.Module | None = None,
    transition_embedding: torch.nn.Module | None = None,
) -> Dict[str, Any]:
    """Compare recurrent states with exact states along held-out response tokens."""

    depths = sorted(set(int(depth) for depth in depths))
    if not depths or depths[0] < 1:
        raise ValueError("depths must be positive")
    max_depth = depths[-1]
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    embeddings = model.get_input_embeddings()(input_ids)
    exact = model.model(
        inputs_embeds=embeddings,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    ).last_hidden_state
    anchors = _random_response_anchors(batch, max_depth, anchors_per_example)
    if not anchors:
        raise RuntimeError("no held-out response has room for the requested depth")

    rows = torch.tensor([row for row, _ in anchors], device=input_ids.device)
    starts = torch.tensor([start for _, start in anchors], device=input_ids.device)
    recurrent = exact[rows, starts]
    head_weight, _ = _valid_head_weight(model, valid_vocab_size)
    metrics: Dict[str, Any] = {
        "mode": "teacher_forced_random_response_anchor",
        "anchors": len(anchors),
        "depths": {},
    }
    for step in range(1, max_depth + 1):
        tokens = input_ids[rows, starts + step]
        token_embeddings = (
            model.get_input_embeddings()
            if transition_embedding is None
            else transition_embedding
        )(tokens)
        recurrent = dynamics(recurrent, token_embeddings)
        if step not in depths:
            continue
        teacher = exact[rows, starts + step]
        smooth, relative_l2, cosine = _state_metrics(teacher, recurrent)
        observed_recurrent = recurrent_readout_state(
            recurrent, readout_adapter, recurrent_depth=step
        )
        kl, top_agreement = _distribution_metrics(
            teacher, observed_recurrent, head_weight
        )
        metrics["depths"][str(step)] = {
            "smooth_l1": smooth,
            "relative_l2": relative_l2,
            "cosine": cosine,
            "output_kl": kl,
            "top_token_agreement": top_agreement,
        }
    return metrics


@torch.inference_mode()
def proposal_recursive_metrics(
    *,
    model: torch.nn.Module,
    dynamics: NextLatDynamics,
    batch: Dict[str, Any],
    depths: Sequence[int],
    proposal_length: int,
    compute_depth_metrics: bool = True,
    valid_vocab_size: int | None = None,
    eos_token_ids: Sequence[int] | int | None = None,
    readout_adapter: torch.nn.Module | None = None,
    transition_embedding: torch.nn.Module | None = None,
) -> Dict[str, Any]:
    """Draft greedily in latent space, then verify each proposal with one exact pass."""

    depths = sorted(
        set(int(depth) for depth in depths if 0 < int(depth) <= proposal_length)
    )
    if compute_depth_metrics and not depths:
        raise ValueError("no requested depth fits proposal_length")
    if not compute_depth_metrics:
        depths = []
    input_ids = batch["input_ids"]
    attention = batch["attention_mask"].bool()
    assistant = batch["assistant_loss_mask"].bool()
    hashes = batch.get("problem_hash") or [str(i) for i in range(input_ids.shape[0])]
    head_weight, resolved_vocab_size = _valid_head_weight(model, valid_vocab_size)
    by_depth: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    accepted_prefixes: List[int] = []
    draft_lengths: List[int] = []
    eos_steps: List[int | None] = []
    anchor_positions: List[int] = []
    eos_ids = _eos_token_set(
        model, eos_token_ids, valid_vocab_size=resolved_vocab_size
    )

    for row in range(input_ids.shape[0]):
        valid_length = int(attention[row].sum().item())
        response_positions = torch.nonzero(
            assistant[row, :valid_length], as_tuple=False
        )
        if response_positions.numel() == 0:
            continue
        first_response = int(response_positions[0].item())
        last_anchor = valid_length - 2
        if first_response - 1 > last_anchor:
            continue
        span = last_anchor - (first_response - 1) + 1
        start = first_response - 1 + _anchor_offset(str(hashes[row]), span, 91)
        anchor_positions.append(start)
        prefix = input_ids[row : row + 1, : start + 1]
        prefix_mask = torch.ones_like(prefix)
        prefix_embeddings = model.get_input_embeddings()(prefix)
        exact_prefix = model.model(
            inputs_embeds=prefix_embeddings,
            attention_mask=prefix_mask,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        state = exact_prefix[:, -1]
        drafted: List[torch.Tensor] = []
        recurrent_states: List[torch.Tensor] = []
        for recurrent_depth in range(proposal_length):
            observed_state = recurrent_readout_state(
                state, readout_adapter, recurrent_depth=recurrent_depth
            )
            token = F.linear(observed_state, head_weight).argmax(dim=-1)
            drafted.append(token)
            token_embeddings = (
                model.get_input_embeddings()
                if transition_embedding is None
                else transition_embedding
            )(token)
            state = dynamics(state, token_embeddings)
            recurrent_states.append(state)
            if int(token.item()) in eos_ids:
                break

        draft_ids = torch.stack(drafted, dim=1)
        draft_length = int(draft_ids.shape[1])
        draft_lengths.append(draft_length)
        eos_steps.append(
            draft_length if int(draft_ids[0, -1].item()) in eos_ids else None
        )
        verification_ids = torch.cat((prefix, draft_ids), dim=1)
        verification_embeddings = model.get_input_embeddings()(verification_ids)
        exact_verification = model.model(
            inputs_embeds=verification_embeddings,
            attention_mask=torch.ones_like(verification_ids),
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        exact_next_tokens = F.linear(
            exact_verification[:, start : start + draft_length], head_weight
        ).argmax(dim=-1)
        accepted = 0
        for proposed, target in zip(draft_ids[0], exact_next_tokens[0]):
            if int(proposed.item()) != int(target.item()):
                break
            accepted += 1
        accepted_prefixes.append(accepted)

        for depth in depths:
            if depth > draft_length:
                continue
            teacher = exact_verification[:, start + depth]
            student = recurrent_states[depth - 1]
            smooth, relative_l2, cosine = _state_metrics(teacher, student)
            observed_student = recurrent_readout_state(
                student, readout_adapter, recurrent_depth=depth
            )
            kl, top_agreement = _distribution_metrics(
                teacher, observed_student, head_weight
            )
            by_depth[depth]["smooth_l1"].append(smooth)
            by_depth[depth]["relative_l2"].append(relative_l2)
            by_depth[depth]["cosine"].append(cosine)
            by_depth[depth]["output_kl"].append(kl)
            by_depth[depth]["top_token_agreement"].append(top_agreement)

    if not accepted_prefixes:
        raise RuntimeError("proposal evaluation found no usable response anchors")
    return {
        "mode": "latent_greedy_proposal_exact_verification",
        "examples": len(accepted_prefixes),
        "proposal_length": proposal_length,
        "draft_lengths": draft_lengths,
        "eos_steps": eos_steps,
        "anchor_positions": anchor_positions,
        "accepted_prefix": {
            "mean": sum(accepted_prefixes) / len(accepted_prefixes),
            "at_least_16_rate": sum(value >= 16 for value in accepted_prefixes)
            / len(accepted_prefixes),
            "values": accepted_prefixes,
        },
        "depths": {
            str(depth): {
                metric: sum(values) / len(values) for metric, values in metrics.items()
            }
            for depth, metrics in by_depth.items()
        },
    }


@torch.inference_mode()
def exact_sampled_random_continuation_metrics(
    *,
    model: torch.nn.Module,
    dynamics: NextLatDynamics,
    batch: Dict[str, Any],
    depths: Sequence[int],
    continuation_length: int,
    temperature: float = 1.0,
    return_trajectories: bool = False,
    valid_vocab_size: int | None = None,
    eos_token_ids: Sequence[int] | int | None = None,
    readout_adapter: torch.nn.Module | None = None,
    transition_embedding: torch.nn.Module | None = None,
) -> Dict[str, Any]:
    """Roll recurrent states along continuations sampled from the exact target.

    ``return_trajectories`` is intended for registered paired evaluations.  The
    exact token path sampled from the immutable parent can then be replayed for
    every descendant instead of silently sampling a different path from each
    changed target model.
    """

    if continuation_length <= 0 or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("continuation_length and temperature must be positive")
    depths = sorted(
        set(int(depth) for depth in depths if 0 < int(depth) <= continuation_length)
    )
    if not depths:
        raise ValueError("no requested depth fits continuation_length")
    input_ids = batch["input_ids"]
    attention = batch["attention_mask"].bool()
    assistant = batch["assistant_loss_mask"].bool()
    hashes = batch.get("problem_hash") or [str(i) for i in range(input_ids.shape[0])]
    head_weight, resolved_vocab_size = _valid_head_weight(model, valid_vocab_size)
    eos_ids = _eos_token_set(
        model, eos_token_ids, valid_vocab_size=resolved_vocab_size
    )
    by_depth: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    sampled_lengths: List[int] = []
    trajectories: List[Dict[str, Any]] = []

    for row in range(input_ids.shape[0]):
        valid_length = int(attention[row].sum().item())
        response_positions = torch.nonzero(
            assistant[row, :valid_length], as_tuple=False
        )
        if response_positions.numel() == 0:
            continue
        first_response = int(response_positions[0].item())
        last_anchor = valid_length - 2
        if first_response - 1 > last_anchor:
            continue
        span = last_anchor - (first_response - 1) + 1
        start = first_response - 1 + _anchor_offset(str(hashes[row]), span, 173)
        prefix = input_ids[row : row + 1, : start + 1]
        prefix_output = model.model(
            input_ids=prefix,
            attention_mask=torch.ones_like(prefix),
            use_cache=True,
            return_dict=True,
        )
        exact_state = prefix_output.last_hidden_state[:, -1]
        recurrent_state = exact_state
        cache = prefix_output.past_key_values
        generator = torch.Generator(device=input_ids.device)
        seed = int(
            hashlib.sha256(f"sample:{hashes[row]}".encode("utf-8")).hexdigest()[:16],
            16,
        ) % (2**63 - 1)
        generator.manual_seed(seed)
        actual_length = 0
        sampled_token_ids: List[int] = []
        eos_step = None
        for step in range(1, continuation_length + 1):
            logits = F.linear(exact_state, head_weight).float() / temperature
            token = torch.multinomial(
                F.softmax(logits, dim=-1), num_samples=1, generator=generator
            )
            sampled_token_ids.append(int(token.item()))
            token_embeddings = (
                model.get_input_embeddings()
                if transition_embedding is None
                else transition_embedding
            )(token)
            recurrent_state = dynamics(recurrent_state, token_embeddings.squeeze(1))
            full_mask = torch.ones(
                (1, prefix.shape[1] + step),
                device=input_ids.device,
                dtype=torch.long,
            )
            exact_output = model.model(
                input_ids=token,
                attention_mask=full_mask,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            cache = exact_output.past_key_values
            exact_state = exact_output.last_hidden_state[:, -1]
            actual_length = step
            if step in depths:
                smooth, relative_l2, cosine = _state_metrics(
                    exact_state, recurrent_state
                )
                observed_recurrent = recurrent_readout_state(
                    recurrent_state, readout_adapter, recurrent_depth=step
                )
                kl, top_agreement = _distribution_metrics(
                    exact_state, observed_recurrent, head_weight
                )
                by_depth[step]["smooth_l1"].append(smooth)
                by_depth[step]["relative_l2"].append(relative_l2)
                by_depth[step]["cosine"].append(cosine)
                by_depth[step]["output_kl"].append(kl)
                by_depth[step]["top_token_agreement"].append(top_agreement)
            if int(token.item()) in eos_ids:
                eos_step = step
                break
        sampled_lengths.append(actual_length)
        trajectories.append(
            {
                "schema_version": 1,
                "problem_hash": str(hashes[row]),
                "anchor_position": start,
                "prefix_length": start + 1,
                "sampling_seed": seed,
                "temperature": float(temperature),
                "requested_length": continuation_length,
                "token_ids": sampled_token_ids,
                "eos_step": eos_step,
                "truncated_at_requested_length": eos_step is None,
                "valid_vocab_size": resolved_vocab_size,
                "eos_token_ids": sorted(eos_ids),
            }
        )

    if not sampled_lengths:
        raise RuntimeError("random-continuation evaluation found no usable anchors")
    result = {
        "mode": "exact_target_sampled_random_continuation",
        "examples": len(sampled_lengths),
        "temperature": temperature,
        "requested_length": continuation_length,
        "sampled_lengths": sampled_lengths,
        "depths": {
            str(depth): {
                metric: sum(values) / len(values) for metric, values in metrics.items()
            }
            for depth, metrics in by_depth.items()
            if metrics
        },
    }
    if return_trajectories:
        result["trajectories"] = trajectories
    return result


@torch.inference_mode()
def frozen_random_continuation_metrics(
    *,
    model: torch.nn.Module,
    dynamics: NextLatDynamics,
    batch: Dict[str, Any],
    depths: Sequence[int],
    trajectory: Mapping[str, Any],
    valid_vocab_size: int | None = None,
    eos_token_ids: Sequence[int] | int | None = None,
    readout_adapter: torch.nn.Module | None = None,
    transition_embedding: torch.nn.Module | None = None,
) -> Dict[str, Any]:
    """Replay one parent-frozen exact-token trajectory on another checkpoint."""

    if batch["input_ids"].shape[0] != 1:
        raise ValueError("frozen random replay requires micro-batch size one")
    requested_length = trajectory.get("requested_length")
    if (
        isinstance(requested_length, bool)
        or not isinstance(requested_length, int)
        or requested_length <= 0
    ):
        raise ValueError("frozen trajectory requested_length must be positive")
    depths = sorted(
        set(int(depth) for depth in depths if 0 < int(depth) <= requested_length)
    )
    if not depths:
        raise ValueError("no requested depth fits frozen trajectory")

    input_ids = batch["input_ids"]
    attention = batch["attention_mask"].bool()
    assistant = batch["assistant_loss_mask"].bool()
    hashes = batch.get("problem_hash") or ["0"]
    problem_hash = str(hashes[0])
    head_weight, resolved_vocab_size = _valid_head_weight(model, valid_vocab_size)
    eos_ids = _eos_token_set(
        model, eos_token_ids, valid_vocab_size=resolved_vocab_size
    )
    if trajectory.get("schema_version") != 1:
        raise RuntimeError("unsupported frozen-trajectory schema")
    if trajectory.get("problem_hash") != problem_hash:
        raise RuntimeError("frozen trajectory names a different problem")
    expected_seed = int(
        hashlib.sha256(f"sample:{problem_hash}".encode("utf-8")).hexdigest()[:16],
        16,
    ) % (2**63 - 1)
    if trajectory.get("sampling_seed") != expected_seed:
        raise RuntimeError("frozen trajectory sampling seed differs")
    temperature = trajectory.get("temperature")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(float(temperature))
        or float(temperature) <= 0.0
    ):
        raise RuntimeError("frozen trajectory temperature differs")

    valid_length = int(attention[0].sum().item())
    response_positions = torch.nonzero(assistant[0, :valid_length], as_tuple=False)
    if response_positions.numel() == 0:
        raise RuntimeError("frozen random replay found no response tokens")
    first_response = int(response_positions[0].item())
    last_anchor = valid_length - 2
    if first_response - 1 > last_anchor:
        raise RuntimeError("frozen random replay found no usable anchor")
    span = last_anchor - (first_response - 1) + 1
    expected_start = first_response - 1 + _anchor_offset(problem_hash, span, 173)
    if trajectory.get("anchor_position") != expected_start:
        raise RuntimeError("frozen trajectory anchor differs from the fixed holdout")
    if trajectory.get("prefix_length") != expected_start + 1:
        raise RuntimeError("frozen trajectory prefix length differs")

    token_ids = trajectory.get("token_ids")
    if (
        not isinstance(token_ids, list)
        or not token_ids
        or len(token_ids) > requested_length
        or any(
            isinstance(token, bool)
            or not isinstance(token, int)
            or token < 0
            or token >= resolved_vocab_size
            for token in token_ids
        )
    ):
        raise RuntimeError("frozen trajectory token roster is invalid")
    if trajectory.get("valid_vocab_size", resolved_vocab_size) != resolved_vocab_size:
        raise RuntimeError("frozen trajectory valid-vocabulary receipt differs")
    if trajectory.get("eos_token_ids", sorted(eos_ids)) != sorted(eos_ids):
        raise RuntimeError("frozen trajectory EOS-token roster differs")
    observed_eos = next(
        (index for index, token in enumerate(token_ids, start=1) if token in eos_ids),
        None,
    )
    if trajectory.get("eos_step") != observed_eos:
        raise RuntimeError("frozen trajectory EOS receipt differs")
    if (trajectory.get("truncated_at_requested_length") is True) != (
        observed_eos is None
    ):
        raise RuntimeError("frozen trajectory truncation receipt differs")
    if observed_eos is None and len(token_ids) != requested_length:
        raise RuntimeError("non-EOS frozen trajectory is shorter than requested")
    if observed_eos is not None and observed_eos != len(token_ids):
        raise RuntimeError("frozen trajectory contains tokens after EOS")

    prefix = input_ids[0:1, : expected_start + 1]
    prefix_output = model.model(
        input_ids=prefix,
        attention_mask=torch.ones_like(prefix),
        use_cache=True,
        return_dict=True,
    )
    exact_state = prefix_output.last_hidden_state[:, -1]
    recurrent_state = exact_state
    cache = prefix_output.past_key_values
    by_depth: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for step, token_id in enumerate(token_ids, start=1):
        token = torch.tensor(
            [[token_id]], device=input_ids.device, dtype=input_ids.dtype
        )
        token_embeddings = (
            model.get_input_embeddings()
            if transition_embedding is None
            else transition_embedding
        )(token)
        recurrent_state = dynamics(recurrent_state, token_embeddings.squeeze(1))
        exact_output = model.model(
            input_ids=token,
            attention_mask=torch.ones(
                (1, prefix.shape[1] + step),
                device=input_ids.device,
                dtype=torch.long,
            ),
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        cache = exact_output.past_key_values
        exact_state = exact_output.last_hidden_state[:, -1]
        if step not in depths:
            continue
        smooth, relative_l2, cosine = _state_metrics(exact_state, recurrent_state)
        observed_recurrent = recurrent_readout_state(
            recurrent_state, readout_adapter, recurrent_depth=step
        )
        kl, top_agreement = _distribution_metrics(
            exact_state, observed_recurrent, head_weight
        )
        by_depth[step]["smooth_l1"].append(smooth)
        by_depth[step]["relative_l2"].append(relative_l2)
        by_depth[step]["cosine"].append(cosine)
        by_depth[step]["output_kl"].append(kl)
        by_depth[step]["top_token_agreement"].append(top_agreement)

    return {
        "mode": "frozen_parent_exact_target_random_continuation",
        "examples": 1,
        "requested_length": requested_length,
        "sampled_lengths": [len(token_ids)],
        "eos_step": observed_eos,
        "depths": {
            str(depth): {
                metric: sum(values) / len(values) for metric, values in metrics.items()
            }
            for depth, metrics in by_depth.items()
            if metrics
        },
    }


@torch.inference_mode()
def benchmark_edge_speed(
    *,
    model: torch.nn.Module,
    dynamics: NextLatDynamics,
    prefix_ids: torch.Tensor,
    edge_length: int,
    repeats: int = 5,
) -> Dict[str, Any]:
    """Time exact KV decode versus recurrence, including the vocabulary head."""

    if prefix_ids.shape[0] != 1 or edge_length <= 0 or repeats <= 0:
        raise ValueError("benchmark requires one prefix and positive length/repeats")
    head_weight = model.lm_head.weight.detach()

    def common_anchor() -> Tuple[torch.Tensor, Any]:
        output = model.model(
            input_ids=prefix_ids,
            attention_mask=torch.ones_like(prefix_ids),
            use_cache=True,
            return_dict=True,
        )
        return output.last_hidden_state[:, -1], output.past_key_values

    def exact_edge(state: torch.Tensor, cache: Any) -> None:
        for step in range(1, edge_length + 1):
            token = F.linear(state, head_weight).argmax(dim=-1, keepdim=True)
            output = model.model(
                input_ids=token,
                attention_mask=torch.ones(
                    (1, prefix_ids.shape[1] + step),
                    device=prefix_ids.device,
                    dtype=torch.long,
                ),
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            state = output.last_hidden_state[:, -1]
            cache = output.past_key_values

    def latent_edge(state: torch.Tensor) -> None:
        for _ in range(edge_length):
            token = F.linear(state, head_weight).argmax(dim=-1)
            state = dynamics(state, model.get_input_embeddings()(token))

    exact_ms: List[float] = []
    latent_ms: List[float] = []
    warm_state, warm_cache = common_anchor()
    exact_edge(warm_state, warm_cache)
    warm_state, _ = common_anchor()
    latent_edge(warm_state)
    torch.cuda.synchronize(prefix_ids.device)
    for _ in range(repeats):
        state, cache = common_anchor()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        exact_edge(state, cache)
        end.record()
        torch.cuda.synchronize(prefix_ids.device)
        exact_ms.append(float(start.elapsed_time(end)))

        state, _ = common_anchor()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        latent_edge(state)
        end.record()
        torch.cuda.synchronize(prefix_ids.device)
        latent_ms.append(float(start.elapsed_time(end)))

    exact_mean = sum(exact_ms) / len(exact_ms)
    latent_mean = sum(latent_ms) / len(latent_ms)
    return {
        "mode": "target_anchored_edge_speed_including_vocabulary_head",
        "edge_length": edge_length,
        "repeats": repeats,
        "common_exact_anchor_excluded": True,
        "exact_target_forwards_per_edge": edge_length,
        "latent_target_forwards_per_edge": 0,
        "exact_edge_ms": exact_ms,
        "latent_edge_ms": latent_ms,
        "exact_edge_mean_ms": exact_mean,
        "latent_edge_mean_ms": latent_mean,
        "speedup": exact_mean / max(latent_mean, 1e-9),
    }


@torch.inference_mode()
def benchmark_end_to_end_speculative_edge(
    *,
    model: torch.nn.Module,
    dynamics: NextLatDynamics,
    prefix_ids: torch.Tensor,
    edge_length: int,
    repeats: int = 5,
) -> Dict[str, Any]:
    """Time complete exact and recurrent-speculative edges.

    Both paths include the exact prefix/anchor forward and every vocabulary-head
    projection.  The recurrent path additionally includes its single batched
    exact-target verification and verification head, so this record cannot be
    mistaken for recurrence-only serving throughput.
    """

    if prefix_ids.ndim != 2 or prefix_ids.shape[0] != 1:
        raise ValueError("end-to-end benchmark requires one rank-two prefix")
    if prefix_ids.shape[1] < 1 or edge_length <= 0 or repeats <= 0:
        raise ValueError("benchmark prefix, edge_length, and repeats must be positive")
    if prefix_ids.device.type != "cuda":
        raise ValueError("end-to-end benchmark requires CUDA timing")
    head_weight = model.lm_head.weight.detach()

    def common_anchor() -> Tuple[torch.Tensor, Any]:
        output = model.model(
            input_ids=prefix_ids,
            attention_mask=torch.ones_like(prefix_ids),
            use_cache=True,
            return_dict=True,
        )
        return output.last_hidden_state[:, -1], output.past_key_values

    def exact_path() -> torch.Tensor:
        state, cache = common_anchor()
        generated: List[torch.Tensor] = []
        for step in range(1, edge_length + 1):
            token = F.linear(state, head_weight).argmax(dim=-1, keepdim=True)
            generated.append(token)
            output = model.model(
                input_ids=token,
                attention_mask=torch.ones(
                    (1, prefix_ids.shape[1] + step),
                    device=prefix_ids.device,
                    dtype=torch.long,
                ),
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            state = output.last_hidden_state[:, -1]
            cache = output.past_key_values
        # Materialize the bonus-token head that is also produced by the
        # speculative verification pass.  This keeps vocabulary-head work
        # matched while both paths leave an exact state at the edge boundary.
        bonus_token = F.linear(state, head_weight).argmax(dim=-1, keepdim=True)
        return torch.cat((*generated, bonus_token), dim=1)

    def speculative_path() -> Tuple[torch.Tensor, torch.Tensor]:
        anchor_state, cache = common_anchor()
        recurrent = anchor_state
        drafted: List[torch.Tensor] = []
        for _ in range(edge_length):
            token = F.linear(recurrent, head_weight).argmax(dim=-1)
            drafted.append(token)
            recurrent = dynamics(recurrent, model.get_input_embeddings()(token))
        draft_ids = torch.stack(drafted, dim=1)
        verified = model.model(
            input_ids=draft_ids,
            attention_mask=torch.ones(
                (1, prefix_ids.shape[1] + edge_length),
                device=prefix_ids.device,
                dtype=torch.long,
            ),
            past_key_values=cache,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        verification_tokens = F.linear(verified, head_weight).argmax(dim=-1)
        exact_tokens = torch.cat((draft_ids[:, :1], verification_tokens[:, :-1]), dim=1)
        prefix_matches = torch.cumprod(
            draft_ids.eq(exact_tokens).to(dtype=torch.int64), dim=1
        )
        accepted = prefix_matches.sum(dim=1)
        return draft_ids, accepted

    def timed(function: Any) -> Tuple[float, Any]:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        value = function()
        end.record()
        torch.cuda.synchronize(prefix_ids.device)
        return float(start.elapsed_time(end)), value

    exact_path()
    speculative_path()
    torch.cuda.synchronize(prefix_ids.device)
    exact_ms: List[float] = []
    speculative_ms: List[float] = []
    accepted_prefixes: List[int] = []
    for repeat in range(repeats):
        ordered = (
            (("exact", exact_path), ("speculative", speculative_path))
            if repeat % 2 == 0
            else (("speculative", speculative_path), ("exact", exact_path))
        )
        for label, function in ordered:
            elapsed, value = timed(function)
            if label == "exact":
                exact_ms.append(elapsed)
            else:
                speculative_ms.append(elapsed)
                accepted_prefixes.append(int(value[1].item()))

    exact_mean = sum(exact_ms) / len(exact_ms)
    speculative_mean = sum(speculative_ms) / len(speculative_ms)
    exact_ordered = sorted(exact_ms)
    speculative_ordered = sorted(speculative_ms)
    median_index = math.ceil(0.5 * repeats) - 1
    p95_index = math.ceil(0.95 * repeats) - 1
    return {
        "mode": "full_end_to_end_recurrent_speculative_edge",
        "edge_length": edge_length,
        "repeats": repeats,
        "common_exact_anchor_included": True,
        "vocabulary_head_included": True,
        "exact_target_verification_included": True,
        "exact_target_forwards_per_exact_edge_including_anchor": edge_length + 1,
        "exact_target_forwards_per_speculative_edge_including_anchor": 2,
        "exact_vocabulary_positions_including_bonus": edge_length + 1,
        "draft_vocabulary_positions": edge_length,
        "verification_vocabulary_positions_including_bonus": edge_length,
        "exact_end_to_end_ms": exact_ms,
        "speculative_end_to_end_ms": speculative_ms,
        "exact_end_to_end_mean_ms": exact_mean,
        "exact_end_to_end_median_ms": exact_ordered[median_index],
        "exact_end_to_end_p95_ms": exact_ordered[p95_index],
        "speculative_end_to_end_mean_ms": speculative_mean,
        "speculative_end_to_end_median_ms": speculative_ordered[median_index],
        "speculative_end_to_end_p95_ms": speculative_ordered[p95_index],
        "attempted_edge_speedup": exact_mean / max(speculative_mean, 1e-9),
        "speculative_accepted_prefixes": accepted_prefixes,
    }
