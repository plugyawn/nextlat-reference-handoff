"""Cumulative recurrent-draft confidence and bounded offline replay.

This module deliberately stops at diagnostics.  It does not choose proposal
lengths online and does not define a value function.  Scores come from the
production recurrent proposal logits, while acceptance labels come from the
exact target verifier.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import torch


CONFIDENCE_MAX_DRAFT_TOKENS = 16
CANDIDATE_VERIFY_PREFIXES = (8, 12, 16)
COMMON_TARGET_ROW_BUDGET = math.lcm(
    *(prefix + 1 for prefix in CANDIDATE_VERIFY_PREFIXES)
)


def top1_log_probability(logits: torch.Tensor, *, vocab_size: int) -> torch.Tensor:
    """Return ``log p(argmax)`` in FP32 for each production proposal row."""

    if logits.ndim != 2:
        raise ValueError("draft logits must have shape [batch, vocab]")
    if vocab_size <= 1 or logits.shape[1] < vocab_size:
        raise ValueError("draft logits do not cover the admitted vocabulary")
    admitted = logits[:, :vocab_size].float()
    # Do not add a host synchronization here: this function runs inside every
    # recurrent proposal step.  Non-finite outputs are rejected when the
    # diagnostic trace is copied to the host and validated.
    return torch.amax(admitted, dim=-1) - torch.logsumexp(admitted, dim=-1)


def _finite_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def validate_confidence_event(
    event: dict[str, Any], *, max_draft_tokens: int = CONFIDENCE_MAX_DRAFT_TOKENS
) -> dict[str, Any]:
    """Validate one worker trace event and its acceptance/score algebra."""

    required = {
        "schema_version",
        "event",
        "forward_iter",
        "request_ids",
        "output_start_indices",
        "proposal_lens",
        "draft_tokens",
        "draft_top1_logprob",
        "cumulative_draft_logprob",
        "num_correct_drafts",
        "accept_lens",
    }
    if not isinstance(event, dict) or set(event) != required:
        raise ValueError("confidence event fields differ from schema v1")
    if event["schema_version"] != 1 or event["event"] != "draft_confidence":
        raise ValueError("confidence event schema or kind differs")
    _nonnegative_int(event["forward_iter"], label="forward_iter")
    request_ids = event["request_ids"]
    if (
        not isinstance(request_ids, list)
        or not request_ids
        or any(not isinstance(value, str) or not value for value in request_ids)
        or len(set(request_ids)) != len(request_ids)
    ):
        raise ValueError("confidence request IDs must be unique non-empty strings")
    batch_size = len(request_ids)
    vector_fields = (
        "output_start_indices",
        "proposal_lens",
        "draft_tokens",
        "draft_top1_logprob",
        "cumulative_draft_logprob",
        "num_correct_drafts",
        "accept_lens",
    )
    for name in vector_fields:
        if not isinstance(event[name], list) or len(event[name]) != batch_size:
            raise ValueError(f"confidence {name} differs from batch size")

    for row in range(batch_size):
        _nonnegative_int(
            event["output_start_indices"][row],
            label=f"output_start_indices[{row}]",
        )
        proposal_len = _nonnegative_int(
            event["proposal_lens"][row], label=f"proposal_lens[{row}]"
        )
        if proposal_len > max_draft_tokens:
            raise ValueError("confidence proposal length exceeds fixed M=16")
        correct = _nonnegative_int(
            event["num_correct_drafts"][row],
            label=f"num_correct_drafts[{row}]",
        )
        if correct > proposal_len:
            raise ValueError("accepted drafts exceed the proposed prefix")
        emitted = _nonnegative_int(
            event["accept_lens"][row], label=f"accept_lens[{row}]"
        )
        if emitted != correct + 1:
            raise ValueError("accept_lens must equal accepted drafts plus target bonus")

        tokens = event["draft_tokens"][row]
        logprob = event["draft_top1_logprob"][row]
        cumulative = event["cumulative_draft_logprob"][row]
        if (
            not isinstance(tokens, list)
            or len(tokens) != max_draft_tokens
            or any(isinstance(value, bool) or not isinstance(value, int) for value in tokens)
        ):
            raise ValueError("draft token row differs from fixed M=16")
        if not isinstance(logprob, list) or len(logprob) != max_draft_tokens:
            raise ValueError("draft log-probability row differs from fixed M=16")
        if not isinstance(cumulative, list) or len(cumulative) != max_draft_tokens:
            raise ValueError("cumulative score row differs from fixed M=16")
        running = 0.0
        previous = 0.0
        for depth, (raw_value, cumulative_value) in enumerate(
            zip(logprob, cumulative), 1
        ):
            raw = _finite_float(raw_value, label=f"logprob[{row},{depth}]")
            score = _finite_float(
                cumulative_value, label=f"cumulative[{row},{depth}]"
            )
            if raw > 1e-6:
                raise ValueError("log p(argmax) cannot be positive")
            running += raw
            if not math.isclose(score, running, rel_tol=1e-6, abs_tol=2e-6):
                raise ValueError("cumulative score differs from the log-probability sum")
            if score > previous + 1e-6:
                raise ValueError("cumulative confidence must be non-increasing in depth")
            previous = score
    return event


def _pairwise_auc(
    positives: list[float], negatives: list[float]
) -> tuple[float, float, int]:
    if not positives or not negatives:
        raise ValueError("AUC requires positive and negative survival examples")
    credit = 0.0
    pairs = 0
    for positive in positives:
        for negative in negatives:
            pairs += 1
            if positive > negative:
                credit += 1.0
            elif positive == negative:
                credit += 0.5
    return credit / pairs, credit, pairs


def depth_conditioned_survival(
    events: Iterable[dict[str, Any]],
    *,
    max_draft_tokens: int = CONFIDENCE_MAX_DRAFT_TOKENS,
) -> dict[str, Any]:
    """Measure whether cumulative confidence orders exact acceptance survival.

    Comparisons are made only within the same recurrent depth.  Pooling raw
    ``s_k`` across depths would create a trivial correlation because cumulative
    log-probability decreases with ``k``.
    """

    rows_by_depth: dict[int, list[tuple[float, bool]]] = {
        depth: [] for depth in range(1, max_draft_tokens + 1)
    }
    event_count = 0
    request_round_count = 0
    for raw_event in events:
        event = validate_confidence_event(
            raw_event, max_draft_tokens=max_draft_tokens
        )
        event_count += 1
        for proposal_len, correct, cumulative in zip(
            event["proposal_lens"],
            event["num_correct_drafts"],
            event["cumulative_draft_logprob"],
        ):
            request_round_count += 1
            for depth in range(1, proposal_len + 1):
                rows_by_depth[depth].append(
                    (float(cumulative[depth - 1]), correct >= depth)
                )
    if event_count == 0 or request_round_count == 0:
        raise ValueError("confidence analysis requires at least one request-round")

    depth_rows: list[dict[str, Any]] = []
    aggregate_credit = 0.0
    aggregate_pairs = 0
    eligible_depths = 0
    for depth, rows in rows_by_depth.items():
        positives = [score for score, survived in rows if survived]
        negatives = [score for score, survived in rows if not survived]
        if positives and negatives:
            auc, credit, pair_count = _pairwise_auc(positives, negatives)
            aggregate_credit += credit
            aggregate_pairs += pair_count
            eligible_depths += 1
        else:
            auc = None
            pair_count = 0
        depth_rows.append(
            {
                "depth": depth,
                "observation_count": len(rows),
                "survived_count": len(positives),
                "failed_before_depth_count": len(negatives),
                "survival_rate": None if not rows else len(positives) / len(rows),
                "survivor_mean_cumulative_logprob": (
                    None if not positives else sum(positives) / len(positives)
                ),
                "failure_mean_cumulative_logprob": (
                    None if not negatives else sum(negatives) / len(negatives)
                ),
                "same_depth_pair_count": pair_count,
                "same_depth_auc": auc,
            }
        )

    aggregate_auc = (
        None if aggregate_pairs == 0 else aggregate_credit / aggregate_pairs
    )
    passed = aggregate_auc is not None and aggregate_auc > 0.5
    return {
        "event_count": event_count,
        "request_round_count": request_round_count,
        "per_depth": depth_rows,
        "ordering": {
            "method": (
                "pair-weighted Mann-Whitney AUC over survivor/failure pairs, "
                "conditioned on identical recurrent depth; ties receive 0.5"
            ),
            "eligible_depth_count": eligible_depths,
            "same_depth_pair_count": aggregate_pairs,
            "aggregate_auc": aggregate_auc,
        },
        "gate": {
            "id": "confidence_orders_acceptance_survival",
            "threshold": (
                "at least one same-depth survivor/failure pair and aggregate "
                "pair-weighted AUC strictly greater than 0.5"
            ),
            "status": "pass" if passed else "fail",
        },
    }


def replay_verify_prefixes(
    events: Iterable[dict[str, Any]],
    *,
    candidates: tuple[int, ...] = CANDIDATE_VERIFY_PREFIXES,
    max_draft_tokens: int = CONFIDENCE_MAX_DRAFT_TOKENS,
    minimum_improvement: float = 0.05,
) -> dict[str, Any]:
    """Replay fixed prefix caps under one common exact-target-row budget.

    Only full-M16 request-rounds are admitted.  For an observed accepted prefix
    ``a``, a candidate cap ``m`` emits ``min(a, m)`` drafts plus one exact target
    bonus and spends ``m + 1`` exact target rows.  This is a local anchored-row
    projection; it does not claim that later recurrent seeds are unchanged.
    """

    if (
        not candidates
        or tuple(sorted(set(candidates))) != candidates
        or candidates[-1] != max_draft_tokens
        or any(value <= 0 or value > max_draft_tokens for value in candidates)
    ):
        raise ValueError("candidate prefixes must be sorted, unique, and end at M=16")
    if not 0.0 < minimum_improvement < 1.0:
        raise ValueError("minimum improvement must be strictly between zero and one")

    accepted_prefixes: list[int] = []
    excluded_tail_rounds = 0
    for raw_event in events:
        event = validate_confidence_event(
            raw_event, max_draft_tokens=max_draft_tokens
        )
        for proposal_len, correct in zip(
            event["proposal_lens"], event["num_correct_drafts"]
        ):
            if proposal_len == max_draft_tokens:
                accepted_prefixes.append(correct)
            else:
                excluded_tail_rounds += 1
    if not accepted_prefixes:
        raise ValueError("offline replay found no full-M16 request-round")

    common_budget = math.lcm(*(value + 1 for value in candidates))
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        emitted = [min(correct, candidate) + 1 for correct in accepted_prefixes]
        mean_yield = sum(emitted) / len(emitted)
        target_rows_per_emitted = (candidate + 1) / mean_yield
        rounds_at_budget = common_budget // (candidate + 1)
        rows.append(
            {
                "verify_prefix": candidate,
                "full_m16_request_round_count": len(accepted_prefixes),
                "mean_emitted_tokens_per_verify": mean_yield,
                "exact_target_rows_per_verify": candidate + 1,
                "exact_target_rows_per_emitted_token": target_rows_per_emitted,
                "common_exact_target_row_budget": common_budget,
                "integer_verify_rounds_at_common_budget": rounds_at_budget,
                "projected_emitted_tokens_at_common_budget": (
                    rounds_at_budget * mean_yield
                ),
            }
        )

    baseline = rows[-1]["exact_target_rows_per_emitted_token"]
    for row in rows:
        row["c_over_y_improvement_vs_m16"] = (
            baseline - row["exact_target_rows_per_emitted_token"]
        ) / baseline
    short_rows = rows[:-1]
    winner = min(
        short_rows, key=lambda row: row["exact_target_rows_per_emitted_token"]
    )
    best_improvement = float(winner["c_over_y_improvement_vs_m16"])
    passed = best_improvement >= minimum_improvement
    return {
        "authority": (
            "event-local replay of exact-target accepted-prefix lengths from full "
            "production M16 proposals"
        ),
        "scope": (
            "fixed-prefix one-step projection only; earlier target refreshes can "
            "change later proposal states and require a measured follow-up"
        ),
        "full_m16_request_round_count": len(accepted_prefixes),
        "excluded_short_final_round_count": excluded_tail_rounds,
        "candidate_rows": rows,
        "best_short_candidate": winner["verify_prefix"],
        "best_c_over_y_improvement_vs_m16": best_improvement,
        "gate": {
            "id": "projected_c_over_y_improves",
            "metric": "exact target verify rows per emitted token",
            "threshold": f"best M=8/12 candidate improves at least {minimum_improvement:.0%} vs M16",
            "status": "pass" if passed else "fail",
        },
    }


def subset_confidence_events(
    events: Iterable[dict[str, Any]], *, request_ids: set[str]
) -> list[dict[str, Any]]:
    """Select complete request rows without changing their within-event order."""

    if not request_ids or any(not isinstance(value, str) or not value for value in request_ids):
        raise ValueError("confidence subset requires non-empty request IDs")
    selected: list[dict[str, Any]] = []
    observed: set[str] = set()
    row_fields = (
        "request_ids",
        "output_start_indices",
        "proposal_lens",
        "draft_tokens",
        "draft_top1_logprob",
        "cumulative_draft_logprob",
        "num_correct_drafts",
        "accept_lens",
    )
    for raw_event in events:
        event = validate_confidence_event(raw_event)
        indices = [
            index
            for index, request_id in enumerate(event["request_ids"])
            if request_id in request_ids
        ]
        if not indices:
            continue
        copied = dict(event)
        for name in row_fields:
            copied[name] = [event[name][index] for index in indices]
        validate_confidence_event(copied)
        observed.update(copied["request_ids"])
        selected.append(copied)
    if observed != request_ids:
        raise ValueError(
            "confidence subset did not observe exactly the requested IDs: "
            f"missing={sorted(request_ids - observed)}, "
            f"extra={sorted(observed - request_ids)}"
        )
    return selected


def _full_m16_rows(events: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    excluded_tail = 0
    for raw_event in events:
        event = validate_confidence_event(raw_event)
        for request_id, proposal_len, correct, cumulative in zip(
            event["request_ids"],
            event["proposal_lens"],
            event["num_correct_drafts"],
            event["cumulative_draft_logprob"],
        ):
            if proposal_len != CONFIDENCE_MAX_DRAFT_TOKENS:
                excluded_tail += 1
                continue
            rows.append(
                {
                    "request_id": request_id,
                    "accepted_prefix": correct,
                    "s8": float(cumulative[7]),
                    "s12": float(cumulative[11]),
                }
            )
    if not rows:
        raise ValueError("variable-prefix replay found no full-M16 rows")
    return rows, excluded_tail


def _policy_summary(
    rows: list[dict[str, Any]],
    *,
    threshold_s8: float | None,
    threshold_s12: float | None,
) -> dict[str, Any]:
    decisions: list[int] = []
    yields: list[int] = []
    for row in rows:
        prefix = 8
        if threshold_s8 is not None and row["s8"] >= threshold_s8:
            prefix = 12
            if threshold_s12 is not None and row["s12"] >= threshold_s12:
                prefix = 16
        decisions.append(prefix)
        yields.append(min(int(row["accepted_prefix"]), prefix) + 1)
    target_rows = sum(prefix + 1 for prefix in decisions)
    emitted = sum(yields)
    c_over_y = target_rows / emitted
    return {
        "request_round_count": len(rows),
        "chosen_prefix_histogram": {
            str(prefix): decisions.count(prefix) for prefix in CANDIDATE_VERIFY_PREFIXES
        },
        "mean_executed_recurrent_steps": sum(decisions) / len(decisions),
        "exact_target_rows": target_rows,
        "emitted_tokens": emitted,
        "exact_target_rows_per_emitted_token": c_over_y,
        "projected_emitted_tokens_at_common_target_row_budget": (
            COMMON_TARGET_ROW_BUDGET / c_over_y
        ),
    }


def calibrate_and_replay_staged_policy(
    calibration_events: Iterable[dict[str, Any]],
    evaluation_events: Iterable[dict[str, Any]],
    *,
    minimum_improvement: float = 0.05,
) -> dict[str, Any]:
    """Calibrate an 8->12->16 confidence policy, then replay it disjointly.

    The policy's production shape is staged: every row executes eight recurrent
    steps; only rows with ``s8 >= t8`` execute through step twelve; among those,
    only rows with ``s12 >= t12`` execute through step sixteen.  Full-M16 traces
    provide counterfactual scores, but projected cost charges only the selected
    staged prefix.  It never models generating sixteen steps and truncating.
    """

    if not 0.0 < minimum_improvement < 1.0:
        raise ValueError("minimum improvement must be strictly between zero and one")
    calibration_rows, calibration_tails = _full_m16_rows(calibration_events)
    evaluation_rows, evaluation_tails = _full_m16_rows(evaluation_events)
    calibration_ids = {row["request_id"] for row in calibration_rows}
    evaluation_ids = {row["request_id"] for row in evaluation_rows}
    if calibration_ids & evaluation_ids:
        raise ValueError("calibration and evaluation request IDs overlap")

    threshold_s8_candidates: list[float | None] = [None, *sorted({row["s8"] for row in calibration_rows})]
    threshold_s12_candidates: list[float | None] = [None, *sorted({row["s12"] for row in calibration_rows})]
    candidates: list[dict[str, Any]] = []
    for threshold_s8 in threshold_s8_candidates:
        # With no 8->12 extension, t12 is unreachable and has one canonical value.
        active_s12 = [None] if threshold_s8 is None else threshold_s12_candidates
        for threshold_s12 in active_s12:
            summary = _policy_summary(
                calibration_rows,
                threshold_s8=threshold_s8,
                threshold_s12=threshold_s12,
            )
            candidates.append(
                {
                    "threshold_s8": threshold_s8,
                    "threshold_s12": threshold_s12,
                    "calibration": summary,
                }
            )
    winner = min(
        candidates,
        key=lambda item: (
            item["calibration"]["exact_target_rows_per_emitted_token"],
            item["calibration"]["mean_executed_recurrent_steps"],
            item["threshold_s8"] is not None,
            item["threshold_s8"] if item["threshold_s8"] is not None else 0.0,
            item["threshold_s12"] is not None,
            item["threshold_s12"] if item["threshold_s12"] is not None else 0.0,
        ),
    )
    evaluation = _policy_summary(
        evaluation_rows,
        threshold_s8=winner["threshold_s8"],
        threshold_s12=winner["threshold_s12"],
    )

    fixed_controls = replay_verify_prefixes(evaluation_events)
    best_fixed = min(
        fixed_controls["candidate_rows"],
        key=lambda row: row["exact_target_rows_per_emitted_token"],
    )
    variable_c_over_y = evaluation["exact_target_rows_per_emitted_token"]
    fixed_c_over_y = best_fixed["exact_target_rows_per_emitted_token"]
    improvement = (fixed_c_over_y - variable_c_over_y) / fixed_c_over_y
    passed = improvement >= minimum_improvement
    return {
        "policy_shape": (
            "generate every row to 8; extend rows with s8>=t8 to 12; extend "
            "those rows with s12>=t12 to 16"
        ),
        "not_admitted_shape": "generate all rows to 16 and truncate after scoring",
        "score": "s_k=sum_{j<=k} log p_recurrent(argmax_j)",
        "threshold_semantics": "null means do not extend at that stage; comparison is >=",
        "calibration": {
            "request_ids": sorted(calibration_ids),
            "full_m16_request_round_count": len(calibration_rows),
            "excluded_short_final_round_count": calibration_tails,
            "candidate_threshold_pair_count": len(candidates),
            "selected_threshold_s8": winner["threshold_s8"],
            "selected_threshold_s12": winner["threshold_s12"],
            "selected_policy": winner["calibration"],
            "selection_objective": (
                "minimum exact target rows per emitted token; deterministic "
                "tie-break toward fewer recurrent steps"
            ),
        },
        "evaluation": {
            "request_ids": sorted(evaluation_ids),
            "full_m16_request_round_count": len(evaluation_rows),
            "excluded_short_final_round_count": evaluation_tails,
            "unchanged_threshold_policy": evaluation,
            "fixed_prefix_controls": fixed_controls["candidate_rows"],
            "best_fixed_prefix": best_fixed["verify_prefix"],
            "best_fixed_exact_target_rows_per_emitted_token": fixed_c_over_y,
            "variable_policy_c_over_y_improvement_vs_best_fixed": improvement,
        },
        "scope": (
            "one-step anchored projection; executing an earlier exact refresh can "
            "change later anchors and requires a measured online follow-up"
        ),
        "gate": {
            "id": "heldout_variable_policy_c_over_y_improves",
            "metric": "exact target verify rows per emitted token",
            "threshold": (
                f"unchanged staged policy improves at least {minimum_improvement:.0%} "
                "over the best fixed M=8/12/16 control on disjoint inputs"
            ),
            "status": "pass" if passed else "fail",
        },
    }
