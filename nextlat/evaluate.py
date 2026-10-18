"""Full recursive-state gate for the fixed Qwen3-1.7B NextLat path."""

from __future__ import annotations

import argparse
import json
import math
import shlex
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
from datasets import Dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from .core import NextLatDynamics
from .eval_recursive import (
    benchmark_edge_speed,
    exact_sampled_random_continuation_metrics,
    proposal_recursive_metrics,
    teacher_forced_recursive_metrics,
)
from .run_utils import (
    assert_claim_source_committed,
    initialize_run_directory,
    load_dynamics_checkpoint,
    verify_checkpoint,
    write_status,
)
from .train import NextLatCollator, _to_device, _validate_prepared_data
from .tracking import start_wandb_tracker


def _nearest_rank(values: Sequence[int], quantile: float) -> int:
    if not values or not 0 < quantile <= 1:
        raise ValueError("nearest-rank quantile requires values and 0 < q <= 1")
    ordered = sorted(int(value) for value in values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def _accepted_prefix_summary(values: Sequence[int]) -> Dict[str, Any]:
    accepted = [int(value) for value in values]
    if not accepted:
        raise RuntimeError("accepted-prefix population is empty")
    return {
        "mean": sum(accepted) / len(accepted),
        "median": _nearest_rank(accepted, 0.5),
        "p90": _nearest_rank(accepted, 0.9),
        "p95": _nearest_rank(accepted, 0.95),
        "p99": _nearest_rank(accepted, 0.99),
        "maximum": max(accepted),
        "at_least_4_rate": sum(value >= 4 for value in accepted) / len(accepted),
        "at_least_8_rate": sum(value >= 8 for value in accepted) / len(accepted),
        "at_least_12_rate": sum(value >= 12 for value in accepted) / len(accepted),
        "at_least_16_rate": sum(value >= 16 for value in accepted) / len(accepted),
        "values": accepted,
    }


def _aggregate_depth_records(
    records: Sequence[Dict[str, Any]], weight_key: str
) -> Dict[str, Dict[str, float]]:
    sums: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    weights: Dict[str, float] = defaultdict(float)
    for record in records:
        weight = float(record.get(weight_key, 1))
        for depth, metrics in record["depths"].items():
            weights[depth] += weight
            for name, value in metrics.items():
                if not math.isfinite(float(value)):
                    raise RuntimeError(f"non-finite {name} at depth {depth}")
                sums[depth][name] += float(value) * weight
    return {
        depth: {name: value / weights[depth] for name, value in metrics.items()}
        for depth, metrics in sorted(sums.items(), key=lambda item: int(item[0]))
    }


def _evaluate_population(
    *,
    model: torch.nn.Module,
    dynamics: NextLatDynamics,
    dataset: Dataset,
    collator: NextLatCollator,
    device: torch.device,
    depths: Sequence[int],
    proposal_length: int,
    examples: int,
    label: str,
) -> Dict[str, Any]:
    teacher_records: List[Dict[str, Any]] = []
    proposal_records: List[Dict[str, Any]] = []
    random_records: List[Dict[str, Any]] = []
    problem_hashes: List[str] = []
    for index in range(min(examples, len(dataset))):
        record = dataset[index]
        batch = _to_device(collator([record]), device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            try:
                teacher = teacher_forced_recursive_metrics(
                    model=model,
                    dynamics=dynamics,
                    batch=batch,
                    depths=depths,
                    anchors_per_example=2,
                )
            except RuntimeError as error:
                if "no held-out response has room" not in str(error):
                    raise
            else:
                teacher_records.append(teacher)
            proposal_records.append(
                proposal_recursive_metrics(
                    model=model,
                    dynamics=dynamics,
                    batch=batch,
                    depths=depths,
                    proposal_length=proposal_length,
                )
            )
            random_records.append(
                exact_sampled_random_continuation_metrics(
                    model=model,
                    dynamics=dynamics,
                    batch=batch,
                    depths=depths,
                    continuation_length=proposal_length,
                )
            )
        problem_hashes.append(str(record["problem_hash"]))
    if not teacher_records or not proposal_records or not random_records:
        raise RuntimeError(f"population {label} did not yield every required metric")

    accepted = [
        int(value)
        for record in proposal_records
        for value in record["accepted_prefix"]["values"]
    ]
    return {
        "label": label,
        "problems": len(problem_hashes),
        "problem_hashes": problem_hashes,
        "teacher_forced_gold_continuations": {
            "records": len(teacher_records),
            "anchors": sum(int(record["anchors"]) for record in teacher_records),
            "depths": _aggregate_depth_records(teacher_records, "anchors"),
        },
        "latent_greedy_proposals": {
            "records": len(proposal_records),
            "accepted_prefix": _accepted_prefix_summary(accepted),
            "depths": _aggregate_depth_records(proposal_records, "examples"),
        },
        "exact_target_sampled_random_continuations": {
            "records": len(random_records),
            "depths": _aggregate_depth_records(random_records, "examples"),
        },
    }


def _load_search_selected(path: Path, stress_candidate: Dataset) -> Dataset:
    selections: List[Dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record.get("problem_hash"), str):
            raise RuntimeError(f"search selection line {line_number} has no problem_hash")
        if not isinstance(record.get("exact_search_run"), str):
            raise RuntimeError(
                f"search selection line {line_number} has no exact_search_run provenance"
            )
        selections.append(record)
    hashes = [record["problem_hash"] for record in selections]
    if not hashes or len(hashes) != len(set(hashes)):
        raise RuntimeError("search-selected problem hashes must be nonempty and unique")
    by_hash = {
        str(problem_hash): index
        for index, problem_hash in enumerate(stress_candidate["problem_hash"])
    }
    missing = set(hashes) - set(by_hash)
    if missing:
        raise RuntimeError(
            f"search-selected file contains {len(missing)} problems outside candidate split"
        )
    return stress_candidate.select([by_hash[value] for value in hashes])


def evaluate(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("full recursive evaluation requires CUDA")
    source_commit = assert_claim_source_committed([args.config])
    config = json.loads(args.config.read_text())
    tracking_config = config.get("tracking")
    if tracking_config is not None and args.wandb_run_id is None:
        raise RuntimeError("--wandb-run-id is required by this config")
    verify_checkpoint(args.checkpoint)
    train_state = json.loads((args.checkpoint / "train_state.json").read_text())
    if train_state.get("source_code_commit") != source_commit:
        raise RuntimeError("checkpoint and evaluator source commits do not match")
    if tracking_config is not None and train_state.get("wandb_run_id") != args.wandb_run_id:
        raise RuntimeError("checkpoint W&B run ID does not match --wandb-run-id")

    train_data = load_from_disk(str(args.prepared_root / "train"))
    holdout = load_from_disk(str(args.prepared_root / "recursive_holdout"))
    stress_candidate = load_from_disk(str(args.prepared_root / "stress_candidate"))
    prepared_manifest = _validate_prepared_data(
        args.prepared_root,
        config,
        {
            "train": train_data,
            "recursive_holdout": holdout,
            "stress_candidate": stress_candidate,
        },
    )
    del train_data

    resolved_config = json.loads(json.dumps(config))
    resolved_config["evaluation"] = {
        "checkpoint": str(args.checkpoint),
        "examples": args.examples,
        "search_selected_jsonl": str(args.search_selected_jsonl)
        if args.search_selected_jsonl
        else None,
        "source_code_commit": source_commit,
    }
    initialize_run_directory(
        args.run_dir,
        resolved_config,
        " ".join(shlex.quote(part) for part in sys.argv),
    )
    tracker = None
    if tracking_config is not None:
        tracker = start_wandb_tracker(
            tracking_config=tracking_config,
            run_dir=args.run_dir,
            run_id=str(args.wandb_run_id),
            resume_mode="must",
            experiment_config=config,
            source_code_commit=source_commit,
            phase="evaluation",
        )

    device = torch.device("cuda", 0)
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint / "model", local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint / "model",
        torch_dtype=torch.float32,
        attn_implementation=config["model"]["attention"],
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to(device)
    dynamics = NextLatDynamics(
        int(model.config.hidden_size),
        projection_factor=float(config["nextlat"]["projection_factor"]),
        dropout=float(config["nextlat"]["dropout"]),
    ).to(device=device, dtype=torch.float32)
    load_dynamics_checkpoint(dynamics, args.checkpoint, device)
    model.eval()
    dynamics.eval()
    collator = NextLatCollator(
        tokenizer.pad_token_id, int(config["data"]["sequence_length"])
    )
    depths = [int(value) for value in config["recursive_gate"]["depths"]]
    proposal_length = int(config["recursive_gate"]["proposal_length"])

    holdout_result = _evaluate_population(
        model=model,
        dynamics=dynamics,
        dataset=holdout,
        collator=collator,
        device=device,
        depths=depths,
        proposal_length=proposal_length,
        examples=args.examples,
        label="untouched_problem_holdout",
    )
    search_selected_result = None
    if args.search_selected_jsonl is not None:
        selected = _load_search_selected(
            args.search_selected_jsonl, stress_candidate
        )
        search_selected_result = _evaluate_population(
            model=model,
            dynamics=dynamics,
            dataset=selected,
            collator=collator,
            device=device,
            depths=depths,
            proposal_length=proposal_length,
            examples=len(selected),
            label="exact_search_selected_stress",
        )

    speed_record = holdout[0]
    speed_batch = _to_device(collator([speed_record]), device)
    response_positions = torch.nonzero(
        speed_batch["assistant_loss_mask"][0], as_tuple=False
    )
    if response_positions.numel() == 0:
        raise RuntimeError("speed benchmark record has no response tokens")
    anchor = max(0, int(response_positions[0].item()) - 1)
    prefix_ids = speed_batch["input_ids"][0:1, : anchor + 1]
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        speed = benchmark_edge_speed(
            model=model,
            dynamics=dynamics,
            prefix_ids=prefix_ids,
            edge_length=proposal_length,
            repeats=args.speed_repeats,
        )

    maximum_accepted = int(
        holdout_result["latent_greedy_proposals"]["accepted_prefix"]["maximum"]
    )
    useful_16_edge = maximum_accepted >= int(
        config["recursive_gate"]["minimum_useful_edge"]
    )
    full_population_coverage = search_selected_result is not None
    result_state = "claim-grade" if full_population_coverage else "observed"
    summary = {
        "result_state": result_state,
        "source_code_commit": source_commit,
        "checkpoint": str(args.checkpoint),
        "checkpoint_tokens_seen": int(train_state["tokens_seen"]),
        "prepared_manifest_sha256": prepared_manifest["manifest_sha256"],
        "holdout": holdout_result,
        "search_selected_stress": search_selected_result,
        "speed": speed,
        "gates": {
            "useful_16_token_edge": useful_16_edge,
            "search_selected_population_present": full_population_coverage,
            "recursive_state_gate_passed": useful_16_edge
            and full_population_coverage,
        },
    }
    (args.run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    if tracker is not None:
        tracker.log_record(
            "evaluation",
            summary,
            tokens_seen=int(train_state["tokens_seen"]),
            update=int(train_state["update"]),
        )
    write_status(
        args.run_dir,
        result_state=result_state,
        passed=[
            item
            for condition, item in (
                (useful_16_edge, "at least one exact-accepted 16-token latent edge"),
                (
                    full_population_coverage,
                    "gold random proposal and search-selected populations measured",
                ),
            )
            if condition
        ],
        failed_or_missing=[
            item
            for condition, item in (
                (useful_16_edge, "useful 16-token latent edge"),
                (full_population_coverage, "exact-search-selected stress set"),
            )
            if not condition
        ],
        main_metric=f"maximum exact-accepted latent prefix {maximum_accepted} tokens",
        baseline="exact target recurrence at the same anchors",
        caveats=[] if full_population_coverage else ["stress candidates not yet search-selected"],
        next_target="minimal latent-search comparison"
        if useful_16_edge and full_population_coverage
        else "stop or apply the single d=4 repair only if one-step dynamics are strong",
    )
    print(json.dumps(summary["gates"], sort_keys=True), flush=True)
    if tracker is not None:
        tracker.finish(summary_namespace="last_evaluation", summary=summary)


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=16)
    parser.add_argument("--speed-repeats", type=int, default=5)
    parser.add_argument("--search-selected-jsonl", type=Path)
    parser.add_argument("--wandb-run-id")
    args = parser.parse_args(argv)
    if args.examples <= 0 or args.speed_repeats <= 0:
        parser.error("--examples and --speed-repeats must be positive")
    evaluate(args)


if __name__ == "__main__":
    main()
