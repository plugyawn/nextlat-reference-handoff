"""Prepare one verified OpenR1 corpus with problem-level holdouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import shlex
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import datasets
import transformers
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

from .run_utils import assert_claim_source_committed


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _problem_hash(problem: str) -> str:
    normalized = re.sub(r"\s+", " ", problem).strip()
    return _sha256(normalized)


def _split_for(problem_hash: str, seed: int) -> Tuple[str, str]:
    digest = _sha256(f"split:{seed}:{problem_hash}")
    bucket = int(digest[:16], 16) % 10_000
    if bucket < 9_500:
        split = "train"
    elif bucket < 9_750:
        split = "recursive_holdout"
    else:
        split = "stress_candidate"
    order = _sha256(f"order:{seed}:{problem_hash}")
    return split, order


def _verified_generation_indices(row: Dict[str, Any]) -> List[int]:
    generations = row.get("generations")
    verified = row.get("correctness_math_verify")
    complete = row.get("is_reasoning_complete")
    if not isinstance(generations, list) or not isinstance(verified, list):
        return []
    if not isinstance(complete, list):
        return []
    return [
        index
        for index, (generation, is_verified, is_complete) in enumerate(
            zip(generations, verified, complete)
        )
        if is_verified is True
        and is_complete is True
        and isinstance(generation, str)
        and generation.strip()
    ]


def _encode(
    row: Dict[str, Any],
    generation_index: int,
    tokenizer: Any,
    sequence_length: int,
    problem_hash: str,
) -> Tuple[Optional[Dict[str, Any]], str]:
    problem = row.get("problem")
    generations = row.get("generations")
    uuid = row.get("uuid")
    if not isinstance(problem, str) or not isinstance(generations, list):
        return None, "invalid_problem_or_generations"
    if generation_index >= len(generations):
        return None, "generation_index_out_of_range"
    assistant = generations[generation_index]
    if not isinstance(assistant, str) or not assistant.strip():
        return None, "empty_generation"

    prompt_messages = [{"role": "user", "content": problem}]
    full_messages = prompt_messages + [{"role": "assistant", "content": assistant}]
    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    full_ids = tokenizer.apply_chat_template(
        full_messages,
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=True,
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise RuntimeError(
            f"Qwen chat-template prefix mismatch for problem {problem_hash}"
        )
    if len(full_ids) > sequence_length:
        return None, "verified_complete_trace_over_sequence_length"
    input_ids = full_ids
    prompt_length = min(len(prompt_ids), len(input_ids))
    assistant_loss_mask = [False] * prompt_length + [True] * (
        len(input_ids) - prompt_length
    )
    if len(input_ids) < 3 or sum(assistant_loss_mask) < 2:
        return None, "too_few_trainable_tokens"

    return {
        "uuid": str(uuid) if uuid is not None else problem_hash,
        "problem_hash": problem_hash,
        "source": str(row.get("source") or "unknown"),
        "generation_index": generation_index,
        "input_ids": input_ids,
        "assistant_loss_mask": assistant_loss_mask,
        "token_count": len(input_ids),
        "assistant_token_count": sum(assistant_loss_mask),
        "was_truncated": False,
    }, "selected"


def split_receipt(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Hash every saved training field in row order and summarize the split."""

    record_digest = hashlib.sha256()
    problem_hashes: List[str] = []
    problems = tokens = assistant_tokens = truncated = 0
    for ordinal, record in enumerate(records):
        canonical = {
            "ordinal": ordinal,
            "uuid": str(record["uuid"]),
            "problem_hash": str(record["problem_hash"]),
            "source": str(record["source"]),
            "generation_index": int(record["generation_index"]),
            "input_ids": [int(value) for value in record["input_ids"]],
            "assistant_loss_mask": [
                bool(value) for value in record["assistant_loss_mask"]
            ],
            "token_count": int(record["token_count"]),
            "assistant_token_count": int(record["assistant_token_count"]),
            "was_truncated": bool(record["was_truncated"]),
        }
        record_digest.update(
            json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        record_digest.update(b"\n")
        problem_hashes.append(canonical["problem_hash"])
        problems += 1
        tokens += canonical["token_count"]
        assistant_tokens += canonical["assistant_token_count"]
        truncated += int(canonical["was_truncated"])
    return {
        "problems": problems,
        "tokens": tokens,
        "assistant_tokens": assistant_tokens,
        "truncated_problems": truncated,
        "problem_hash_digest": _sha256("\n".join(sorted(problem_hashes))),
        "record_digest": record_digest.hexdigest(),
    }


def _save_split(records: List[Dict[str, Any]], destination: Path) -> None:
    if not records:
        raise RuntimeError(f"refusing to save empty split {destination.name}")
    dataset = Dataset.from_list(records)
    dataset.save_to_disk(str(destination))


def prepare(config_path: Path, output_root: Path) -> Path:
    source_commit = assert_claim_source_committed([config_path])
    config = json.loads(config_path.read_text())
    dataset_config = config["dataset"]
    model_config = config["model"]
    data_config = config["data"]
    seed = int(dataset_config["split_seed"])
    sequence_length = int(data_config["sequence_length"])

    output_dir = output_root / data_config["prepared_name"]
    if output_dir.exists():
        raise FileExistsError(
            f"prepared output already exists: {output_dir}; use a new destination"
        )
    output_dir.mkdir(parents=True)
    (output_dir / "INCOMPLETE").write_text("prepared-data build in progress\n")
    (output_dir / "config.resolved.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "command.txt").write_text(
        " ".join(shlex.quote(part) for part in sys.argv) + "\n"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_config["id"], revision=model_config["revision"]
    )
    source = load_dataset(
        dataset_config["id"],
        dataset_config["config"],
        split=dataset_config["split"],
        revision=dataset_config["revision"],
    )

    # Scan only the columns needed to establish the split and verification gate.
    uuids = source["uuid"]
    problems = source["problem"]
    verified = source["correctness_math_verify"]
    complete = source["is_reasoning_complete"]
    candidates: Dict[str, List[Tuple[str, int, str]]] = {
        "train": [],
        "recursive_holdout": [],
        "stress_candidate": [],
    }
    skip_counts: Counter[str] = Counter()
    seen_problem_hashes = set()
    for index, (uuid, problem, checks, completeness) in enumerate(
        zip(uuids, problems, verified, complete)
    ):
        del uuid  # UUID is retained in the selected row, not trusted for splitting.
        if not isinstance(problem, str) or not problem.strip():
            skip_counts["invalid_problem"] += 1
            continue
        if not isinstance(checks, list) or not isinstance(completeness, list):
            skip_counts["invalid_verification_fields"] += 1
            continue
        if not any(
            check is True and done is True
            for check, done in zip(checks, completeness)
        ):
            skip_counts["no_math_verified_complete_trace"] += 1
            continue
        problem_hash = _problem_hash(problem)
        if problem_hash in seen_problem_hashes:
            skip_counts["duplicate_problem_text"] += 1
            continue
        seen_problem_hashes.add(problem_hash)
        split, order = _split_for(problem_hash, seed)
        candidates[split].append((order, index, problem_hash))

    for split_candidates in candidates.values():
        split_candidates.sort()

    caps = {
        "recursive_holdout": int(data_config["recursive_holdout_problems"]),
        "stress_candidate": int(data_config["stress_candidate_problems"]),
    }
    records: Dict[str, List[Dict[str, Any]]] = {
        "train": [],
        "recursive_holdout": [],
        "stress_candidate": [],
    }
    train_token_target = int(data_config["prepared_train_tokens"])
    train_tokens = 0
    encode_skips: Counter[str] = Counter()

    for split in ("recursive_holdout", "stress_candidate", "train"):
        for _, index, problem_hash in candidates[split]:
            if split != "train" and len(records[split]) >= caps[split]:
                break
            if split == "train" and train_tokens >= train_token_target:
                break
            row = source[index]
            generation_indices = _verified_generation_indices(row)
            if not generation_indices:
                encode_skips["verified_index_mismatch"] += 1
                continue
            encoded = None
            rejection_reasons: List[str] = []
            for generation_index in generation_indices:
                encoded, reason = _encode(
                    row,
                    generation_index,
                    tokenizer,
                    sequence_length,
                    problem_hash,
                )
                if encoded is not None:
                    break
                rejection_reasons.append(reason)
            if encoded is None:
                if rejection_reasons and all(
                    reason == "verified_complete_trace_over_sequence_length"
                    for reason in rejection_reasons
                ):
                    encode_skips[
                        "all_verified_complete_traces_over_sequence_length"
                    ] += 1
                else:
                    encode_skips["all_verified_complete_traces_rejected"] += 1
                continue
            records[split].append(encoded)
            if split == "train":
                train_tokens += encoded["token_count"]

    if train_tokens < train_token_target:
        raise RuntimeError(
            f"only prepared {train_tokens:,} train tokens; target was {train_token_target:,}"
        )
    for split, cap in caps.items():
        if len(records[split]) != cap:
            raise RuntimeError(
                f"prepared {len(records[split])} {split} problems; required {cap}"
            )

    split_hashes = {
        split: {record["problem_hash"] for record in split_records}
        for split, split_records in records.items()
    }
    for left, right in (
        ("train", "recursive_holdout"),
        ("train", "stress_candidate"),
        ("recursive_holdout", "stress_candidate"),
    ):
        overlap = split_hashes[left] & split_hashes[right]
        if overlap:
            raise RuntimeError(f"problem leakage between {left} and {right}")

    for split, split_records in records.items():
        _save_split(split_records, output_dir / split)

    manifest = {
        "schema_version": 1,
        "result_state": "observed",
        "source": {
            "dataset": dataset_config["id"],
            "config": dataset_config["config"],
            "split": dataset_config["split"],
            "revision": dataset_config["revision"],
            "fingerprint": source._fingerprint,
        },
        "tokenizer": {
            "model": model_config["id"],
            "revision": model_config["revision"],
            "sequence_length": sequence_length,
        },
        "selection": {
            "require_correctness_math_verify": True,
            "require_reasoning_complete": True,
            "one_trajectory_per_problem": True,
            "reject_trajectories_over_sequence_length": True,
            "problem_key": "sha256(normalized problem text)",
            "split_seed": seed,
            "split_buckets": {
                "train": "[0,9500)",
                "recursive_holdout": "[9500,9750)",
                "stress_candidate": "[9750,10000)",
            },
            "stress_candidate_is_not_yet_search_selected": True,
        },
        "targets": {
            "prepared_train_tokens": train_token_target,
            "recursive_holdout_problems": caps["recursive_holdout"],
            "stress_candidate_problems": caps["stress_candidate"],
        },
        "splits": {},
        "skip_counts": dict(skip_counts + encode_skips),
        "source_code_commit": source_commit,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "datasets": datasets.__version__,
            "transformers": transformers.__version__,
        },
        "config_digest": _sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":"))
        ),
    }
    for split, split_records in records.items():
        manifest["splits"][split] = split_receipt(split_records)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "INCOMPLETE").unlink()
    (output_dir / "COMPLETE").write_text("ok\n")
    return output_dir


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    output = prepare(args.config, args.output_root)
    print(output)


if __name__ == "__main__":
    main()
