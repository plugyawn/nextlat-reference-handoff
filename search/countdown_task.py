"""Strict, dependency-free Countdown-4 task and expression helpers.

The model-facing protocol forces the literal prefix ``FINAL: `` and asks the
model to complete it with one arithmetic expression.  ``verify_expression``
therefore accepts exactly one nonempty line containing only that expression.
Free-form reasoning, if desired, belongs before the forced prefix and is not
part of ``text``.

The verifier intentionally does not use ``eval``.  It parses a Python
expression, admits only integer literals and the four Countdown binary
operators, then evaluates with :class:`fractions.Fraction`.
"""

from __future__ import annotations

import ast
import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Iterable


_OPERATOR_SYMBOLS: dict[type[ast.operator], str] = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
}
_EXPRESSION_CHARACTERS = re.compile(r"[0-9.()+\-*/ \t]+")


@dataclass(frozen=True, slots=True)
class CountdownTask:
    """One four-input integer Countdown problem and generator metadata."""

    inputs: tuple[int, ...]
    target: int
    witness: str
    solution_count: int


@dataclass(frozen=True, slots=True)
class _Term:
    value: int
    expression: str


class _VerificationFailure(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _task_error(task: CountdownTask) -> str | None:
    if len(task.inputs) != 4:
        return "task_input_count"
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in task.inputs
    ):
        return "task_noninteger_input"
    if any(value < 1 or value > 99 for value in task.inputs):
        return "task_input_range"
    if isinstance(task.target, bool) or not isinstance(task.target, int):
        return "task_noninteger_target"
    if task.target < 10 or task.target > 100:
        return "task_target_range"
    return None


def task_hash(task: CountdownTask) -> str:
    """Hash the mathematical problem, independent of witness metadata."""

    problem = {
        "inputs": sorted(int(value) for value in task.inputs),
        "target": int(task.target),
    }
    payload = json.dumps(problem, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalized_expression(node: ast.AST) -> str:
    if isinstance(node, ast.Constant):
        return str(node.value)
    if isinstance(node, ast.BinOp):
        symbol = _OPERATOR_SYMBOLS.get(type(node.op))
        if symbol is None:
            raise _VerificationFailure("unsupported_operator")
        return (
            f"({_normalized_expression(node.left)}{symbol}"
            f"{_normalized_expression(node.right)})"
        )
    raise _VerificationFailure("disallowed_syntax")


def _inspect_expression(node: ast.AST) -> tuple[int, list[int]]:
    """Return binary-op count and integer leaves under an AST whitelist."""

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int):
            raise _VerificationFailure("noninteger_literal")
        return 0, [node.value]
    if isinstance(node, ast.BinOp):
        if type(node.op) not in _OPERATOR_SYMBOLS:
            raise _VerificationFailure("unsupported_operator")
        left_ops, left_leaves = _inspect_expression(node.left)
        right_ops, right_leaves = _inspect_expression(node.right)
        return 1 + left_ops + right_ops, left_leaves + right_leaves
    raise _VerificationFailure("disallowed_syntax")


def _evaluate_expression(node: ast.AST) -> Fraction:
    if isinstance(node, ast.Constant):
        return Fraction(node.value)
    if not isinstance(node, ast.BinOp):
        raise _VerificationFailure("disallowed_syntax")

    left = _evaluate_expression(node.left)
    right = _evaluate_expression(node.right)
    if isinstance(node.op, ast.Add):
        return left + right
    if isinstance(node.op, ast.Mult):
        return left * right
    if isinstance(node.op, ast.Sub):
        if left < right:
            raise _VerificationFailure("negative_subtraction")
        return left - right
    if isinstance(node.op, ast.Div):
        if right == 0:
            raise _VerificationFailure("division_by_zero")
        result = left / right
        if result.denominator != 1:
            raise _VerificationFailure("nonexact_division")
        return result
    raise _VerificationFailure("unsupported_operator")


def _verification_result(
    *,
    passed: bool,
    failure_code: str,
    normalized_expression: str | None,
    value: int | None,
    operation_count: int | None = None,
) -> dict[str, Any]:
    return {
        "passed": passed,
        "failure_code": failure_code,
        "normalized_expression": normalized_expression,
        "value": value,
        "operation_count": operation_count,
    }


def verify_expression(text: str, task: CountdownTask) -> dict[str, Any]:
    """Verify one strict expression completion after a forced ``FINAL: ``.

    Blank surrounding lines are ignored, but more than one nonempty line is a
    hard failure.  A passing expression has exactly three binary operations,
    uses the four input integers with exact multiset multiplicity, never makes
    a negative subtraction, uses only exact integer division, and equals the
    target under exact rational arithmetic.
    """

    task_error = _task_error(task)
    if task_error is not None:
        return _verification_result(
            passed=False,
            failure_code=task_error,
            normalized_expression=None,
            value=None,
        )
    if not isinstance(text, str):
        return _verification_result(
            passed=False,
            failure_code="nonstring_completion",
            normalized_expression=None,
            value=None,
        )

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return _verification_result(
            passed=False,
            failure_code="empty_completion",
            normalized_expression=None,
            value=None,
        )
    if len(lines) != 1:
        return _verification_result(
            passed=False,
            failure_code="extra_nonempty_lines",
            normalized_expression=None,
            value=None,
        )
    expression = lines[0]
    if _EXPRESSION_CHARACTERS.fullmatch(expression) is None:
        return _verification_result(
            passed=False,
            failure_code="invalid_character",
            normalized_expression=None,
            value=None,
        )
    try:
        parsed = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError):
        return _verification_result(
            passed=False,
            failure_code="syntax_error",
            normalized_expression=None,
            value=None,
        )

    normalized: str | None = None
    operation_count: int | None = None
    try:
        operation_count, leaves = _inspect_expression(parsed.body)
        normalized = _normalized_expression(parsed.body)
        if operation_count != 3:
            raise _VerificationFailure("operation_count")
        if Counter(leaves) != Counter(task.inputs):
            raise _VerificationFailure("input_multiset")
        exact_value = _evaluate_expression(parsed.body)
        if exact_value.denominator != 1:
            raise _VerificationFailure("noninteger_result")
        value = exact_value.numerator
        if value != task.target:
            return _verification_result(
                passed=False,
                failure_code="wrong_target",
                normalized_expression=normalized,
                value=value,
                operation_count=operation_count,
            )
    except _VerificationFailure as failure:
        return _verification_result(
            passed=False,
            failure_code=failure.code,
            normalized_expression=normalized,
            value=None,
            operation_count=operation_count,
        )
    except (MemoryError, OverflowError, RecursionError):
        return _verification_result(
            passed=False,
            failure_code="resource_limit",
            normalized_expression=normalized,
            value=None,
            operation_count=operation_count,
        )

    return _verification_result(
        passed=True,
        failure_code="ok",
        normalized_expression=normalized,
        value=value,
        operation_count=operation_count,
    )


def _ordered_terms(terms: Iterable[_Term]) -> tuple[_Term, ...]:
    return tuple(sorted(terms, key=lambda term: (term.value, term.expression)))


def _candidate_terms(left: _Term, right: _Term) -> tuple[_Term, ...]:
    """Enumerate unique legal pair reductions in deterministic order."""

    candidates: dict[str, _Term] = {}
    commutative_left, commutative_right = sorted(
        (left, right), key=lambda term: (term.expression, term.value)
    )
    for symbol, value in (
        ("+", left.value + right.value),
        ("*", left.value * right.value),
    ):
        expression = (
            f"({commutative_left.expression}{symbol}"
            f"{commutative_right.expression})"
        )
        candidates[expression] = _Term(value=value, expression=expression)

    for minuend, subtrahend in ((left, right), (right, left)):
        if minuend.value >= subtrahend.value:
            expression = f"({minuend.expression}-{subtrahend.expression})"
            candidates[expression] = _Term(
                value=minuend.value - subtrahend.value,
                expression=expression,
            )
    for numerator, denominator in ((left, right), (right, left)):
        if denominator.value != 0 and numerator.value % denominator.value == 0:
            expression = f"({numerator.expression}/{denominator.expression})"
            candidates[expression] = _Term(
                value=numerator.value // denominator.value,
                expression=expression,
            )
    return tuple(candidates[key] for key in sorted(candidates))


def _solutions(inputs: tuple[int, ...], target: int) -> tuple[str, ...]:
    found: set[str] = set()
    visited: set[tuple[tuple[int, str], ...]] = set()

    def search(terms: tuple[_Term, ...]) -> None:
        signature = tuple((term.value, term.expression) for term in terms)
        if signature in visited:
            return
        visited.add(signature)
        if len(terms) == 1:
            if terms[0].value == target:
                found.add(terms[0].expression)
            return
        for left_index in range(len(terms) - 1):
            for right_index in range(left_index + 1, len(terms)):
                remainder = [
                    term
                    for index, term in enumerate(terms)
                    if index not in (left_index, right_index)
                ]
                for combined in _candidate_terms(
                    terms[left_index], terms[right_index]
                ):
                    search(_ordered_terms((*remainder, combined)))

    search(
        _ordered_terms(
            _Term(value=value, expression=str(value)) for value in inputs
        )
    )
    return tuple(sorted(found))


def generate_tasks(
    seed: int,
    count: int,
    exclude: set[str] | None = None,
) -> list[CountdownTask]:
    """Generate a deterministic list of solvable, nontrivial Countdown tasks.

    Inputs are sampled from 1..99 and targets from 10..100.  A problem is
    considered nontrivial when its target is not already one of its inputs.
    ``solution_count`` counts distinct normalized legal expression trees under
    the same rules enforced by :func:`verify_expression`; the lexicographically
    first expression is retained as a deterministic witness.
    """

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("count must be a nonnegative integer")
    excluded = set() if exclude is None else set(exclude)
    if any(not isinstance(value, str) for value in excluded):
        raise TypeError("exclude must contain task-hash strings")

    generator = random.Random(seed)
    tasks: list[CountdownTask] = []
    seen = set(excluded)
    attempts = 0
    maximum_attempts = max(1_000, count * 2_000)
    while len(tasks) < count and attempts < maximum_attempts:
        attempts += 1
        inputs = tuple(sorted(generator.randint(1, 99) for _ in range(4)))
        target = generator.randint(10, 100)
        if target in inputs:
            continue
        provisional = CountdownTask(inputs, target, "", 0)
        problem_hash = task_hash(provisional)
        if problem_hash in seen:
            continue
        solutions = _solutions(inputs, target)
        if not solutions:
            continue
        task = CountdownTask(
            inputs=inputs,
            target=target,
            witness=solutions[0],
            solution_count=len(solutions),
        )
        witness_result = verify_expression(task.witness, task)
        if not witness_result["passed"]:
            raise AssertionError(
                "generated witness failed strict verification: "
                f"{witness_result['failure_code']}"
            )
        tasks.append(task)
        seen.add(problem_hash)

    if len(tasks) != count:
        raise RuntimeError(
            f"generated only {len(tasks)} of {count} tasks after {attempts} attempts"
        )
    return tasks
