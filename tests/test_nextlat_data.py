import copy
import unittest

from nextlat.data import (
    _encode,
    _problem_hash,
    _split_for,
    _verified_generation_indices,
    split_receipt,
)


class FakeTokenizer:
    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, enable_thinking
    ):
        del tokenize, enable_thinking
        prompt = [10, 11, 12]
        if add_generation_prompt:
            return prompt
        assistant = messages[-1]["content"]
        width = 12 if assistant == "too long" else 3
        return prompt + list(range(20, 20 + width))


class NextLatDataTests(unittest.TestCase):
    def test_all_verified_complete_candidates_are_retained_in_order(self):
        row = {
            "generations": ["too long", "fits", "incorrect"],
            "correctness_math_verify": [True, True, False],
            "is_reasoning_complete": [True, True, True],
        }
        self.assertEqual(_verified_generation_indices(row), [0, 1])
        selected = None
        for index in _verified_generation_indices(row):
            selected, _ = _encode(
                {"problem": "p", "uuid": "u", **row},
                index,
                FakeTokenizer(),
                sequence_length=8,
                problem_hash=_problem_hash("p"),
            )
            if selected is not None:
                break
        self.assertIsNotNone(selected)
        self.assertEqual(selected["generation_index"], 1)
        self.assertFalse(selected["was_truncated"])

    def test_problem_split_is_deterministic_from_normalized_text(self):
        left = _problem_hash("  a   problem\nwith space ")
        right = _problem_hash("a problem with space")
        self.assertEqual(left, right)
        self.assertEqual(_split_for(left, 1234), _split_for(right, 1234))

    def test_split_receipt_covers_serialized_token_fields(self):
        records = [
            {
                "uuid": "u",
                "problem_hash": "h",
                "source": "s",
                "generation_index": 0,
                "input_ids": [1, 2, 3],
                "assistant_loss_mask": [False, True, True],
                "token_count": 3,
                "assistant_token_count": 2,
                "was_truncated": False,
            }
        ]
        first = split_receipt(records)
        changed = copy.deepcopy(records)
        changed[0]["input_ids"][2] = 4
        second = split_receipt(changed)
        self.assertNotEqual(first["record_digest"], second["record_digest"])
        self.assertEqual(first["problem_hash_digest"], second["problem_hash_digest"])


if __name__ == "__main__":
    unittest.main()
