from __future__ import annotations

import unittest

import torch

from nextlat.core import (
    _TeacherStudentKLForwardGradient,
    _masked_teacher_student_kl,
    _masked_teacher_student_kl_forward_gradient,
)


class DenseKLForwardGradientTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(2701)
        self.teacher = torch.randn(2, 6, 13, requires_grad=True)
        self.student_template = torch.randn(2, 6, 13)
        self.head = torch.randn(37, 13, requires_grad=True)
        self.mask = torch.tensor(
            [
                [False, True, True, False, True, True],
                [True, False, True, True, False, True],
            ]
        )

    def _pair(self, chunk_tokens: int):
        authority_student = self.student_template.clone().requires_grad_(True)
        candidate_student = self.student_template.clone().requires_grad_(True)
        authority = _masked_teacher_student_kl(
            self.teacher, authority_student, self.mask, self.head, chunk_tokens
        )
        candidate = _masked_teacher_student_kl_forward_gradient(
            self.teacher, candidate_student, self.mask, self.head, chunk_tokens
        )
        return authority, candidate, authority_student, candidate_student

    def test_scalar_and_student_gradient_match_authority(self) -> None:
        for chunk_tokens in (1, 2, 3, 8, 64):
            with self.subTest(chunk_tokens=chunk_tokens):
                authority, candidate, authority_student, candidate_student = self._pair(
                    chunk_tokens
                )
                torch.testing.assert_close(candidate, authority, rtol=0.0, atol=0.0)
                scale = torch.tensor(-1.375)
                authority.backward(scale)
                candidate.backward(scale)
                torch.testing.assert_close(
                    candidate_student.grad,
                    authority_student.grad,
                    rtol=2e-6,
                    atol=2e-7,
                )
                self.assertEqual(
                    candidate_student.grad[~self.mask].abs().sum().item(), 0.0
                )
                self.teacher.grad = None
                self.head.grad = None

    def test_teacher_and_head_are_detached(self) -> None:
        _, candidate, _, candidate_student = self._pair(3)
        candidate.backward()
        self.assertIsNone(self.teacher.grad)
        self.assertIsNone(self.head.grad)
        self.assertGreater(candidate_student.grad.abs().sum().item(), 0.0)

    def test_only_hidden_width_gradient_is_saved(self) -> None:
        teacher = torch.randn(7, 11)
        student = torch.randn(7, 11, requires_grad=True)
        head = torch.randn(29, 11)
        saved = []

        def pack(tensor: torch.Tensor) -> torch.Tensor:
            saved.append((tuple(tensor.shape), tensor.dtype, tensor.requires_grad))
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            _TeacherStudentKLForwardGradient.apply(
                teacher.detach(), student, head.detach()
            ).backward()
        self.assertEqual(saved, [((7, 11), student.dtype, False)])

    def test_invalid_inputs_fail_closed(self) -> None:
        student = self.student_template.clone().requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "KL mask has zero valid positions"):
            _masked_teacher_student_kl_forward_gradient(
                self.teacher,
                student,
                torch.zeros_like(self.mask),
                self.head,
                2,
            )
        with self.assertRaisesRegex(ValueError, "chunk_tokens must be positive"):
            _masked_teacher_student_kl_forward_gradient(
                self.teacher, student, self.mask, self.head, 0
            )

    def test_higher_order_backward_fails_explicitly(self) -> None:
        student = self.student_template.clone().requires_grad_(True)
        loss = _masked_teacher_student_kl_forward_gradient(
            self.teacher, student, self.mask, self.head, 3
        )
        with self.assertRaisesRegex(RuntimeError, "first-order gradients only"):
            torch.autograd.grad(loss, student, create_graph=True)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA BF16 autocast")
    def test_cuda_bf16_autocast_matches_authority(self) -> None:
        torch.manual_seed(2702)
        teacher = torch.randn(2, 17, 256, device="cuda", dtype=torch.bfloat16)
        template = torch.randn(2, 17, 256, device="cuda", dtype=torch.bfloat16)
        head = torch.randn(1024, 256, device="cuda", dtype=torch.float32)
        mask = torch.rand(2, 17, device="cuda") > 0.3
        authority_student = template.clone().requires_grad_(True)
        candidate_student = template.clone().requires_grad_(True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            authority = _masked_teacher_student_kl(
                teacher, authority_student, mask, head, 7
            )
            candidate = _masked_teacher_student_kl_forward_gradient(
                teacher, candidate_student, mask, head, 7
            )
        authority.backward()
        candidate.backward()
        torch.testing.assert_close(candidate, authority, rtol=0.0, atol=0.0)
        actual = candidate_student.grad.float()
        reference = authority_student.grad.float()
        relative_l2 = (actual - reference).norm() / reference.norm().clamp_min(1e-12)
        cosine = torch.nn.functional.cosine_similarity(
            actual.flatten(), reference.flatten(), dim=0
        )
        self.assertLessEqual(relative_l2.item(), 0.005)
        self.assertGreaterEqual(cosine.item(), 0.9999)


if __name__ == "__main__":
    unittest.main()
