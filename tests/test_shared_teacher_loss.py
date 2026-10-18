from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

import torch

from nextlat.core import (
    NextLatDynamics,
    NextLatLossConfig,
    compute_nextlat_losses,
    nextlat_position_masks,
)
from nextlat.shared_teacher_loss import (
    _HorizonStackedStudentKLFromTeacherLogProbs,
    _HorizonStackedStudentKLTriton,
    _StudentKLFromTeacherLogProbs,
    _compact_teacher_indices_d4,
    _compact_generated_teacher_indices_d4,
    compute_nextlat_losses_shared_teacher_d4,
    compute_shared_teacher_ce_kl_d4,
)
from nextlat.triton_horizon_kl import (
    triton_runtime_available,
    triton_tokenwise_kl_and_probability_delta,
)


def _roll_d4(
    dynamics: NextLatDynamics,
    hidden: torch.Tensor,
    embeddings: torch.Tensor,
) -> list[torch.Tensor]:
    recurrent = hidden
    result = []
    for step in range(1, 5):
        recurrent = dynamics(recurrent[:, :-1], embeddings[:, step:])
        result.append(recurrent)
    return result


class SharedTeacherLossTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(913)
        self.batch = 2
        self.length = 10
        self.width = 8
        self.vocab = 23
        self.ids = torch.randint(0, self.vocab, (self.batch, self.length))
        self.attention = torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            ],
            dtype=torch.bool,
        )
        self.assistant = torch.tensor(
            [
                [0, 0, 1, 1, 0, 1, 1, 1, 1, 1],
                [0, 1, 1, 0, 1, 1, 1, 1, 0, 0],
            ],
            dtype=torch.bool,
        )

    def test_index_map_is_exact_shifted_ce_row(self) -> None:
        ce_mask, _, kl_masks = nextlat_position_masks(
            self.attention, self.assistant, horizon=4
        )
        mappings = _compact_teacher_indices_d4(ce_mask, kl_masks)
        fast_mappings = _compact_generated_teacher_indices_d4(ce_mask, kl_masks)
        compact_coordinates = ce_mask.nonzero(as_tuple=False)

        for step, (kl_mask, mapping, fast_mapping) in enumerate(
            zip(kl_masks, mappings, fast_mappings), start=1
        ):
            torch.testing.assert_close(fast_mapping, mapping)
            # This is the formal mask-subset premise used by the schedule.
            self.assertFalse(bool((kl_mask & ~ce_mask[:, step:]).any()))
            local_coordinates = kl_mask.nonzero(as_tuple=False)
            expected_coordinates = local_coordinates.clone()
            expected_coordinates[:, 1] += step
            actual_coordinates = compact_coordinates[mapping]
            torch.testing.assert_close(actual_coordinates, expected_coordinates)

    def test_scalar_counts_and_full_first_order_gradients_match_authority(self) -> None:
        for chunk_tokens in (1, 3, 7, 64):
            with self.subTest(chunk_tokens=chunk_tokens):
                torch.manual_seed(914)
                hidden_template = torch.randn(
                    self.batch, self.length, self.width, dtype=torch.float64
                )
                embedding_template = torch.randn_like(hidden_template)
                head_template = torch.randn(
                    self.vocab, self.width, dtype=torch.float64
                )
                authority_dynamics = NextLatDynamics(
                    self.width, projection_factor=8.0
                ).double()
                candidate_dynamics = copy.deepcopy(authority_dynamics)

                authority_hidden = hidden_template.clone().requires_grad_(True)
                authority_embeddings = (
                    embedding_template.clone().requires_grad_(True)
                )
                authority_head = head_template.clone().requires_grad_(True)
                candidate_hidden = hidden_template.clone().requires_grad_(True)
                candidate_embeddings = (
                    embedding_template.clone().requires_grad_(True)
                )
                candidate_head = head_template.clone().requires_grad_(True)

                config = NextLatLossConfig(
                    horizon=4,
                    vocab_chunk_tokens=chunk_tokens,
                    lambda_h=0.0,
                    lambda_kl=0.37,
                )
                authority = compute_nextlat_losses(
                    hidden_states=authority_hidden,
                    token_embeddings=authority_embeddings,
                    input_ids=self.ids,
                    attention_mask=self.attention,
                    assistant_loss_mask=self.assistant,
                    head_weight=authority_head,
                    dynamics=authority_dynamics,
                    config=config,
                )
                candidate_predictions = _roll_d4(
                    candidate_dynamics, candidate_hidden, candidate_embeddings
                )
                candidate = compute_shared_teacher_ce_kl_d4(
                    hidden_states=candidate_hidden,
                    input_ids=self.ids,
                    attention_mask=self.attention,
                    assistant_loss_mask=self.assistant,
                    head_weight=candidate_head,
                    predicted_states_by_horizon=candidate_predictions,
                    chunk_tokens=chunk_tokens,
                )

                torch.testing.assert_close(candidate["ce"], authority["ce"])
                torch.testing.assert_close(
                    candidate["kl_by_horizon"], authority["kl_by_horizon"]
                )
                torch.testing.assert_close(candidate["kl"], authority["kl"])
                torch.testing.assert_close(
                    candidate["ce_positions"], authority["ce_positions"]
                )
                torch.testing.assert_close(
                    candidate["kl_positions_by_horizon"],
                    authority["kl_positions_by_horizon"],
                )

                authority_objective = authority["ce"] + 0.37 * authority["kl"]
                candidate_objective = candidate["ce"] + 0.37 * candidate["kl"]
                authority_parameters = [
                    authority_hidden,
                    authority_embeddings,
                    authority_head,
                    *authority_dynamics.parameters(),
                ]
                candidate_parameters = [
                    candidate_hidden,
                    candidate_embeddings,
                    candidate_head,
                    *candidate_dynamics.parameters(),
                ]
                authority_gradients = torch.autograd.grad(
                    authority_objective, authority_parameters
                )
                candidate_gradients = torch.autograd.grad(
                    candidate_objective, candidate_parameters
                )
                for candidate_gradient, authority_gradient in zip(
                    candidate_gradients, authority_gradients
                ):
                    # The fused schedule changes only floating-point reduction
                    # grouping: teacher rows are grouped by CE chunks rather
                    # than independently by horizon.  The first-order formula
                    # is identical; on this FP32-log-softmax authority path the
                    # observed worst relative L2 error is below 4e-8.
                    relative_l2 = (
                        (candidate_gradient - authority_gradient).norm()
                        / authority_gradient.norm().clamp_min(1e-30)
                    )
                    self.assertLessEqual(float(relative_l2), 1e-7)
                    torch.testing.assert_close(
                        candidate_gradient,
                        authority_gradient,
                        rtol=3e-6,
                        atol=6e-9,
                    )

    def test_full_core_dispatch_matches_authority_dictionary_and_gradients(self) -> None:
        torch.manual_seed(915)
        hidden_template = torch.randn(
            self.batch, self.length, self.width, dtype=torch.float64
        )
        embedding_template = torch.randn_like(hidden_template)
        head_template = torch.randn(self.vocab, self.width, dtype=torch.float64)
        authority_dynamics = NextLatDynamics(
            self.width, projection_factor=8.0
        ).double()
        candidate_dynamics = copy.deepcopy(authority_dynamics)

        authority_hidden = hidden_template.clone().requires_grad_(True)
        authority_embeddings = embedding_template.clone().requires_grad_(True)
        authority_head = head_template.clone().requires_grad_(True)
        candidate_hidden = hidden_template.clone().requires_grad_(True)
        candidate_embeddings = embedding_template.clone().requires_grad_(True)
        candidate_head = head_template.clone().requires_grad_(True)

        authority_config = NextLatLossConfig(
            horizon=4,
            vocab_chunk_tokens=3,
            lambda_h=0.61,
            lambda_kl=0.37,
            kl_implementation="autograd",
        )
        candidate_config = NextLatLossConfig(
            horizon=4,
            vocab_chunk_tokens=3,
            lambda_h=0.61,
            lambda_kl=0.37,
            kl_implementation="shared_teacher_forward_gradient",
        )
        authority = compute_nextlat_losses(
            hidden_states=authority_hidden,
            token_embeddings=authority_embeddings,
            input_ids=self.ids,
            attention_mask=self.attention,
            assistant_loss_mask=self.assistant,
            head_weight=authority_head,
            dynamics=authority_dynamics,
            config=authority_config,
        )
        # Calling the public authority entry point proves the new core dispatch,
        # while this direct symbol check keeps the integrated implementation a
        # first-class test target rather than dead prototype code.
        self.assertTrue(callable(compute_nextlat_losses_shared_teacher_d4))
        candidate = compute_nextlat_losses(
            hidden_states=candidate_hidden,
            token_embeddings=candidate_embeddings,
            input_ids=self.ids,
            attention_mask=self.attention,
            assistant_loss_mask=self.assistant,
            head_weight=candidate_head,
            dynamics=candidate_dynamics,
            config=candidate_config,
        )

        self.assertEqual(set(candidate), set(authority))
        count_keys = {
            "ce_positions",
            "transition_positions",
            "kl_positions",
            "transition_positions_by_horizon",
            "kl_positions_by_horizon",
        }
        list_keys = {
            "predicted_states_by_horizon",
        }
        for key in authority:
            with self.subTest(key=key):
                if key in list_keys:
                    self.assertEqual(len(candidate[key]), len(authority[key]))
                    for actual, expected in zip(candidate[key], authority[key]):
                        torch.testing.assert_close(actual, expected)
                elif key in count_keys:
                    torch.testing.assert_close(
                        candidate[key], authority[key], rtol=0.0, atol=0.0
                    )
                else:
                    torch.testing.assert_close(
                        candidate[key], authority[key], rtol=3e-6, atol=1e-6
                    )

        authority_parameters = [
            authority_hidden,
            authority_embeddings,
            authority_head,
            *authority_dynamics.parameters(),
        ]
        candidate_parameters = [
            candidate_hidden,
            candidate_embeddings,
            candidate_head,
            *candidate_dynamics.parameters(),
        ]
        authority_gradients = torch.autograd.grad(
            authority["loss"], authority_parameters
        )
        candidate_gradients = torch.autograd.grad(
            candidate["loss"], candidate_parameters
        )
        for actual, expected in zip(candidate_gradients, authority_gradients):
            relative_l2 = (
                (actual - expected).norm() / expected.norm().clamp_min(1e-30)
            )
            self.assertLessEqual(float(relative_l2), 1e-7)
            torch.testing.assert_close(
                actual, expected, rtol=3e-6, atol=6e-9
            )

    def test_shared_backend_is_opt_in_and_d4_only(self) -> None:
        self.assertEqual(NextLatLossConfig().kl_implementation, "autograd")
        hidden = torch.randn(1, 8, self.width)
        embeddings = torch.randn_like(hidden)
        head = torch.randn(self.vocab, self.width)
        dynamics = NextLatDynamics(self.width, projection_factor=8.0)
        for implementation in (
            "shared_teacher_forward_gradient",
            "shared_teacher_horizon_stacked",
            "shared_teacher_horizon_stacked_triton",
        ):
            with (
                self.subTest(implementation=implementation),
                self.assertRaisesRegex(ValueError, "requires NextLat horizon=4"),
            ):
                compute_nextlat_losses(
                    hidden_states=hidden,
                    token_embeddings=embeddings,
                    input_ids=torch.randint(0, self.vocab, (1, 8)),
                    attention_mask=torch.ones(1, 8, dtype=torch.bool),
                    assistant_loss_mask=torch.ones(1, 8, dtype=torch.bool),
                    head_weight=head,
                    dynamics=dynamics,
                    config=NextLatLossConfig(
                        horizon=2,
                        kl_implementation=implementation,
                    ),
                )

    def test_full_backend_constructs_masks_once_without_tensor_scalar_reads(self) -> None:
        hidden = torch.randn(
            self.batch, self.length, self.width, requires_grad=True
        )
        embeddings = torch.randn_like(hidden, requires_grad=True)
        head = torch.randn(self.vocab, self.width, requires_grad=True)
        dynamics = NextLatDynamics(self.width, projection_factor=8.0)
        mask_calls = []
        original_masks = nextlat_position_masks

        def counted_masks(*args, **kwargs):
            mask_calls.append((args, kwargs))
            return original_masks(*args, **kwargs)

        # Tensor.item()/Tensor.__bool__ are CUDA synchronization points when
        # their input is on device.  The shared backend's trusted-mask hot path
        # must not use either; shape-derived Python integers remain safe.
        with (
            patch(
                "nextlat.shared_teacher_loss.nextlat_position_masks",
                side_effect=counted_masks,
            ),
            patch.object(
                torch.Tensor,
                "item",
                side_effect=AssertionError("unexpected tensor scalar read"),
            ),
            patch.object(
                torch.Tensor,
                "__bool__",
                side_effect=AssertionError("unexpected tensor truth test"),
            ),
        ):
            result = compute_nextlat_losses(
                hidden_states=hidden,
                token_embeddings=embeddings,
                input_ids=self.ids,
                attention_mask=self.attention,
                assistant_loss_mask=self.assistant,
                head_weight=head,
                dynamics=dynamics,
                config=NextLatLossConfig(
                    horizon=4,
                    vocab_chunk_tokens=3,
                    kl_implementation="shared_teacher_forward_gradient",
                ),
            )
        self.assertEqual(len(mask_calls), 1)
        self.assertEqual(
            set(result),
            {
                "loss",
                "ce",
                "smooth_l1",
                "kl",
                "predicted_next_states",
                "predicted_states_by_horizon",
                "smooth_l1_by_horizon",
                "kl_by_horizon",
                "transition_positions",
                "kl_positions",
                "transition_positions_by_horizon",
                "kl_positions_by_horizon",
                "ce_positions",
            },
        )

    def test_kl_does_not_update_exact_teacher_or_head(self) -> None:
        hidden = torch.randn(
            self.batch, self.length, self.width, requires_grad=True
        )
        embeddings = torch.randn_like(hidden, requires_grad=True)
        head = torch.randn(self.vocab, self.width, requires_grad=True)
        dynamics = NextLatDynamics(self.width, projection_factor=8.0)
        predicted = _roll_d4(dynamics, hidden, embeddings)
        result = compute_shared_teacher_ce_kl_d4(
            hidden_states=hidden,
            input_ids=self.ids,
            attention_mask=self.attention,
            assistant_loss_mask=self.assistant,
            head_weight=head,
            predicted_states_by_horizon=predicted,
            chunk_tokens=3,
        )
        hidden_grad, embedding_grad, head_grad = torch.autograd.grad(
            result["kl"], [hidden, embeddings, head], allow_unused=True
        )
        # Hidden/embedding gradients here come only through the connected
        # student recurrence.  The detached teacher/head path is checked by
        # replacing students with independent leaves below.
        self.assertGreater(float(hidden_grad.abs().sum()), 0.0)
        self.assertGreater(float(embedding_grad.abs().sum()), 0.0)
        self.assertIsNone(head_grad)

        independent_students = [
            tensor.detach().clone().requires_grad_(True) for tensor in predicted
        ]
        detached_path = compute_shared_teacher_ce_kl_d4(
            hidden_states=hidden,
            input_ids=self.ids,
            attention_mask=self.attention,
            assistant_loss_mask=self.assistant,
            head_weight=head,
            predicted_states_by_horizon=independent_students,
            chunk_tokens=3,
        )
        exact_grad, detached_head_grad = torch.autograd.grad(
            detached_path["kl"], [hidden, head], allow_unused=True
        )
        self.assertIsNone(exact_grad)
        self.assertIsNone(detached_head_grad)

    def test_student_kl_retains_only_hidden_width_tensor(self) -> None:
        teacher_log_probs = torch.log_softmax(torch.randn(5, 17), dim=-1)
        student = torch.randn(5, 7, requires_grad=True)
        head = torch.randn(17, 7)
        saved = []

        def pack(tensor: torch.Tensor) -> torch.Tensor:
            saved.append(tuple(tensor.shape))
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            value = _StudentKLFromTeacherLogProbs.apply(
                teacher_log_probs.detach(), student, head.detach()
            )
            value.backward()
        self.assertEqual(saved, [(5, 7)])

    def test_horizon_stacked_tokenwise_values_and_scaled_gradients(self) -> None:
        torch.manual_seed(916)
        horizon_positions = (2, 1, 4, 3)
        rows = sum(horizon_positions)
        teacher_log_probs = torch.log_softmax(
            torch.randn(rows, 19), dim=-1
        ).detach()
        student_template = torch.randn(rows, 7)
        head = torch.randn(19, 7, requires_grad=True)
        candidate_student = student_template.clone().requires_grad_(True)
        authority_student = student_template.clone().requires_grad_(True)

        candidate_values = _HorizonStackedStudentKLFromTeacherLogProbs.apply(
            teacher_log_probs,
            candidate_student,
            head.detach(),
        )
        authority_log_probs = torch.log_softmax(
            torch.nn.functional.linear(authority_student, head.detach()).float(),
            dim=-1,
        )
        authority_values = torch.nn.functional.kl_div(
            authority_log_probs,
            teacher_log_probs,
            log_target=True,
            reduction="none",
        ).sum(dim=-1)
        torch.testing.assert_close(candidate_values, authority_values)

        horizon_scales = torch.tensor([0.7, -0.2, 1.3, 0.4])
        row_scales = torch.cat(
            [
                torch.full((positions,), horizon_scales[index])
                for index, positions in enumerate(horizon_positions)
            ]
        )
        candidate_values.backward(row_scales)
        authority_values.backward(row_scales)
        torch.testing.assert_close(
            candidate_student.grad,
            authority_student.grad,
            rtol=2e-6,
            atol=2e-7,
        )
        self.assertIsNone(head.grad)

    def test_horizon_stacked_retains_only_hidden_width_tensor(self) -> None:
        teacher_log_probs = torch.log_softmax(torch.randn(9, 17), dim=-1)
        student = torch.randn(9, 7, requires_grad=True)
        head = torch.randn(17, 7)
        saved = []

        def pack(tensor: torch.Tensor) -> torch.Tensor:
            saved.append(tuple(tensor.shape))
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            values = _HorizonStackedStudentKLFromTeacherLogProbs.apply(
                teacher_log_probs.detach(), student, head.detach()
            )
            values.sum().backward()
        self.assertEqual(saved, [(9, 7)])

    def test_triton_horizon_backend_fails_closed_without_cuda(self) -> None:
        if torch.cuda.is_available() and triton_runtime_available():
            self.skipTest("CUDA Triton runtime is available")
        teacher = torch.log_softmax(torch.randn(3, 17), dim=-1)
        student = torch.log_softmax(torch.randn(5, 17), dim=-1)
        rows = torch.tensor([0, 1, 2, 0, 1])
        with self.assertRaisesRegex(RuntimeError, "requires (Triton|CUDA)"):
            triton_tokenwise_kl_and_probability_delta(
                student_log_probs=student,
                teacher_log_probs=teacher,
                teacher_rows=rows,
                delta_dtype=torch.float32,
            )

    def test_triton_full_backend_never_falls_back_to_eager(self) -> None:
        if torch.cuda.is_available() and triton_runtime_available():
            self.skipTest("CUDA Triton runtime is available")
        hidden = torch.randn(
            self.batch, self.length, self.width, requires_grad=True
        )
        embeddings = torch.randn_like(hidden, requires_grad=True)
        head = torch.randn(self.vocab, self.width, requires_grad=True)
        dynamics = NextLatDynamics(self.width, projection_factor=8.0)
        with patch.object(
            _HorizonStackedStudentKLFromTeacherLogProbs,
            "apply",
            side_effect=AssertionError("eager fallback executed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "requires (Triton|CUDA)"):
                compute_nextlat_losses(
                    hidden_states=hidden,
                    token_embeddings=embeddings,
                    input_ids=self.ids,
                    attention_mask=self.attention,
                    assistant_loss_mask=self.assistant,
                    head_weight=head,
                    dynamics=dynamics,
                    config=NextLatLossConfig(
                        horizon=4,
                        vocab_chunk_tokens=3,
                        kl_implementation=(
                            "shared_teacher_horizon_stacked_triton"
                        ),
                    ),
                )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_triton_horizon_values_and_gradients_match_eager(self) -> None:
        if not triton_runtime_available():
            self.skipTest("Triton is unavailable")
        torch.manual_seed(918)
        rows, teacher_count, vocab, width = 11, 4, 4099, 32
        teacher_log_probs = torch.log_softmax(
            torch.randn(teacher_count, vocab, device="cuda"), dim=-1
        ).detach()
        teacher_rows = torch.tensor(
            [0, 1, 2, 3, 0, 2, 1, 3, 2, 0, 1],
            dtype=torch.long,
            device="cuda",
        )
        template = torch.randn(rows, width, device="cuda")
        candidate_student = template.clone().requires_grad_(True)
        authority_student = template.clone().requires_grad_(True)
        head = torch.randn(vocab, width, device="cuda")
        row_scales = torch.randn(rows, device="cuda")

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            candidate = _HorizonStackedStudentKLTriton.apply(
                teacher_log_probs,
                teacher_rows,
                candidate_student,
                head.detach(),
            )
            authority = _HorizonStackedStudentKLFromTeacherLogProbs.apply(
                teacher_log_probs.index_select(0, teacher_rows),
                authority_student,
                head.detach(),
            )
        candidate.backward(row_scales)
        authority.backward(row_scales)
        torch.testing.assert_close(candidate, authority, rtol=8e-5, atol=2e-5)
        torch.testing.assert_close(
            candidate_student.grad,
            authority_student.grad,
            rtol=8e-4,
            atol=3e-4,
        )

    def test_horizon_stacked_full_objective_and_unequal_masks_match(self) -> None:
        torch.manual_seed(917)
        hidden_template = torch.randn(
            self.batch, self.length, self.width, dtype=torch.float64
        )
        embedding_template = torch.randn_like(hidden_template)
        head_template = torch.randn(
            self.vocab, self.width, dtype=torch.float64
        )
        authority_dynamics = NextLatDynamics(
            self.width, projection_factor=8.0
        ).double()
        candidate_dynamics = copy.deepcopy(authority_dynamics)

        authority_parameters = [
            hidden_template.clone().requires_grad_(True),
            embedding_template.clone().requires_grad_(True),
            head_template.clone().requires_grad_(True),
            *authority_dynamics.parameters(),
        ]
        candidate_parameters = [
            hidden_template.clone().requires_grad_(True),
            embedding_template.clone().requires_grad_(True),
            head_template.clone().requires_grad_(True),
            *candidate_dynamics.parameters(),
        ]
        authority = compute_nextlat_losses(
            hidden_states=authority_parameters[0],
            token_embeddings=authority_parameters[1],
            input_ids=self.ids,
            attention_mask=self.attention,
            assistant_loss_mask=self.assistant,
            head_weight=authority_parameters[2],
            dynamics=authority_dynamics,
            config=NextLatLossConfig(
                horizon=4,
                vocab_chunk_tokens=3,
                lambda_h=0.61,
                lambda_kl=0.37,
            ),
        )
        candidate = compute_nextlat_losses(
            hidden_states=candidate_parameters[0],
            token_embeddings=candidate_parameters[1],
            input_ids=self.ids,
            attention_mask=self.attention,
            assistant_loss_mask=self.assistant,
            head_weight=candidate_parameters[2],
            dynamics=candidate_dynamics,
            config=NextLatLossConfig(
                horizon=4,
                vocab_chunk_tokens=3,
                lambda_h=0.61,
                lambda_kl=0.37,
                kl_implementation="shared_teacher_horizon_stacked",
            ),
        )
        self.assertGreater(
            torch.unique(candidate["kl_positions_by_horizon"]).numel(), 1
        )
        for name in (
            "loss",
            "ce",
            "smooth_l1",
            "kl",
            "smooth_l1_by_horizon",
            "kl_by_horizon",
            "ce_positions",
            "transition_positions_by_horizon",
            "kl_positions_by_horizon",
        ):
            torch.testing.assert_close(
                candidate[name], authority[name], rtol=3e-6, atol=1e-7
            )

        authority_gradients = torch.autograd.grad(
            authority["loss"], authority_parameters
        )
        candidate_gradients = torch.autograd.grad(
            candidate["loss"], candidate_parameters
        )
        for actual, expected in zip(candidate_gradients, authority_gradients):
            relative_l2 = (
                (actual - expected).norm() / expected.norm().clamp_min(1e-30)
            )
            self.assertLessEqual(float(relative_l2), 1e-7)
            torch.testing.assert_close(
                actual, expected, rtol=3e-6, atol=6e-9
            )

    def test_horizon_stacked_uses_one_student_projection_per_ce_chunk(self) -> None:
        hidden = torch.randn(
            self.batch, self.length, self.width, requires_grad=True
        )
        embeddings = torch.randn_like(hidden, requires_grad=True)
        head = torch.randn(self.vocab, self.width, requires_grad=True)
        dynamics = NextLatDynamics(self.width, projection_factor=8.0)
        projected_rows = []
        original_linear = torch.nn.functional.linear

        def record_linear(input_tensor, weight, bias=None):
            if weight.data_ptr() == head.data_ptr():
                projected_rows.append(input_tensor.shape[0])
            return original_linear(input_tensor, weight, bias)

        with patch(
            "nextlat.shared_teacher_loss.F.linear", side_effect=record_linear
        ):
            compute_nextlat_losses(
                hidden_states=hidden,
                token_embeddings=embeddings,
                input_ids=self.ids,
                attention_mask=self.attention,
                assistant_loss_mask=self.assistant,
                head_weight=head,
                dynamics=dynamics,
                config=NextLatLossConfig(
                    horizon=4,
                    vocab_chunk_tokens=3,
                    kl_implementation="shared_teacher_horizon_stacked",
                ),
            )
        ce_positions = int(
            nextlat_position_masks(
                self.attention, self.assistant, horizon=4
            )[0].sum()
        )
        ce_chunks = (ce_positions + 2) // 3
        self.assertEqual(len(projected_rows), 2 * ce_chunks)
        for exact_rows, student_rows in zip(
            projected_rows[0::2], projected_rows[1::2]
        ):
            self.assertLessEqual(exact_rows, 3)
            self.assertLessEqual(student_rows, 4 * exact_rows)


if __name__ == "__main__":
    unittest.main()
