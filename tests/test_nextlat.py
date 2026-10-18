import copy
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from nextlat.core import (
    NextLatDynamics,
    NextLatLossConfig,
    compute_nextlat_losses,
    nextlat_position_counts,
)


def pinned_official_d1_reference(
    hidden_states,
    token_embeddings,
    input_ids,
    attention_mask,
    assistant_mask,
    head_weight,
    dynamics,
    lambda_h,
    lambda_kl,
):
    """Independent, unchunked transcription of pinned upstream d=1 indexing."""

    ce_mask = (
        attention_mask[:, :-1].bool()
        & attention_mask[:, 1:].bool()
        & assistant_mask[:, 1:].bool()
    )
    logits = F.linear(hidden_states[:, :-1][ce_mask], head_weight)
    ce = F.cross_entropy(logits.float(), input_ids[:, 1:][ce_mask])

    predicted = dynamics(hidden_states[:, :-1], token_embeddings[:, 1:])
    transition_mask = attention_mask[:, :-1].bool() & attention_mask[:, 1:].bool()
    smooth_elements = F.smooth_l1_loss(
        predicted, hidden_states[:, 1:].detach(), reduction="none"
    )
    smooth = (
        smooth_elements * transition_mask.unsqueeze(-1)
    ).sum() / (transition_mask.sum() * hidden_states.shape[-1])

    kl_mask = (
        attention_mask[:, :-2].bool()
        & attention_mask[:, 1:-1].bool()
        & attention_mask[:, 2:].bool()
        & assistant_mask[:, 2:].bool()
    )
    teacher_logits = F.linear(
        hidden_states[:, 1:-1][kl_mask].detach(), head_weight.detach()
    )
    student_logits = F.linear(predicted[:, :-1][kl_mask], head_weight.detach())
    teacher_log_probs = F.log_softmax(teacher_logits.float(), dim=-1)
    student_log_probs = F.log_softmax(student_logits.float(), dim=-1)
    kl = F.kl_div(
        student_log_probs,
        teacher_log_probs,
        log_target=True,
        reduction="batchmean",
    )
    return {
        "ce": ce,
        "smooth_l1": smooth,
        "kl": kl,
        "loss": ce + lambda_h * smooth + lambda_kl * kl,
    }


def pinned_official_horizon_reference(
    hidden_states,
    token_embeddings,
    input_ids,
    attention_mask,
    assistant_mask,
    head_weight,
    dynamics,
    horizon,
    lambda_h,
    lambda_kl,
):
    """Independent transcription of the pinned recursive horizon loop."""

    ce_mask = (
        attention_mask[:, :-1].bool()
        & attention_mask[:, 1:].bool()
        & assistant_mask[:, 1:].bool()
    )
    ce = F.cross_entropy(
        F.linear(hidden_states[:, :-1][ce_mask], head_weight).float(),
        input_ids[:, 1:][ce_mask],
    )
    recurrent = hidden_states
    smooth_losses = []
    kl_losses = []
    length = hidden_states.shape[1]
    for step in range(1, horizon + 1):
        recurrent = dynamics(recurrent[:, :-1], token_embeddings[:, step:])
        transition_width = length - step
        transition_mask = torch.stack(
            [
                attention_mask[:, offset : offset + transition_width].bool()
                for offset in range(step + 1)
            ]
        ).all(dim=0)
        smooth_elements = F.smooth_l1_loss(
            recurrent, hidden_states[:, step:].detach(), reduction="none"
        )
        smooth_losses.append(
            (smooth_elements * transition_mask.unsqueeze(-1)).sum()
            / (transition_mask.sum() * hidden_states.shape[-1])
        )

        kl_width = length - step - 1
        kl_mask = torch.stack(
            [
                attention_mask[:, offset : offset + kl_width].bool()
                for offset in range(step + 2)
            ]
        ).all(dim=0) & assistant_mask[:, step + 1 :].bool()
        teacher_logits = F.linear(
            hidden_states[:, step:-1][kl_mask].detach(), head_weight.detach()
        )
        student_logits = F.linear(recurrent[:, :-1][kl_mask], head_weight.detach())
        kl_losses.append(
            F.kl_div(
                F.log_softmax(student_logits.float(), dim=-1),
                F.log_softmax(teacher_logits.float(), dim=-1),
                log_target=True,
                reduction="batchmean",
            )
        )

    smooth = torch.stack(smooth_losses).mean()
    kl = torch.stack(kl_losses).mean()
    return {
        "ce": ce,
        "smooth_l1": smooth,
        "kl": kl,
        "loss": ce + lambda_h * smooth + lambda_kl * kl,
        "smooth_l1_by_horizon": torch.stack(smooth_losses),
        "kl_by_horizon": torch.stack(kl_losses),
    }


class NextLatParityTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)
        self.batch = 2
        self.length = 7
        self.hidden_size = 64
        self.vocab_size = 19
        self.hidden = torch.randn(
            self.batch, self.length, self.hidden_size, requires_grad=True
        )
        self.embeddings = torch.randn(
            self.batch, self.length, self.hidden_size, requires_grad=True
        )
        self.ids = torch.randint(0, self.vocab_size, (self.batch, self.length))
        self.attention = torch.tensor(
            [[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0, 0]], dtype=torch.bool
        )
        self.assistant = torch.tensor(
            [[0, 0, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 0, 0]], dtype=torch.bool
        )
        self.head = torch.randn(
            self.vocab_size, self.hidden_size, requires_grad=True
        )
        self.dynamics = NextLatDynamics(self.hidden_size, projection_factor=1.0)

    def test_matches_pinned_official_d1_reference(self):
        config = NextLatLossConfig(
            lambda_h=0.7, lambda_kl=1.3, vocab_chunk_tokens=2
        )
        actual = compute_nextlat_losses(
            hidden_states=self.hidden,
            token_embeddings=self.embeddings,
            input_ids=self.ids,
            attention_mask=self.attention,
            assistant_loss_mask=self.assistant,
            head_weight=self.head,
            dynamics=self.dynamics,
            config=config,
        )
        expected = pinned_official_d1_reference(
            self.hidden,
            self.embeddings,
            self.ids,
            self.attention,
            self.assistant,
            self.head,
            self.dynamics,
            config.lambda_h,
            config.lambda_kl,
        )
        for name in ("ce", "smooth_l1", "kl", "loss"):
            torch.testing.assert_close(actual[name], expected[name], rtol=1e-6, atol=1e-6)

    def test_d4_matches_pinned_official_recursive_reference(self):
        config = NextLatLossConfig(
            lambda_h=0.7,
            lambda_kl=1.3,
            vocab_chunk_tokens=2,
            horizon=4,
        )
        actual = compute_nextlat_losses(
            hidden_states=self.hidden,
            token_embeddings=self.embeddings,
            input_ids=self.ids,
            attention_mask=self.attention,
            assistant_loss_mask=self.assistant,
            head_weight=self.head,
            dynamics=self.dynamics,
            config=config,
        )
        expected = pinned_official_horizon_reference(
            self.hidden,
            self.embeddings,
            self.ids,
            self.attention,
            self.assistant,
            self.head,
            self.dynamics,
            config.horizon,
            config.lambda_h,
            config.lambda_kl,
        )
        for name in (
            "ce",
            "smooth_l1",
            "kl",
            "loss",
            "smooth_l1_by_horizon",
            "kl_by_horizon",
        ):
            torch.testing.assert_close(actual[name], expected[name], rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(
            actual["transition_positions_by_horizon"], torch.tensor([10, 8, 6, 4])
        )
        torch.testing.assert_close(
            actual["kl_positions_by_horizon"], torch.tensor([8, 6, 4, 2])
        )

    def _assert_count_weighted_microbatches_match_full_batch_gradient(
        self, device: torch.device
    ):
        config = NextLatLossConfig(
            lambda_h=0.7,
            lambda_kl=1.3,
            vocab_chunk_tokens=2,
            horizon=4,
        )
        ids = self.ids.to(device)
        full_hidden = self.hidden.detach().clone().to(device).requires_grad_(True)
        full_embeddings = (
            self.embeddings.detach().clone().to(device).requires_grad_(True)
        )
        full_head = self.head.detach().clone().to(device).requires_grad_(True)
        full_dynamics = copy.deepcopy(self.dynamics).to(device)
        attention = torch.ones_like(self.attention).to(device)
        assistant = torch.tensor(
            [[0, 0, 1, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1]],
            dtype=torch.bool,
            device=device,
        )
        full = compute_nextlat_losses(
            hidden_states=full_hidden,
            token_embeddings=full_embeddings,
            input_ids=ids,
            attention_mask=attention,
            assistant_loss_mask=assistant,
            head_weight=full_head,
            dynamics=full_dynamics,
            config=config,
        )
        full["loss"].backward()

        accum_hidden = self.hidden.detach().clone().to(device).requires_grad_(True)
        accum_embeddings = (
            self.embeddings.detach().clone().to(device).requires_grad_(True)
        )
        accum_head = self.head.detach().clone().to(device).requires_grad_(True)
        accum_dynamics = copy.deepcopy(self.dynamics).to(device)
        total_counts = nextlat_position_counts(
            attention, assistant, config.horizon
        ).to(dtype=torch.float32)
        for row in range(self.batch):
            micro = compute_nextlat_losses(
                hidden_states=accum_hidden[row : row + 1],
                token_embeddings=accum_embeddings[row : row + 1],
                input_ids=ids[row : row + 1],
                attention_mask=attention[row : row + 1],
                assistant_loss_mask=assistant[row : row + 1],
                head_weight=accum_head,
                dynamics=accum_dynamics,
                config=config,
            )
            counts = nextlat_position_counts(
                attention[row : row + 1],
                assistant[row : row + 1],
                config.horizon,
            ).to(dtype=torch.float32)
            scales = counts / total_counts
            loss = (
                micro["ce"] * scales[0]
                + config.lambda_h
                * (micro["smooth_l1_by_horizon"] * scales[1:5]).mean()
                + config.lambda_kl
                * (micro["kl_by_horizon"] * scales[5:]).mean()
            )
            loss.backward()

        torch.testing.assert_close(accum_hidden.grad, full_hidden.grad)
        torch.testing.assert_close(accum_embeddings.grad, full_embeddings.grad)
        torch.testing.assert_close(accum_head.grad, full_head.grad)
        for accum_parameter, full_parameter in zip(
            accum_dynamics.parameters(), full_dynamics.parameters()
        ):
            torch.testing.assert_close(accum_parameter.grad, full_parameter.grad)

    def test_count_weighted_microbatches_match_full_batch_gradient(self):
        self._assert_count_weighted_microbatches_match_full_batch_gradient(
            torch.device("cpu")
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_count_weighted_microbatches_match_full_batch_gradient_on_cuda(self):
        self._assert_count_weighted_microbatches_match_full_batch_gradient(
            torch.device("cuda", 0)
        )

    def test_d4_joint_backward_reaches_theta_and_psi(self):
        losses = compute_nextlat_losses(
            hidden_states=self.hidden,
            token_embeddings=self.embeddings,
            input_ids=self.ids,
            attention_mask=self.attention,
            assistant_loss_mask=self.assistant,
            head_weight=self.head,
            dynamics=self.dynamics,
            config=NextLatLossConfig(vocab_chunk_tokens=3, horizon=4),
        )
        losses["loss"].backward()
        theta_grad = self.hidden.grad.abs().sum() + self.embeddings.grad.abs().sum()
        psi_grad = sum(
            parameter.grad.abs().sum().item()
            for parameter in self.dynamics.parameters()
            if parameter.grad is not None
        )
        self.assertTrue(torch.isfinite(theta_grad))
        self.assertGreater(theta_grad.item(), 0.0)
        self.assertGreater(psi_grad, 0.0)

    def test_auxiliary_stop_gradients(self):
        config = NextLatLossConfig(vocab_chunk_tokens=3)
        losses = compute_nextlat_losses(
            hidden_states=self.hidden,
            token_embeddings=self.embeddings,
            input_ids=self.ids,
            attention_mask=self.attention,
            assistant_loss_mask=self.assistant,
            head_weight=self.head,
            dynamics=self.dynamics,
            config=config,
        )
        auxiliary = losses["smooth_l1"] + losses["kl"]
        auxiliary.backward()

        self.assertIsNone(self.head.grad, "auxiliary path updated detached output head")
        self.assertGreater(self.embeddings.grad.abs().sum().item(), 0.0)
        self.assertGreater(self.hidden.grad[:, 0].abs().sum().item(), 0.0)
        # The final state is only a SmoothL1 target at d=1 and must be detached.
        self.assertEqual(self.hidden.grad[:, -1].abs().sum().item(), 0.0)
        psi_grad = sum(
            parameter.grad.abs().sum().item()
            for parameter in self.dynamics.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(psi_grad, 0.0)

    def test_joint_loss_reaches_theta_and_psi(self):
        losses = compute_nextlat_losses(
            hidden_states=self.hidden,
            token_embeddings=self.embeddings,
            input_ids=self.ids,
            attention_mask=self.attention,
            assistant_loss_mask=self.assistant,
            head_weight=self.head,
            dynamics=self.dynamics,
            config=NextLatLossConfig(vocab_chunk_tokens=4),
        )
        losses["loss"].backward()
        theta_grad = self.hidden.grad.abs().sum() + self.embeddings.grad.abs().sum()
        psi_grad = sum(
            parameter.grad.abs().sum()
            for parameter in self.dynamics.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(theta_grad.item(), 0.0)
        self.assertGreater(psi_grad.item(), 0.0)

    def test_tied_embedding_role_receives_auxiliary_gradient(self):
        shared = torch.nn.Parameter(torch.randn(self.vocab_size, self.hidden_size))
        ids = self.ids.detach()
        token_embeddings = F.embedding(ids, shared)
        hidden = self.hidden.detach().clone().requires_grad_(True)
        losses = compute_nextlat_losses(
            hidden_states=hidden,
            token_embeddings=token_embeddings,
            input_ids=ids,
            attention_mask=self.attention,
            assistant_loss_mask=self.assistant,
            head_weight=shared,
            dynamics=self.dynamics,
            config=NextLatLossConfig(vocab_chunk_tokens=4),
        )
        (losses["smooth_l1"] + losses["kl"]).backward()
        self.assertIsNotNone(shared.grad)
        self.assertGreater(shared.grad.abs().sum().item(), 0.0)

        # The detached KL head cannot be the source of this gradient: detaching
        # the embedding role as well removes every auxiliary path to the weight.
        self.dynamics.zero_grad(set_to_none=True)
        shared.grad = None
        hidden = self.hidden.detach().clone().requires_grad_(True)
        detached_embeddings = F.embedding(ids, shared).detach().requires_grad_(True)
        detached_losses = compute_nextlat_losses(
            hidden_states=hidden,
            token_embeddings=detached_embeddings,
            input_ids=ids,
            attention_mask=self.attention,
            assistant_loss_mask=self.assistant,
            head_weight=shared,
            dynamics=self.dynamics,
            config=NextLatLossConfig(vocab_chunk_tokens=4),
        )
        (detached_losses["smooth_l1"] + detached_losses["kl"]).backward()
        self.assertIsNone(shared.grad)

    def test_masks_match_expected_positions(self):
        losses = compute_nextlat_losses(
            hidden_states=self.hidden,
            token_embeddings=self.embeddings,
            input_ids=self.ids,
            attention_mask=self.attention,
            assistant_loss_mask=self.assistant,
            head_weight=self.head,
            dynamics=self.dynamics,
            config=NextLatLossConfig(vocab_chunk_tokens=4),
        )
        self.assertEqual(losses["ce_positions"].item(), 9)
        self.assertEqual(losses["transition_positions"].item(), 10)
        self.assertEqual(losses["kl_positions"].item(), 8)

    def test_dynamics_checkpoint_roundtrip(self):
        original = copy.deepcopy(self.dynamics).eval()
        current = torch.randn(3, self.hidden_size)
        token = torch.randn(3, self.hidden_size)
        expected = original(current, token)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dynamics.pt"
            torch.save(original.state_dict(), path)
            restored = NextLatDynamics(self.hidden_size, projection_factor=1.0).eval()
            restored.load_state_dict(torch.load(path, weights_only=True))
            actual = restored(current, token)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_qwen_projection_width(self):
        dynamics = NextLatDynamics(2048, projection_factor=1.6)
        self.assertEqual(dynamics.projection_size, 6528)


if __name__ == "__main__":
    unittest.main()
