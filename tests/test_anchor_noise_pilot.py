"""Narrow checks for the only scientific change in the 800M noise pilot."""
import copy
import unittest

import torch

from nextlat.core import (
    NextLatDynamics, NextLatLossConfig, compute_nextlat_losses, noisy_training_anchor,
)


class AnchorNoisePilotTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        torch.set_num_threads(1)
        self.h = torch.randn(2, 11, 64, requires_grad=True)
        self.e = torch.randn_like(self.h, requires_grad=True)
        self.w = torch.randn(41, 64, requires_grad=True)
        self.ids = torch.randint(0, 41, (2, 11))
        self.mask = torch.ones_like(self.ids, dtype=torch.bool)
        self.d = NextLatDynamics(64, projection_factor=1.0)

    def loss(self, d, cfg):
        return compute_nextlat_losses(
            hidden_states=self.h, token_embeddings=self.e, head_weight=self.w,
            input_ids=self.ids, attention_mask=self.mask,
            assistant_loss_mask=self.mask, dynamics=d, config=cfg,
        )

    def test_zero_is_exact_identity_and_gradients_match(self):
        for backend in ("autograd", "shared_teacher_horizon_stacked"):
            d2 = copy.deepcopy(self.d)
            rng = torch.get_rng_state().clone()
            self.assertIs(noisy_training_anchor(self.h, dynamics=self.d, relative=0), self.h)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            a = self.loss(self.d, NextLatLossConfig(horizon=4, kl_implementation=backend))
            b = self.loss(d2, NextLatLossConfig(horizon=4, kl_implementation=backend, anchor_noise_relative=0))
            for k in ("ce", "smooth_l1", "kl", "loss"):
                self.assertTrue(torch.equal(a[k], b[k]), (backend, k))
            ga = torch.autograd.grad(a["loss"], [self.h, self.e, self.w, *self.d.parameters()])
            gb = torch.autograd.grad(b["loss"], [self.h, self.e, self.w, *d2.parameters()])
            self.assertTrue(all(torch.equal(x, y) for x, y in zip(ga, gb)))

    def test_noise_scale_identity_gradient_and_rng_replay(self):
        original = self.h.detach().clone()
        rng = torch.get_rng_state().clone()
        a = noisy_training_anchor(self.h, dynamics=self.d, relative=0.1)
        rms = self.h.detach().square().mean(-1, keepdim=True).sqrt()
        self.assertTrue(bool(((a.detach() - original).abs() <= 0.1*rms + 1e-6).all()))
        self.assertFalse(torch.equal(a, self.h))
        self.assertTrue(torch.equal(self.h.detach(), original))
        g, = torch.autograd.grad(a.sum(), [self.h])
        self.assertTrue(torch.equal(g, torch.ones_like(self.h)))
        torch.set_rng_state(rng)
        b = noisy_training_anchor(self.h, dynamics=self.d, relative=0.1)
        self.assertTrue(torch.equal(a, b))
        self.d.eval()
        rng = torch.get_rng_state().clone()
        self.assertIs(noisy_training_anchor(self.h, dynamics=self.d, relative=0.1), self.h)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_once_only_clean_ce_and_connected_all_horizons(self):
        for backend in ("autograd", "shared_teacher_horizon_stacked"):
            cfg = NextLatLossConfig(horizon=4, kl_implementation=backend, anchor_noise_relative=0.1)
            rng = torch.get_rng_state().clone()
            expected = noisy_training_anchor(self.h, dynamics=self.d, relative=0.1)
            predicted = []
            for step in range(1, 5):
                expected = self.d(expected[:, :-1], self.e[:, step:])
                predicted.append(expected)
            torch.set_rng_state(rng)
            actual = self.loss(self.d, cfg)
            for x, y in zip(actual["predicted_states_by_horizon"], predicted):
                self.assertTrue(torch.equal(x, y))
            clean = self.loss(self.d, NextLatLossConfig(horizon=4, kl_implementation=backend))
            self.assertTrue(torch.equal(actual["ce"], clean["ce"]))
            gs = torch.autograd.grad(actual["smooth_l1"] + actual["kl"], [self.h, self.e, *self.d.parameters()])
            self.assertTrue(all(bool(torch.isfinite(g).all()) and float(g.norm()) > 0 for g in gs))
            # The independent clean teacher formula must match every horizon.
            smooth = torch.stack([
                torch.nn.functional.smooth_l1_loss(p, self.h[:, s:].detach())
                for s, p in enumerate(predicted, 1)
            ])
            torch.testing.assert_close(actual["smooth_l1_by_horizon"], smooth)


if __name__ == "__main__":
    unittest.main()
