import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nextlat.evaluate import _accepted_prefix_summary
from nextlat.tracking import numeric_payload, start_wandb_tracker
from nextlat.train import _optimizer_checkpoint_tokens
from nextlat.train import _resolve_training_sdpa_backend, _training_sdpa_context


class FakeRun:
    def __init__(self):
        self.entity = "runix"
        self.project = "nextlat-mcts"
        self.id = "run_123"
        self.name = "long-run"
        self.url = "https://wandb.ai/runix/nextlat-mcts/runs/run_123"
        self.summary = {}
        self.defined = []
        self.logged = []
        self.finished = []

    def define_metric(self, *args, **kwargs):
        self.defined.append((args, kwargs))

    def log(self, payload):
        self.logged.append(payload)

    def finish(self, exit_code=0):
        self.finished.append(exit_code)


TRACKING_CONFIG = {
    "backend": "wandb",
    "required": True,
    "mode": "online",
    "entity": "runix",
    "project": "nextlat-mcts",
    "run_name": "long-run",
    "group": "nextlat",
    "job_type": "train",
    "tags": ["test"],
}


class NextLatRuntimeTests(unittest.TestCase):
    def test_training_sdpa_backend_is_explicit_and_fail_closed(self):
        self.assertIsNone(
            _resolve_training_sdpa_backend({"attention": "sdpa"})
        )
        with self.assertRaisesRegex(RuntimeError, "must be omitted"):
            _resolve_training_sdpa_backend(
                {"attention": "sdpa", "sdpa_backend": "flash"}
            )
        with self.assertRaisesRegex(RuntimeError, "requires model.attention"):
            _resolve_training_sdpa_backend(
                {"attention": "eager", "sdpa_backend": "cudnn"}
            )
        with mock.patch(
            "nextlat.train.torch.backends.cudnn.is_available", return_value=False
        ):
            with self.assertRaisesRegex(RuntimeError, "cuDNN is unavailable"):
                _resolve_training_sdpa_backend(
                    {"attention": "sdpa", "sdpa_backend": "cudnn"}
                )
        with mock.patch(
            "nextlat.train.torch.backends.cudnn.is_available", return_value=True
        ), mock.patch(
            "nextlat.train.torch.backends.cuda.cudnn_sdp_enabled",
            return_value=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "disabled at runtime"):
                _resolve_training_sdpa_backend(
                    {"attention": "sdpa", "sdpa_backend": "cudnn"}
                )

    def test_training_sdpa_context_forces_only_cudnn(self):
        sentinel = object()
        with mock.patch(
            "nextlat.train.torch.backends.cudnn.is_available", return_value=True
        ), mock.patch(
            "nextlat.train.torch.backends.cuda.cudnn_sdp_enabled",
            return_value=True,
        ):
            backend = _resolve_training_sdpa_backend(
                {"attention": "sdpa", "sdpa_backend": "cudnn"}
            )
        with mock.patch("nextlat.train.sdpa_kernel", return_value=sentinel) as force:
            self.assertIs(_training_sdpa_context(backend), sentinel)
        force.assert_called_once_with(backend)

    def test_numeric_payload_uses_tokens_as_shared_axis(self):
        payload = numeric_payload(
            "train",
            {
                "type": "train",
                "tokens_seen": 24,
                "update": 3,
                "loss": 1.5,
                "position_totals": {"ce": 9},
                "kl_by_horizon": [0.4, 0.8],
                "values": [1, 2, 3],
                "label": "ignored",
            },
        )
        self.assertEqual(payload["tokens_seen"], 24.0)
        self.assertEqual(payload["update"], 3.0)
        self.assertEqual(payload["train/loss"], 1.5)
        self.assertEqual(payload["train/position_totals/ce"], 9.0)
        self.assertEqual(payload["train/kl_by_horizon/h1"], 0.4)
        self.assertEqual(payload["train/kl_by_horizon/h2"], 0.8)
        self.assertNotIn("train/values", payload)
        self.assertNotIn("train/label", payload)

    def test_nonfinite_tracking_metric_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            numeric_payload("train", {"tokens_seen": 1, "loss": float("nan")})

    def test_required_tracking_rejects_missing_key(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ, {}, clear=True
        ):
            with self.assertRaisesRegex(RuntimeError, "WANDB_API_KEY"):
                start_wandb_tracker(
                    tracking_config=TRACKING_CONFIG,
                    run_dir=Path(directory),
                    run_id="run_123",
                    resume_mode="never",
                    experiment_config={"claim": "test"},
                    source_code_commit="a" * 40,
                    phase="train",
                )

    def test_online_tracker_writes_secret_free_receipt_and_resumes(self):
        fake = FakeRun()
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ, {"WANDB_API_KEY": "secret-key"}, clear=True
        ), mock.patch("nextlat.tracking.wandb.init", return_value=fake) as init, mock.patch(
            "nextlat.tracking.wandb.Settings", return_value=object()
        ):
            tracker = start_wandb_tracker(
                tracking_config=TRACKING_CONFIG,
                run_dir=Path(directory),
                run_id="run_123",
                resume_mode="must",
                experiment_config={"claim": "test"},
                source_code_commit="a" * 40,
                phase="evaluation",
            )
            receipt_text = tracker.receipt_path.read_text()
            receipt = json.loads(receipt_text)
            self.assertEqual(receipt["url"], fake.url)
            self.assertEqual(receipt["resume_mode"], "must")
            self.assertNotIn("secret-key", receipt_text)
            self.assertEqual(init.call_args.kwargs["resume"], "must")
            tracker.log_record("train", {"tokens_seen": 10, "update": 2, "loss": 1})
            tracker.finish(summary_namespace="final", summary={"tokens_seen": 10})
            self.assertEqual(fake.logged[-1]["train/loss"], 1.0)
            self.assertEqual(fake.summary["final/tokens_seen"], 10.0)
            self.assertEqual(fake.finished, [0])

    def test_registered_rungs_are_optimizer_resumable(self):
        config = {
            "checkpoint_tokens": [24, 120, 240],
            "resumable_checkpoint_tokens": [120, 240],
            "registered_token_rungs": [120, 240],
        }
        self.assertEqual(_optimizer_checkpoint_tokens(config), {120, 240})
        config["registered_token_rungs"] = [24, 120, 240]
        with self.assertRaisesRegex(RuntimeError, "optimizer-resumable"):
            _optimizer_checkpoint_tokens(config)

    def test_prefix_summary_uses_stable_tail_rates_and_nearest_rank(self):
        summary = _accepted_prefix_summary([1, 4, 8, 12, 16])
        self.assertEqual(summary["median"], 8)
        self.assertEqual(summary["p90"], 16)
        self.assertEqual(summary["maximum"], 16)
        self.assertEqual(summary["at_least_8_rate"], 3 / 5)
        self.assertEqual(summary["at_least_16_rate"], 1 / 5)


if __name__ == "__main__":
    unittest.main()
