"""Executable parity checks against the pinned upstream NextLat source."""

from __future__ import annotations

import ast
import copy
import hashlib
import os
import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from nextlat.core import (
    NextLatDynamics,
    NextLatLossConfig,
    compute_nextlat_losses,
)


UPSTREAM_COMMIT = "3770be6009cea2b3c455a9ce7f2ca88b504bb955"
UPSTREAM_SHA256 = {
    "models/model_nextlat.py": (
        "bd324f6c8ac21b7a475adfcf8ccde7724cfe5af607ff1f281cbd7c3fd877c096"
    ),
    "models/model_base.py": (
        "e2645ed8e4d83de4daf9a3759b91537ed9c4216d4a189a9723d6d8cdd14ba34a"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _upstream_root() -> Path:
    candidates = []
    if os.environ.get("NEXTLAT_OFFICIAL_ROOT"):
        candidates.append(Path(os.environ["NEXTLAT_OFFICIAL_ROOT"]))
    candidates.extend(
        [
            Path.home() / "nextlat-official",
            Path("/private/tmp/nextlat-official-20260826"),
        ]
    )
    for candidate in candidates:
        if (candidate / "models/model_nextlat.py").is_file():
            return candidate
    raise RuntimeError(
        "pinned official NextLat checkout is required; set NEXTLAT_OFFICIAL_ROOT"
    )


def _validate_upstream(root: Path) -> None:
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != UPSTREAM_COMMIT:
        raise RuntimeError(f"upstream commit mismatch: {commit} != {UPSTREAM_COMMIT}")
    for relative, expected in UPSTREAM_SHA256.items():
        observed = _sha256(root / relative)
        if observed != expected:
            raise RuntimeError(
                f"upstream source hash mismatch for {relative}: {observed} != {expected}"
            )


def _class_from_source(
    path: Path,
    class_name: str,
    namespace: Dict[str, Any],
    method_names: Iterable[str] | None = None,
    strip_bases: bool = False,
) -> type:
    tree = ast.parse(path.read_text(), filename=str(path))
    original = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    extracted = copy.deepcopy(original)
    if strip_bases:
        extracted.bases = []
        extracted.keywords = []
    extracted.decorator_list = []
    if method_names is not None:
        wanted = set(method_names)
        extracted.body = [
            node
            for node in extracted.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in wanted
        ]
    module = ast.fix_missing_locations(
        ast.Module(body=[extracted], type_ignores=[])
    )
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[class_name]


class OfficialNextLatParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = _upstream_root()
        _validate_upstream(cls.root)
        namespace: Dict[str, Any] = {"torch": torch, "nn": nn, "F": F}
        cls.official_norm = _class_from_source(
            cls.root / "models/model_base.py", "LayerNorm", namespace
        )
        namespace["LayerNorm"] = cls.official_norm
        cls.official_dynamics = _class_from_source(
            cls.root / "models/model_nextlat.py",
            "NextLatDynamicsModel",
            namespace,
        )
        cls.official_loss = _class_from_source(
            cls.root / "models/model_nextlat.py",
            "NextLat",
            namespace,
            method_names=("_categorical_kl_loss", "_nextlat_loss_function"),
            strip_bases=True,
        )

    def test_dynamics_matches_executed_official_class(self) -> None:
        torch.manual_seed(17)
        config = SimpleNamespace(
            n_embd=64, proj_factor=1.0, dropout=0.0, bias=False
        )
        official = self.official_dynamics(config).eval()
        local = NextLatDynamics(64, projection_factor=1.0, dropout=0.0).eval()
        local.load_state_dict(
            {
                name.replace("norm_x.", "norm."): value
                for name, value in official.state_dict().items()
            }
        )
        current = torch.randn(2, 5, 64)
        token = torch.randn(2, 5, 64)
        torch.testing.assert_close(
            local(current, token), official(current, token), rtol=0, atol=0
        )

    def test_losses_match_executed_official_methods(self) -> None:
        torch.manual_seed(23)
        batch, length, hidden_size, vocab = 2, 7, 64, 31
        hidden = torch.randn(batch, length, hidden_size)
        embeddings = torch.randn(batch, length, hidden_size)
        input_ids = torch.randint(0, vocab, (batch, length))
        attention = torch.tensor(
            [[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0, 0]],
            dtype=torch.bool,
        )
        assistant = torch.tensor(
            [[0, 0, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 0, 0]],
            dtype=torch.bool,
        )
        head = torch.randn(vocab, hidden_size)
        dynamics = NextLatDynamics(hidden_size, projection_factor=1.0)
        config = NextLatLossConfig(
            lambda_h=0.7, lambda_kl=1.3, vocab_chunk_tokens=2
        )
        local = compute_nextlat_losses(
            hidden_states=hidden,
            token_embeddings=embeddings,
            input_ids=input_ids,
            attention_mask=attention,
            assistant_loss_mask=assistant,
            head_weight=head,
            dynamics=dynamics,
            config=config,
        )

        predicted = dynamics(hidden[:, :-1], embeddings[:, 1:])
        transition_mask = attention[:, :-1] & attention[:, 1:]
        kl_mask = (
            attention[:, :-2]
            & attention[:, 1:-1]
            & attention[:, 2:]
            & assistant[:, 2:]
        )
        harness = self.official_loss()
        harness.model = SimpleNamespace(
            lm_head=SimpleNamespace(weight=nn.Parameter(head.clone()))
        )
        teacher_logits = F.linear(hidden[:, 1:-1], head)
        official_smooth, _, official_kl = harness._nextlat_loss_function(
            predicted,
            hidden[:, 1:],
            input_ids[:, 2:],
            teacher_logits,
            token_pred_mask=kl_mask,
            mse_mask=transition_mask,
        )
        torch.testing.assert_close(
            local["smooth_l1"], official_smooth, rtol=1e-6, atol=1e-6
        )
        torch.testing.assert_close(local["kl"], official_kl, rtol=1e-6, atol=1e-6)

    def test_d4_recursive_losses_match_executed_official_methods(self) -> None:
        torch.manual_seed(29)
        batch, length, hidden_size, vocab = 2, 8, 64, 31
        hidden = torch.randn(batch, length, hidden_size)
        embeddings = torch.randn(batch, length, hidden_size)
        input_ids = torch.randint(0, vocab, (batch, length))
        attention = torch.tensor(
            [[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 0, 0]],
            dtype=torch.bool,
        )
        assistant = torch.tensor(
            [[0, 0, 1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1, 0, 0]],
            dtype=torch.bool,
        )
        head = torch.randn(vocab, hidden_size)
        dynamics = NextLatDynamics(hidden_size, projection_factor=1.0)
        config = NextLatLossConfig(
            lambda_h=0.7,
            lambda_kl=1.3,
            vocab_chunk_tokens=2,
            horizon=4,
        )
        local = compute_nextlat_losses(
            hidden_states=hidden,
            token_embeddings=embeddings,
            input_ids=input_ids,
            attention_mask=attention,
            assistant_loss_mask=assistant,
            head_weight=head,
            dynamics=dynamics,
            config=config,
        )

        harness = self.official_loss()
        harness.model = SimpleNamespace(
            lm_head=SimpleNamespace(weight=nn.Parameter(head.clone()))
        )
        recurrent = hidden
        official_smooth = []
        official_kl = []
        for step in range(1, config.horizon + 1):
            recurrent = dynamics(recurrent[:, :-1], embeddings[:, step:])
            transition_width = length - step
            transition_mask = torch.stack(
                [
                    attention[:, offset : offset + transition_width]
                    for offset in range(step + 1)
                ]
            ).all(dim=0)
            kl_width = length - step - 1
            kl_mask = torch.stack(
                [
                    attention[:, offset : offset + kl_width]
                    for offset in range(step + 2)
                ]
            ).all(dim=0) & assistant[:, step + 1 :]
            teacher_logits = F.linear(hidden[:, step:-1], head)
            smooth, _, kl = harness._nextlat_loss_function(
                recurrent,
                hidden[:, step:],
                input_ids[:, step + 1 :],
                teacher_logits,
                token_pred_mask=kl_mask,
                mse_mask=transition_mask,
            )
            official_smooth.append(smooth)
            official_kl.append(kl)

        torch.testing.assert_close(
            local["smooth_l1_by_horizon"],
            torch.stack(official_smooth),
            rtol=1e-6,
            atol=1e-6,
        )
        torch.testing.assert_close(
            local["kl_by_horizon"],
            torch.stack(official_kl),
            rtol=1e-6,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
