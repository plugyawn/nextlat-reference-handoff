# NextLat joint post-training and recovered SGLang serving: reference handoff

Update, October 18: the original export was training-only. A later SSD search
recovered `systems/sglang-nextlat`, including the Slim B1/M8 FP8 graph path,
native verification/KV code, and historical exact/spec serving receipts.
See `SERVING_RECOVERY.md` and `SERVING_ARCHIVE_SOURCE.json`. The repository is
private following the collaborator's confirmed initial clone.

Focused source snapshot shared by Progyan's existing NextLat Codex agent with
Onix for a training/decoding audit. This is **historical reference code, not a
newly validated production release**.

The files are recovered from archived source commit
`1224ca1073f0bbfada0751178cf26479dd8d0a71` (September 7, 2026). The new public
repository intentionally has no original Git history, checkpoints, datasets,
credentials, raw conversations, or operational logs. `ARCHIVE_SOURCE.json`
identifies each original file by SHA-256.

## Read these first

- `nextlat/core.py`: residual dynamics, exact token offsets, uniform multi-step
  SmoothL1 + full-vocabulary KL, and the joint Qwen wrapper.
- `configs/nextlat_qwen3_1_7b_d4_long_1xh200.json`: historical original d4 recipe.
- `configs/nextlat_qwen3_1_7b_joint_d4_400m_to800m.json`: later continuation recipe,
  including its actual parent history. This was not a fresh 800M-token run.
- `nextlat/train.py`: actual trainer, optimizer groups, and resume paths.
- `nextlat/eval_recursive.py`: recursive-state and accepted-prefix diagnostics.
- `nextlat/shared_teacher_loss.py` and `nextlat/triton_horizon_kl.py`: later
  training-side vocabulary-loss optimizations; these are not serving kernels.

## Semantics to compare with the new implementation

The state in this Qwen wrapper is the decoder's **post-final-RMSNorm**
`last_hidden_state`, consumed directly by `lm_head`. Do not normalize that state
a second time at readout. Other state conventions can be valid, but training
and inference must agree.

At horizon j, the transition consumes the next observed corpus token embedding:
`(h_t, e(x_{t+1})) -> h_hat_{t+1}`. It then reuses its predicted state for the
remaining horizon, without detaching between recurrent steps. The exact source
state and token embeddings remain gradient-connected to Qwen. Future exact
hidden states, teacher logits, and the output-head weights used in auxiliary KL
are stop-gradient targets. Ordinary CE still trains the output head.

For the original d4 recipe, the loss is CE plus the mean of four SmoothL1
terms and the mean of four KL terms, both with coefficient 1. The default
three-linear residual GELU sidecar has 82,726,912 parameters at Qwen3-1.7B's
hidden size. See the code for masking and the terminal-state KL offset; these
details should be compared directly rather than inferred from the phrase
"NextLat loss".

The snapshot contains optional anchor noise and imported DMT/Countdown helpers
because the archived trainer imports them. Their presence does not imply they
were enabled in the original d4 runs. In particular, ordinary joint d4 does
not use detached repair or policy-gradient training. Historical config `claim`
fields record experimental intentions, not established conclusions.

## What this does not establish

The **initial training-only export** did not contain the historical optimized
Slim M8 SGLang worker. The subsequently recovered package is now included, but
its runtime has not yet been revalidated on the current allocation. The older
greedy speed measurements used different checkpoints and
workloads from Onix's current runs and are not a matched baseline. No new GPU
training, loss-parity run, or speed benchmark was performed when exporting
this code. The initial `ARCHIVE_SOURCE.json` describes that first export's
scope; the new serving manifest describes the added material separately.

`requirements-nextlat.txt` is the archived environment pin, not a claim that
an arbitrary current environment is compatible. Review runtime contracts in
the selected configs before attempting a run. The trainer expects prepared
data, and continuation recipes additionally require their recorded parent
artifacts; neither is included.

## Small checks

In a compatible PyTorch environment, the core checks can be run with:

```sh
python -m unittest discover -s tests -p test_nextlat.py
python -m unittest discover -s tests -p test_dense_kl_forward_gradient.py
```

The upstream parity test additionally requires a local checkout of
[JaydenTeoh/NextLat](https://github.com/JaydenTeoh/NextLat) pinned at
`3770be6009cea2b3c455a9ce7f2ca88b504bb955`; set `NEXTLAT_OFFICIAL_ROOT` to that
checkout. This is the formulation the historical implementation targeted.

The current handoff is for code inspection and narrow comparison. Finish the
already-running matched AR/spec measurements before importing old optimizations
or changing the training recipe.
