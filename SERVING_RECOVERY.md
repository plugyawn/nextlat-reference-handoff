# Recovered serving source and the historical continuation candidate

Recovered on October 18, 2026 from the SSD. This corrects the initial handoff's
missing-serving-source limitation; it does not announce a reproduced speedup.

## Code

`systems/sglang-nextlat` is the archived September 9 package. Its `worker.py`
contains `NEXTLAT_B1_FP8_SLIM`, captured M8 recurrence, FP8 draft/readout weights,
and the hot-vocabulary path. Native verification and KV commit code are also
included. SGLang is pinned to
`baf09f39544d3cd812c77ed78132eedefaee5c5d`.

The package's old README and runtime-receipt defaults describe an earlier
eager-only qualification boundary. They were preserved as recovered historical
files and are NOT an up-to-date feature/validation matrix for the later worker.
Inspect the actual code, flags and archived executed receipts. The recovered
Dockerfile is `systems/Dockerfile.sglang-recovered`; its context is `systems`.
That Docker build additionally requires the pinned upstream checkout at
`systems/sglang`; the checkout itself is not vendored here. The recovered
`run_serving_cell.py`, exporter and bootstrap reference are under
`systems/recovered-serving-controller`. They retain historical paths and flags:
adapt those deliberately, and do not copy old allocation/teardown controllers.
Not every file mentioned by older image-building helpers was in this archive.

September 9 source has not been shown byte-identical to the September 5 winning
overlay. Bind the selected overlay and runtime before claiming reproduction;
do not make recovery into a long qualification matrix that blocks training.

## Concrete historical result, not a broad benchmark

The included `evidence/historical-slim-m8/REVIEW_AND_DECISION.md` and raw exact/M8
JSON receipts identify a real narrow B1 greedy result:

| Arm | Output tok/s | Relative speed | Extra GPU memory |
|---|---:|---:|---:|
| Exact target | 226.771 | 1.000x | reference |
| Slim FP8 M8 | 553.704 | 2.442x | 266 MiB |

Both arms used the same H100 PCIe allocation, target artifact and two fixed,
previously debugged prompts, full-length warmups, 128 output tokens and three
sequential repetitions. Output-token parity was 768/768. Realized emitted yield
was 3.908 post-prefill tokens per counted verification, including bonus—not an
offline mean accepted length of six. The source review records all limitations.
The raw `phase-cost.json` contains earlier traced original-M8/tree arms, not a
matched decomposition of the final slim speed point. Do not combine them into
an apparently exact timing bar for the final slim result.

This is a candidate to recover and continue, not proof that it outperforms
Onix's current model on its broader workload or that it retains general quality.

## Checkpoint and continuation

- Serving artifact: `runix/nextlat-mcts/joint-d4-800007805-nextlat-serving:v0`.
- Artifact digest: `915a5803e0ca5630b5431d75181b0a43`.
- Full local source checkpoint on the SSD:
  `nextlat/serving/staging/checkpoint-800007805-clean`.
- Complete local payload includes target model, FP32 dynamics, optimizer state,
  RNG, checkpoint hashes and train state. No weights or optimizer were committed
  to GitHub.
- Recorded tokens: 800,007,805; update: 36,379; original world size: one.
- Training source: `30469bff0b7f2936ba5878eca69c7e02ecd7e96c`.
- W&B run: `jointd4_550m_800m_triton_30469bf_h200_r1`.

Import optimizer state by the recorded parameter/group mapping, not an assumed
new parameter order. Changing world size and corpus must preserve the declared
global optimizer/token clocks while explicitly recording the new sampler origin.
If only weights can be obtained promptly, label a new optimizer/schedule honestly
rather than calling it an exact resume. No frozen-Qwen repair is intended.

For a useful next run, preserve the original residual architecture and connected
d4 objective while using genuinely fresh reasoning data plus general-language
rehearsal, with correct chat/thinking masks and problem-level exclusions. Choose
the concrete mixture/rates from available data and retention evidence; the old
tiny repeated math pool is not a general-purpose scaling recipe.
