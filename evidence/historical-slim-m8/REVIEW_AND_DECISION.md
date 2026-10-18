# NextLat B1 serving review — 5 September 2026

## Goal and scope

Find the fastest useful batch-1 self-speculative decoder at a small incremental GPU-memory cost, using the joint d4-800M Qwen3-1.7B checkpoint and approximately 83M-parameter NextLat transition. The transformer hidden vector remains the complete recurrent state; the target owns verification, accepted-token commit, and state refresh. This review does not launch training, change the model architecture, or evaluate latent MCTS.

The immediate question is whether a small beam/tree improves **accepted output per unit of complete serving time**. A higher accepted count is insufficient if draft, verification, or scheduling costs rise faster. Selecting a different output by sequence log-probability is a separate quality-versus-compute experiment, not a lossless-decoding speed claim.

## What was going wrong

1. **Target arithmetic was being confused with recurrent error.** On one registered divergence, ordinary B1 and B14 target decodes with identical sequentially built prefix KV produce bitwise-identical full-vocabulary logits. Changing only attention to prefix/self split plus merge flips token 315 to 3363, exactly reproducing the native tree failure. Restoring ordinary attention restores every logit. This isolates the native FA3 cascade arithmetic as a sufficient cause in this case; it does not prove the cause of every possible mismatch.
2. **The beam prototype carried avoidable host synchronization.** Tensor scalar indexing in its protected greedy path forced CPU synchronization. The graphable fix preserves selected lines and FP32 scores exactly on actual-checkpoint A/B/A anchors. After that fix, eager versus captured FP8 drafting took 8.422 versus 2.497 ms. This does not isolate the scalar fix's speed benefit and is not an end-to-end speedup.
3. **The whole tree round still costs too much.** Traced full-H8 rounds averaged 9.342 ms for the native tree versus 5.867 ms for M8: about 59% more. The observed component differences include draft/packing (+1.651 ms), target execution (+0.594 ms), and verification setup (+1.360 ms). These are ordinal-4 workload averages over different anchors and trajectories, not paired same-state operator overhead. Trace intervals include host submission gaps and perturb scheduling; untraced client-wall measurements determine speed.
4. **Cold completion paths contaminated timing.** A 16-token warmup did not warm the full 128-token completion. Both fixtures now receive excluded full-length warmups. Original raw repetitions remain intact, with warm/cold arithmetic explicitly separated below.
5. **M8 retained unnecessary BF16 draft weights and used an expensive eager tail.** The opt-in slim B1/M8 path retains FP8 matrix copies, offloads the unused BF16 MLP, releases the gathered BF16 vocabulary head, and uses the existing FP8 graph for partial tails. The actual proposed length and canonical target/KV commit logic are unchanged. Live final validation is recorded below.
6. **Integration and result persistence had concrete bugs.** The ordinary-DECODE tree startup lacked the `TREE_NODE_BUDGET` import. After that fix, permanently reserved virtual request rows violated SGLang's idle pool invariant (`available_size=1`, `total_size=15`). Rows now use the existing allocator's `alloc_rows`/`free_rows` for one decode call, with return in `finally`; startup and cache clearing reserve nothing. The invariant checker remains enabled. The selector also saved completed arms only after all servers finished, so the first startup failure lost an earlier exact/M8 measurement. Each completed arm now saves immediately. Failed attempts are retained as failures, not promoted as evidence.

## Matched baseline and native-tree result

Same H100 PCIe allocation, target artifact, BF16 target arithmetic settings, FA3, two fixed prompts (ordinals 4 and 5), 128 output tokens per request, three repetitions, and separate untimed memory measurement. Runs are sequential, not interleaved AB/BA. These prompts have been used for debugging; this is a narrow systems decision, not an unseen-workload or deployment benchmark.

| Arm | All three repetitions | Warm repetitions 1–2 only | Output parity |
|---|---:|---:|---:|
| Exact target | 227.158 tok/s | 227.116 tok/s | reference |
| Original optimized M8 | 495.846 tok/s (2.183×) | 554.006 tok/s (2.439×) | 768/768 |
| Native FP8 N14 tree | 312.378 tok/s | 360.297 tok/s | 522/768 |

The warm-only calculation is a transparent post-hoc breakdown, not a replacement of the original all-three result. Native tree is about 35% slower than M8 in the warm breakdown and is not an admitted strict-parity speed point. Repeated tree divergences occur at ordinal 4/index 63 and ordinal 5/index 58.

Original M8 adds 546 MiB over exact (5,594 versus 5,048 MiB total GPU memory). Its emitted post-prefill yield is 762/195 = 3.908 tokens per counted verification, including the bonus. Native tree yields 762/177 = 4.305 on its own different trajectories; that is **not** a matched-trajectory acceptance improvement. Trace counters may include extra overlapped physical work; request counts and physical counts must not be interchanged.

## Final slim-M8 / ordinary-DECODE tree decision

After the startup-import fix, each arm saves separately. Both full 128-token fixtures are warmed before all three measured repetitions; memory polling is excluded from timing.

| Arm | Output tok/s | Speedup over exact | Extra GPU memory | Output parity |
|---|---:|---:|---:|---:|
| Exact target | 226.771 | 1.000× | reference: 5,048 MiB total | reference |
| Slim FP8 M8 | 553.704 | 2.442× | 266 MiB | 768/768 |
| Ordinary-DECODE FP8 N14 tree | 440.473 | 1.942× | 618 MiB | 768/768 |

Slim M8 repetitions are 554.440, 551.825, and 554.858 tok/s. Its ready and separately sampled peak memory are both 5,314 MiB. This saves **280 MiB**, about **51% of the original incremental memory**, without a demonstrated throughput gain: 553.704 tok/s is essentially the earlier warm M8's 554.006 tok/s. Realized yield remains 762/195 = 3.908 emitted post-prefill tokens per counted verification, including bonus. The result supports the leaner opt-in B1/M8 path, not a new speed record or broad-workload claim.

The final ordinary-DECODE tree-only comparison follows the row-lifetime fix and reuses these saved exact/M8 controls on the same allocation. No exact/M8 measurements were repeated; controls are sequential rather than interleaved. Startup, idle transitions, and full measured requests passed with the invariant checker enabled.

**Decision: keep slim M8; do not promote this N14 tree design.** Tree repetitions are 438.506, 440.984, and 441.943 tok/s. Its yield is 762/171 = 4.456 tokens per verification: now a valid **14.04% gain on identical outputs**. But amortized client-wall cost per counted verification rises from 7.113 to 10.196 ms (43.35%, including each request's shared overhead), so throughput is **20.45% lower**. This is enough to stop optimizing this particular width-8/two-line/N14 proposal policy as the next serving improvement; it does not reject every possible branching policy or sequence-scoring quality experiment.

Tree ready/peak memory is 5,666 MiB total, 618 MiB over exact. Unlike slim M8, the tree still retains its BF16 draft copies; these are current implementation footprints, not an irreducible memory cost of branching. Further deduplicating tree weights would not establish a speed win.

## Implications

- Keep model training and serving defects separate. More tokens cannot repair the identified verifier arithmetic, host synchronization, stale weight copies, or startup bug.
- Do not promote this width-8/two-line/N14 design on acceptance counts alone. It must beat the working M8 path on complete measured cost while preserving the declared target outputs.
- Do not add a heuristic low-logit-margin correction decoder. Reuse the existing target execution and KV ownership path; a margin heuristic cannot establish losslessness.
- Longer horizon training and the proposed noise, normalization, transition, and post-norm ablations remain deferred until a new model-fidelity hypothesis is actually being tested. No checkpoint was modified here.
- Current results cannot establish superiority to DSpark, frontier-scale speed, stochastic-decoding parity, or broad-workload deployment readiness.

## Evidence

- `fp8-tree-narrow3.json` and `.raw.json`: original untraced comparison and complete request payloads.
- `beam-capture-fp8.log`: actual-checkpoint eager/graph component comparison.
- `SAME_KV_CAUSAL_DIAGNOSIS.md`, `same-kv-b14.json`, tensor dumps, and `same-kv-probe-final.py`: same-prefix causal intervention and exact restoration.
- `phase-cost.json` and per-arm traces: complete-round timing attribution; not headline throughput.
- `slim-diagonal-warm3.log`: first retry failed startup; no final speed claim from that attempt.
- `diagonal-startup-repro.log`: captured missing-import root cause.
- `slim-diagonal-retry3.exact.json` and `.m8.json`: independently saved final full-warm measurements.
- `slim-diagonal-final-tree.json`: final passing tree comparison and explicit links to the saved exact/M8 controls.
- `slim-diagonal-final-tree.tree_n14_diagonal_target_graph.json`: independent tree arm, including all warmups, measured output IDs, counters, timing, memory, and launch command.

Checkpoint: `runix/nextlat-mcts/joint-d4-800007805-nextlat-serving:v0`, digest `915a5803e0ca5630b5431d75181b0a43`. Pinned SGLang commit: `baf09f39544d3cd812c77ed78132eedefaee5c5d`. Local overlay edits are an experimental working tree, not a regenerated/released content-addressed package. No commits or training launches were made.

The owned H100 pod `f214ac523ec349569b91393448a1f4f8` was gracefully released and provider-confirmed `TERMINATED` at approximately 02:24 UTC. Estimated allocation cost: **$3.77** for 1.15 hours. Artifacts are saved locally; the deadline watcher independently confirmed provider absence. Unrelated allocations were untouched.
