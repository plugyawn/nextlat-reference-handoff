# SGLang NextLat plugin

This directory is the narrow out-of-tree integration for exact Qwen3-1.7B
NextLat self-speculative decoding. It is source-locked to SGLang commit
`baf09f39544d3cd812c77ed78132eedefaee5c5d` and the acceptance semantics are
cross-checked against the pinned DeepSpec and SpecForge references.

## Implemented claim boundary

- One synchronous, eager, TP=PP=DP=1 serving tier: `topk=1`, a configured
  maximum of 1-32 drafts, BF16 target, and BF16 runtime dynamics. The default
  qualification window is eight drafts.
- Batched per-request proposal lengths in `[0, configured maximum]`. The output
  and commit tensors retain the fixed configured stride `M+1`, while every
  target verify row is compact `[anchor, proposed drafts...]`; the anchor is
  always present.
- Consecutive greedy accepted-prefix computation, fixed-stride output
  (`accepted drafts + target bonus`), and the exact hidden row for the next
  recurrent seed.
- No draft model and no draft KV. Scratch reservation comes from SGLang's
  synchronous V2 scheduler path and the only commit calls SGLang's
  `move_accept_tokens_to_target_kvcache`.
- Fail-closed sidecar hashes, executable target-model manifest binding, installed
  SGLang source hashes, local safetensors-only target loading, and deterministic
  runtime receipts.

Unsupported sampling, penalties, logit bias, custom processors, grammar,
logprobs, LoRA, quantization, overlap, CUDA graphs, distributed serving, and live
weight updates raise errors. There is no ordinary-decoding or CPU fallback.

This is now a launchable implementation, but it is **not GPU-qualified yet**.
The checked-in evidence is CPU-only; no claim is made yet for FlashInfer runtime
parity, target-KV movement on CUDA, accepted-length distribution, or speed.
Latent MCTS is explicitly outside this package.

## Artifact admission

The dynamics sidecar contains exactly:

```text
COMPLETE
config.json
dynamics.safetensors
sha256.json
```

Its `target_checkpoint_sha256` binds the executable target payload: target
`config.json` plus either `model.safetensors`, or the safetensors index and every
referenced shard. Alternate, unreferenced, symlinked, incomplete, or modified
weight payloads are rejected. The sidecar stores FP32 source dynamics; the
serving worker makes one explicit BF16 runtime copy and records both dtypes.
The config preserves one of two explicitly different source authorities:

- `qualified_megatron_tp1_export_v2` preserves the TP1 export manifest,
  qualification, checkpoint, source-lock, runtime-image, token-count, update,
  and branch lineage.
- `verified_pytorch_joint_checkpoint_v1` preserves the strict
  `nextlat.run_utils` checkpoint manifest and aggregate integrity digest,
  train-state digest, source commit, token-count, update, branch (when the
  checkpoint records enough horizon state), optimizer presence, and W&B run or
  checkpoint-artifact metadata when present.

The PyTorch authority is an inference admission path. It is not described as a
Megatron conversion, TP1 numerical qualification, or distributed-training
result anywhere in the sidecar or runtime receipt.

Do not assemble these files manually. Derive them atomically from the completed
`systems/export_tp1.py` output:

```bash
sglang-nextlat-build-sidecar \
  --export /absolute/path/to/tp1-export \
  --output /absolute/path/to/new-serving-sidecar
```

Alternatively, derive the sidecar directly from a completed, verified joint
PyTorch checkpoint:

```bash
sglang-nextlat-build-sidecar \
  --pytorch-checkpoint /absolute/path/to/checkpoint-246022601-tokens \
  --expected-checkpoint-manifest-sha256 <trusted-lowercase-sha256> \
  --output /absolute/path/to/new-serving-sidecar
```

That checkpoint must have exact `COMPLETE` contents, no `INCOMPLETE` marker or
symlinks, and a `sha256.json` covering every payload under `model/`,
`dynamics.safetensors`, `train_state.json`, optional `optimizer.pt`, and an
explicitly receipted `rng_state.pt` when the train state declares one. Extra or
missing files, non-FP32 dynamics, tensor roster/shape drift, target-manifest
drift, malformed source/W&B metadata, and optimizer/RNG presence mismatches all
fail before output creation. The expected manifest digest is mandatory and must
come from trusted checkpoint provenance: the builder hashes the literal source
`sha256.json` and compares it before accepting any payload hashes. Do not derive
that expected value from the local checkpoint after transfer, because doing so
would not detect a payload plus recomputed-manifest substitution.

Both builders rehash the full source manifest, target model, and recurrent
weights, then promote the four-file sidecar only after its strict reader accepts
it. The TP1 path additionally verifies `export_receipt.json` and its live FP32
parameter/qualification lineage.

## CPU evidence

From the repository root:

```bash
PYTHONPATH=systems/sglang-nextlat/src:systems/sglang-nextlat/tests \
  .venv/bin/python -m unittest discover \
  -s systems/sglang-nextlat/tests -v
```

The tests cover mixed proposal lengths `[0, 2, 4, 8]`, early mismatch, full
acceptance, anchor/bonus/KV-index semantics, configured recurrent windows,
strict sidecar and target admission, atomic export-to-sidecar lineage, plugin
routing, immutable image contracts, and byte-deterministic receipts. Test-only
small tensor contracts avoid materializing the production 83M-parameter
dynamics model.

## Claim-grade SM90 image

A mutable host installation is only an engineering fallback. The serving claim
uses `Dockerfile.sm90` and an immutable SGLang base image built from the pinned
checkout. Build that checkout's official `runtime` target with
`BRANCH_TYPE=local` and `SGLANG_BUILD_COMMIT=baf09f...`, inspect its local
`sha256:` image ID, then derive and receipt this plugin image:

```bash
sglang-nextlat-build-sm90-image \
  --package-root /absolute/path/to/systems/sglang-nextlat \
  --base-image sha256:BASE_IMAGE_ID \
  --tag nextlat-sglang:sm90 \
  --receipt /absolute/path/to/new-image-receipt.json
```

The derived build refuses mutable base tags. It verifies the protected plugin
source manifest, hashes 26 pinned installed SGLang source files, requires an
SM90-capable Torch wheel plus FlashInfer, checks the installed entry point and
Python dependencies, and emits a self-hashed receipt containing both base and
derived content-addressed image IDs. No image has been built in this workspace.

## Real GPU smoke entrypoint

Install this package into the pinned SGLang environment so its entry point is
discoverable, then run:

```bash
sglang-nextlat-smoke \
  --model /absolute/path/to/qualified-target-model \
  --sidecar /absolute/path/to/nextlat-sidecar \
  --receipt /absolute/path/to/new-smoke-receipt.json \
  --max-draft-tokens 8 \
  --gpu-index 0
```

The command launches the real SGLang HTTP server, exercises one four-request
batch whose first decode has proposal lengths `[0, 2, 4, 8]`, terminates its
owned process group, and writes a pass receipt only after successful generation.
It refuses to overwrite a different receipt. It has not been executed in this
workspace.

## Source and proof notes

See [SOURCE_ANCHORS.md](SOURCE_ANCHORS.md) for exact upstream APIs and the
anchor/accepted-draft/bonus/KV invariant. Static source receipts are in
`receipts/`.

Current blockers to a serving claim are operational rather than hidden by the
code:

1. One admitted source must finish: either a qualified TP1 export or a completed
   verified joint PyTorch checkpoint. The atomic builder must then produce its
   target-bound sidecar. Admitting the latter does not close the TP1 gate.
2. The pinned SGLang runtime base and derived SM90 image must build and receive
   their content-addressed image receipt. The local CPU environment cannot
   import pinned SGLang because it lacks runtime packages such as `orjson`.
3. The GPU smoke above must pass inside that image before this worker is called runtime-correct.
   A larger parity/stress run is still required before any efficiency claim.
