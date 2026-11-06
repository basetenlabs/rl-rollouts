# RL Rollouts

Your rollout deployment serves a model whose weights follow your trainer. You publish checkpoints
to an Amazon S3 or Google Cloud Storage (GCS) bucket you own, or with `rl.cannery` to Cannery volume
versions, and request each one; every replica applies the new weights in place and keeps serving.
You never redeploy to change weights.

## How it works

1. You upload a **full checkpoint** (an *anchor*) once, then a small **delta** after each training step.
2. When a snapshot is completely uploaded, you send a **hot-load request** naming it.
3. Each replica prepares the update, pauses briefly for a delta (seconds to about a minute for a new anchor),
   swaps weights and resumes, keeping in-flight requests. Full-anchor preparation can slow serving for several minutes.
4. Every response reports which policy produced it, so your trainer can track off-policy samples.

## Use a model we ship

Each recipe is a `config.yaml` under `models/<model>/<topology>/`:

| Recipe | GPUs |
|---|---|
| [`models/glm-5.3-flash/b200-8`](models/glm-5.3-flash/b200-8/config.yaml) | B200:8 |
| [`models/deepseek-v4.1-flash-mxfp4/b200-8`](models/deepseek-v4.1-flash-mxfp4/b200-8/config.yaml) | B200:8 |
| [`models/glm-5.2/b300-8`](models/glm-5.2/b300-8/config.yaml) | B300:8 |
| [`models/kimi-k3/gb300-8`](models/kimi-k3/gb300-8/config.yaml) | two GB300:4 nodes (MXFP4) |
| [`models/qwen3.5-9b/h100-1`](models/qwen3.5-9b/h100-1/config.yaml) | H100:1 |
| [`models/qwen3.8-27b/h200-1`](models/qwen3.8-27b/h200-1/config.yaml) | H200:1 |

1. [Deploy](docs/deploy.md) a recipe: storage, push, call.
2. [Publish checkpoints and request them](docs/checkpoints.md): bucket layout, hot-load API; the [delta format](docs/delta-format.md).
3. Read the [RL request and response fields](docs/rl-api.md): policy versions, token IDs, expert routes.
4. Wire it into your trainer: [Miles and Slime](docs/training-loops.md).

[`publish_checkpoint.py`](publish_checkpoint.py) publishes anchors and deltas in the expected layout.

## Bring your own model

A model we do not ship can use the same in-place updates when you supply its weight spec, and its
SGLang model class if SGLang lacks one. Start at [Add your own model](docs/add-your-own-model.md);
[`examples/`](examples/) holds two worked packages, and the [extension API](docs/extension-api.md) lists what
a package may rely on.

Every recipe and example sets the current image as `gpuTRTImage`; a `sglang.patch` is written against
the current SGLang release. A new image with a new SGLang pin gets its release after review; until
then, the release below is the previous pin's, and a patch written against it does not apply to the new image.

<!-- image:start -->
**Current image:** `baseten/dynamo-cache-aware-routing@sha256:e0a8df8a0dd9442710030ef68538665b1ef0024b7d856403904e02a7e142cab2`.
<!-- image:end -->
<!-- sglang-release:start -->
**Current SGLang release:** [`sglang-222dc664fc24`](https://github.com/basetenlabs/rl-rollouts/releases/tag/sglang-222dc664fc24).
<!-- sglang-release:end -->

Custom models and SGLang patches are available only for organizations where they are enabled.
Contact the Baseten team (support@baseten.co) for details.
