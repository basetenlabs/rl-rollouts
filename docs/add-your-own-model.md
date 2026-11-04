# Add your own model

A model we do not ship can use the same in-place weight updates as the [recipes we ship](../README.md#use-a-model-we-ship).
You bring one Python package; Baseten's serving image does the rest.

> Custom models and SGLang patches need enablement for your organization; see [availability](../README.md#bring-your-own-model).

## Which case are you?

| Your model | Your package holds | Example |
|---|---|---|
| SGLang already has its model class | a [weight spec](weight-spec.md) | [DeepSeek V3.2](../examples/deepseek-v3.2/README.md) |
| Its forward pass differs from any SGLang class | the class, an [`sglang.patch`](sglang-patches.md) if the engine needs changes, and a weight spec | [joey-1](../examples/joey-1/README.md) |

Either way your `config.json` names an architecture your spec describes, and it may not be one of ours.
When you bring a class, give your checkpoint its own architecture name in every anchor you publish
(Transformers loads by `model_type`, so training is unaffected):

```python
import json
config = json.load(open("config.json"))
config["architectures"] = ["MyModelForCausalLM"]
json.dump(config, open("config.json", "w"), indent=2)
```

## What you have, and what you don't

| You have | You don't have |
|---|---|
| This repository: the guide, recipes and both examples | The weight-sync runtime source (the image ships it as bytecode) |
| The serving image, with the `weight_spec` library and the [conformance kit](conformance-kit.md) | The protected SGLang files and definitions ([patch rules](sglang-patches.md#rules)) |
| The image's SGLang export, as a [release](https://github.com/basetenlabs/rl-rollouts/releases) named on the README's [current-image line](../README.md#bring-your-own-model) | Changes to the image: the kit checks every file against its manifest |
| Baseten workstations and BIS on your workspace | Compiled kernels (Triton kernels in Python files are fine) |
| | Multi-node recipes for your own model |
| | Our architecture names |

## The dev loop

```text
export ──► class + patch ──► spec ──► CPU spec check ──► kit --dev ──► upload ──► deploy ──► your semantics checks
                ▲                           │               │                                         │
                └───────── fix ─────────────┴───────────────┘           for support: kit run ──► kit-out to us
```

1. **Get the export** of the image's SGLang pin (only if you bring a class or a patch): the README's
   [current-image line](../README.md#bring-your-own-model) names the image (`gpuTRTImage`, which the examples already set) and its SGLang
   release; download that [release](https://github.com/basetenlabs/rl-rollouts/releases) ([details](sglang-patches.md)).
2. **Write the class and the patch**: the class in your package's `models/`, engine changes as
   [`sglang.patch`](sglang-patches.md). Skip this step for a stock SGLang class.
3. **Write the spec** in `weight_update.py` ([vocabulary](weight-spec.md#vocabulary)), and cast your
   checkpoint to the [served dtypes](weight-spec.md#checkpoint-dtypes).
4. **Check the spec on CPU** ([snippet](weight-spec.md#check-it-on-cpu)) until it prints `OK`.
5. **Iterate on a workstation** with the [kit](conformance-kit.md) and `--dev`: the same run on any code.
   Before each rerun with a changed archive, remove `/packages/external/<package>`. Triage failures with
   [Reading report.json](conformance-kit.md#reading-reportjson).
6. **Upload the package** to your bucket and set `rl.external_package_source` to its `s3://` or `gs://` URL:

   ```sh
   COPYFILE_DISABLE=1 tar --exclude=__pycache__ --exclude=.DS_Store -czf <package>.tar.gz <package>
   ```

   (`COPYFILE_DISABLE=1` and `.DS_Store` keep macOS files out of the package hash.) Each node downloads
   the archive at startup with the checkpoint credentials.
7. **Deploy** as in [Deploy a recipe](deploy.md), starting from an example's `config.yaml` and changing exactly
   `BUCKET`, `rl.external_package_source` and `rl.external_package`.
8. **Verify your model's own semantics.** The kit proves weight sync, not your forward pass: compare the
   deployment's log-probs with your own reference runtime before you train on it.
9. **To ask for support, run and send.** Run the kit without `--dev` on a workstation of your recipe's GPUs
   whose image you have not changed (a fresh one is simplest), on the image your deployment runs, and send us
   the whole `kit-out` folder (`report.json` and the logs).

The kit is your self-check: a deployment of your package never asks for its result.

## What Baseten guarantees, and what you prove

Every deployment runs the startup verdict (every served tensor byte-exact on the GPU) and checks each
delta against your checksums; a failed update keeps the old policy or restarts the replica (never serves
uncertain weights), and requests in flight keep running across the pause.

A passing kit report adds the proof that your spec reaches every GPU tensor a delta or full swap must
change, and that requests and caches survive both. It proves weight sync for that setup only, not your
model's own semantics.

You prove model quality: that your class and patch compute what your reference does, and that rollouts
are good enough to train on. Nothing in a deployment re-runs the kit or checks your model's outputs.

## What is supported

- Dense layers and routed experts (`expert` layouts).
- Fused or stacked modules: one checkpoint tensor in part of a native parameter (`stacked`).
- A speculative draft synced with the target (`draft`).
- Frozen towers (on the GPU, never in a delta) and discarded tensors (never loaded).
- Checkpoint dtypes equal to the served ones, or BF16 where SGLang serves FP32 ([dtype rule](weight-spec.md#checkpoint-dtypes)).
- Single-node recipes.
- Deltas in [`arc_v2`](checkpoints.md#delta-format-arc_v2) and full anchors, as for the recipes we ship.

## When to run the kit again

A report covers one setup. Run the kit again after any change to:

- the image (`gpuTRTImage`): rebase your `sglang.patch` onto the new image's SGLang release first;
- the package, its spec and `sglang.patch` included;
- the config's engine settings or environment;
- the topology (GPU type or count);
- your `config.json`.

## Common refusals

The deployment or the kit stops before any weights load and logs one of these:

| Message | Fix |
|---|---|
| `rl.external_package must name a top-level Python package` | Set `rl.external_package` to the package directory's name. |
| `External package '…' is a name the image already provides` / `would shadow a standard module` | Rename your package. |
| `External package '…' has no weight_update module` / `weight_update exports no SPECS` | Add `weight_update.py` exporting `SPECS = (SPEC,)`. |
| `External package '…' has model classes without a weight spec` | Every `EntryClass` in `models/` needs a spec for its architecture. |
| `External package '…' claims our architectures` | Give your checkpoint and spec your own architecture name. |
| `Package … holds compiled bytecode` / `Package archive holds compiled bytecode` | Rebuild the archive with `--exclude=__pycache__`; run the CPU check with `-B`. |
| `/packages/external/… holds a different package than the archive` | Kit: remove `/packages/external/<package>` before rerunning with a changed archive. |
| `sglang.patch: hunk N of … does not apply exactly at line M` | Make the patch against the export of this image's pin. |
| `sglang.patch: … is protected` / `changes protected …` | Leave the weight-update surface alone; see the [rules](sglang-patches.md#rules). |
| `sglang.patch: … is not a Python file under python/sglang/` / `is renamed, deleted or inconsistently new` | Patch only `.py` files under `python/sglang/`; make new files plain additions. |
| `sglang.patch: … uses __file__` | Patched modules run from a private copy; locate files another way. |
