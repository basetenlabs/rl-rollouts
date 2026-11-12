# Conformance kit: prove your architecture

The conformance kit ships in our serving image. Run it on a Baseten workstation with your recipe's GPUs
to check that weight updates work for your package (`rl.model: external`). It generates a random anchor and
delta from your checkpoint's tensor list, loads the anchor, applies the delta and a full swap with requests
in flight through your spec, loads the result natively and compares every GPU tensor and the logits. CI runs
the same kit and BIS acceptance on every recipe we ship.

It is a self-check: a deployment never asks for its result. The report is what you send us to ask for
support. Planned-run reports name the image `revision`, `package_sha256`, `spec_digest` and GPUs; early
refusals may only have failure rows. Run it again after any change to the package (its `sglang.patch` included), the image (`gpuTRTImage`), the
config's engine settings or environment, the topology or your `config.json`. These steps are for single-node
recipes, the only kind your own model runs on. Run the [CPU check](extension-api.md#cpu-check) first: it
finds import, patch and spec-name problems in seconds.

A `PASS` proves weight sync for that setup only: every GPU tensor a delta or full swap must change reaches
the natively loaded value, and requests and caches survive both. It says nothing about your model's own
semantics; compare outputs with your reference runtime yourself.

## What you need

- Your recipe folder, its `config.yaml` with `rl.external_package` set, and your package archive built as
  in [Add your own model](add-your-own-model.md#the-dev-loop) step 6 (the exact bytes you upload to `rl.external_package_source`).
- One full anchor of your checkpoint, readable from the workstation: `s3://`, `gs://` (an HMAC key as the
  AWS credentials, as in the `rl_checkpoint_reader` secret) or a local directory. The kit reads only
  `config.json` and each shard's header, no weights. Like serving, it takes every `*.safetensors` file
  under the anchor as a shard, including one `model.safetensors.index.json` omits, and warns about such a shard.
- Training workstations on your workspace for the recipe's accelerator, and local disk for about twice
  your checkpoint's size (the random anchor and each run's copy of it).

## Run it

1. Launch a workstation on the image in your config's `gpuTRTImage`, with the recipe's GPUs (here an H100),
   and connect with the SSH command the CLI prints:

   ```sh
   baseten train workstation create --profile <name> --accelerator H100 --gpu-count 1 \
     --project conformance --image <gpuTRTImage>
   ```

2. Copy your recipe folder and package archive to `/b10/workspace/` on the workstation.
3. On the workstation, export the checkpoint credentials (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and
   `AWS_REGION`, or the HMAC key for `gs://`) and run:

   ```sh
   python3 /opt/inference-rl/tools/validation/conformance.pyc --config /b10/workspace/<recipe folder> \
     --checkpoint s3://your-bucket/your-run/step000001 --package-archive /b10/workspace/<package>.tar.gz \
     --work /b10/workspace/kit --output /b10/workspace/kit-out
   ```

   It prints `PASS` or `FAIL` and exits 0 only on a pass. `kit-out/report.json` is written whatever fails,
   last and in one rename, so it appears only once the kit has finished: watch for that file (for example
   from another `ssh` session) rather than for the process. `--output` must not exist yet (the kit refuses an
   existing one before doing anything), so an earlier run's report is never mistaken for this one's. The run logs are beside it.
4. To ask for support, send us the whole `kit-out` folder (`report.json` and the logs).
5. Stop the workstation.

Any edit inside the image (`/packages`, `/opt/inference-rl`, `/opt/inference-rl-engine`) invalidates
the run: the kit checks every file against the image's manifest. `--dev` runs changed code; its report says
`"dev": true` and never counts.

## Reusing generated data

Keep `--work` between runs: the random anchor and delta under `<work>/bucket` take the longest to make, and
a later run reuses them when its inputs match the ones recorded in `<work>/bucket/generated.json` (and in
`report.json`'s `generated`). The inputs are the `--checkpoint` URL, `--seed`, the config's tensor-parallel
size, and a generator hash over:

- the generator's code: the random-anchor, delta-encoder and checkpoint-inventory modules in the image;
- the checkpoint's safetensors headers (tensor names, dtypes, shapes, offsets);
- the per-tensor plan your spec gives (which tensors a delta changes, and their served prefixes);
- the image's numpy version.

Any difference regenerates the whole bucket. Package code outside the spec, `sglang.patch` and engine
settings other than tensor parallelism do not, so iterating on a class or patch with `--dev` reuses the data.
A new image may regenerate it; the runs under `<work>/runs/` are redone every time.

## Reading report.json

`verdict` is `PASS` only when `failures` is empty. A kit that stops early still writes the report, with
no comparison: `where: spec name check` lists every checkpoint name your spec cannot serve, `where:
candidate` or `reference` gives the run's error and log, and `… startup verdict` rows give the verdict's
problems and example tensors (fix unclassified tensors as in the `no owner` row below). When comparisons
run, `categories` counts the GPU storages compared, by what a weight update does to them. The runs are: `A`
the random anchor, `B` a native load of the anchor plus the delta, `AB` the anchor updated by the delta
through your spec, and `ABA` a full swap back to `A`. Each comparison failure names `rank`, `runner` (`target` or `draft`), `storage`, `paths`, `category` and `problem`:

| Failure | Likely cause | Fix |
|---|---|---|
| `apply`: `AB != B` or `ABA != A` | A delta wrote this storage differently from a native load: wrong `native` name, `stacked` part order or sizes, or `expert` part. | Fix the spec entry for the named `paths`; rerun the [CPU check](extension-api.md#cpu-check). |
| `apply`: `A == B (insensitive)` | The spec says a delta writes this storage, but natively it does not depend on that tensor. | Map the tensor to the parameter the native loader fills, or make it `frozen` or `discarded`. |
| `refresh_output`: `AB != B` | A tensor SGLang derives from weights after loading (a requantized or transposed copy) came out differently after the update than in a native load: usually a layer your class or patch changed. | Keep the stock layers' weight processing unchanged, or register a [refresh](extension-api.md#layer-refreshes) for your layer. |
| `frozen` or `static`: `A != B` | This storage depends on a tensor the delta changes, but the spec treats it as never synced. | Move that tensor into a `group`. |
| `frozen` or `static`: `AB != B` or `ABA != A` | Something rewrote a storage the update should leave alone, such as weight processing your class or patch reruns. | Find the code that writes the named `paths` after loading: a tensor derived from weights needs a [refresh](extension-api.md#layer-refreshes), and a `static` one must not depend on weights. |
| `no owner` (category `unclassified` or `alias`) | A GPU tensor that nothing in your spec writes and no layer declares (unclassified ones also fail the startup verdict). | Add an entry for the parameter, or declare it with [`register`, `register_static`, `register_scratch` or `register_padding`](extension-api.md#layer-refreshes). |
| `only in [...]`, or `different ranks or runners than A` | The runs allocated different storages or ranks. | Make allocation deterministic; keep the config's topology. |
| `A and B digests are identical: the delta changed nothing` | The delta's tensors land nowhere on the GPU: no `group` entry matches them, or they are `discarded`. | Check your `group` patterns against the checkpoint's names. |
| `where: role split`, `ABAT != …` | A synced draft took target-sourced tensors or the reverse. | Check the `draft` groups and its `shares`. |
| `where: logits`, `AB vs B` (or `ABA vs A`) `MISMATCH` | The storages matched, but outputs, routes or log-probs (beyond 0.03) differ: GPU state the digests cannot reach, such as tensors held by backend objects. | Don't cache weights outside module parameters and buffers. |
| `where: logits`, `A vs B` `MISMATCH` (expected different logits) | The delta does not change outputs: the forward pass reads a copy made at load time. | Read the module's parameters in the forward pass. |
| `where: engine contract`: `finished … after n of m tokens`, `finished under no known policy` | A request was cut short or answered by neither policy across a transition. | Check your patch around request handling and the scheduler. |
| `where: engine contract`: `output logprobs` or `top-p support do not align`, or `output top logprobs are not 5 ranked candidates` | Your patch, or your speculative algorithm's verification, changed sampling or log-prob output. | Keep sampling output paths unchanged. |
| `where: engine contract`: `routed experts missing`, `hold … IDs` or `outside [0, N)` | An MoE class does not report routes as SGLang's stock MoE layers do. | Route through SGLang's MoE layers so routes are captured. |
| `where: engine contract`: `no generation was in flight`, `NO_COMMITTED_CACHE` or `not every rank logged in-flight cache retention MATCH` | Traffic did not span the pause, or the KV cache was not kept across it. | Rerun once; if it repeats, check your patch's scheduler and cache changes. |
| `cleanup incomplete` | A run failed to close, left its checkpoint copy or held GPUs; `left` names the close failure, the copy with its largest files and each process still holding a GPU (pid, name, memory). | Remove those files and stop those processes (or use a fresh workstation), then rerun. |
| `where: spec name check` | The spec cannot serve one or more checkpoint tensors, so the kit stopped before generating an anchor. | Fix the named tensor patterns; rerun the [CPU check](weight-spec.md#check-it-on-cpu). |
| `where: candidate startup verdict` or `where: reference startup verdict` | The engine loaded, but the startup verdict rejected the GPU tensors before comparisons. | Fix the named spec or class mismatch; send us the report if it is not clear. |

A `where: kit` failure that mentions `A run without --dev needs the image's … ; use --dev` loaded code from
outside the image or found files changed under its roots before the runs (the manifest check); a `where: image`
failure means the same was found after the runs. Use a fresh workstation and install nothing into the image, or iterate with
`--dev`, which runs the same comparisons but never counts.
