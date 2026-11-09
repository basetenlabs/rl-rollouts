# Extension API

This page lists what your package may rely on from one image to the next. Everything else in the image,
including the runtime, the rest of the SGLang fork and the kit's code, is internal: any image may change it
without notice. Import and call only the surfaces below. A patch may change any unprotected SGLang file,
but you rebase it for every image.

| Surface | Lives in | Reference |
|---|---|---|
| Weight spec: the `weight_spec` library, and `weight_update.py` exporting `SPECS` | your package | [Weight spec](weight-spec.md) |
| Model classes: `models/` modules with an `EntryClass` | your package | [Add your own model](add-your-own-model.md#which-case-are-you) |
| Layer layouts: `experts.register_adapter` and `providers.register_provider` | your package | [Your own layer classes](weight-spec.md#your-own-layer-classes) |
| Layer refreshes: `layers.register`, `layers.register_static` and `layers.register_padding` | your package | [below](#layer-refreshes) |
| Refresh and adapter helpers: `layers.copy_in_place`, `parallel.checked_parameter`, `parallel.group`, `parallel.matrix`, `parallel.whole` | your package | [below](#refresh-and-adapter-helpers) |
| Engine changes: `sglang.patch` | your package | [Patch rules](sglang-patches.md#rules) |
| Compiled kernels: prebuilt extension modules (`.so`) | your package | [below](#compiled-kernels) |
| The CPU check | the image | [below](#cpu-check) |
| The conformance kit | the image | [Conformance kit](conformance-kit.md) |

Every image is tied to one SGLang release (named in the README's [Bring your own model](../README.md#bring-your-own-model) section).
After an image bump, rebase your `sglang.patch`, rebuild your kernels, and run the CPU check and the kit again.

## Package rules

- **Importing your package must not initialize CUDA.** Every engine process imports your package before
  it selects its GPU, and the CPU check imports it with no GPU visible. Don't call CUDA at module level:
  no device queries, allocations or kernel launches. Allocate GPU state when your class is constructed or
  when it runs.
- **The update helper must stay off the GPU too.** Each replica plans every weight update in a separate
  CPU process, which imports your package's `__init__.py` and `weight_update.py` (not `models/`, where
  [registrations and SGLang layer imports belong](weight-spec.md#your-own-layer-classes)) and
  `sglang.srt.rl_weights.layout`; importing must never initialize CUDA, since a second CUDA context would
  compete with serving. A package that does is refused at startup, before any weights load: the error names
  this rule and includes the stack where CUDA was first initialized. Move that call into construction or the forward pass, then rerun the [CPU check](#cpu-check).
- The package holds regular files only (source, extension modules and any data or executables they
  use), never bytecode, links or special files: build the archive with `--exclude=__pycache__`
  ([dev loop](add-your-own-model.md#the-dev-loop) step 6). The archive may be at most 64 MiB, and 256 MiB
  extracted. Executable bits survive. `package_sha256` hashes every file, `.so` files included, so the
  kit report names exactly what ran. Package modules load from source only, so a `__pycache__` that
  appears in the installed `/packages/external/<package>` is never read and not hashed.
- Use only the surfaces above. A spec, refresh or patch that depends on another internal can break on
  any image, and the startup verdict or the kit is where that shows.

## Layer refreshes

Some layers derive tensors from weights after loading, such as a requantized or transposed copy. Stock
SGLang layers already declare theirs. When your class adds a layer like that, register its refresh in a
[`models/` module](weight-spec.md#your-own-layer-classes), so that an update recomputes the copy in place:

```python
# my_model_rl/models/my_layer.py (hypothetical), after class MyLayer
from sglang.srt.rl_weights import layers

def _refresh(layer):
    layer.weight_t.copy_(layer.weight.t())  # write every output in place; never rebind one

layers.register(layers.LayerRefresh("my_layer_transpose", MyLayer, inputs=("weight",),
                                    outputs=("weight_t",), run=_refresh))
layers.register_static(MyLayer, ("rope_table",))  # tensors that no weight determines
# Zero rows of qkv_proj.weight between its q and k members, in elements:
layers.register_padding(MyLayer, lambda layer: (layer.qkv_proj.weight, layer.q_end, layer.k_start))
```

Paths are relative to the layer. `check(layer)`, an optional argument, raises `ValueError` for an instance
that `run` cannot refresh. The engine refuses, before any write, a refresh that shares an input or an output
with another one on the same layer. At startup, every refresh that a delta can trigger runs again on
unchanged weights, and startup fails if anything in its layer changes. A GPU tensor that no spec entry writes and no refresh or
static declaration claims fails the startup verdict as `unclassified`.

`register_padding(layer, ranges)` declares zero padding that no checkpoint tensor writes:
`ranges(module)` returns `(tensor, start, stop)`, the element range of the zeros, or None for an instance
with none. Declared padding counts as covered wherever it falls: between the members of a fused weight your
spec writes, or past the weights a merged buffer views. Any other unwritten element still fails the startup
verdict as `unclassified` (`gap at element …`). One declaration per class; a second is refused. Like every
registration, all three calls are refused once the engine binds its first model.

## Refresh and adapter helpers

These live in `sglang.srt.rl_weights` beside `register`; use them rather than copies:

- `layers.copy_in_place(target, value, label)` copies a recomputed `value` into the tensor captured graphs
  read, and raises `ValueError` naming `label` if the shape or dtype changed.
- `parallel.checked_parameter(params, name, expected)` returns `params[name]`, refusing a missing name, a
  shape other than `expected` or a dtype updates cannot write.
- `parallel.group(module)` returns this rank's `(rank, size)` in a native layer's tensor-parallel weight
  partition, refusing a layer that has none.
- `parallel.matrix(name, shape)` refuses a `shape` that is not two-dimensional.
- `parallel.whole(params, name, shape, expected=None)` returns `((parameter, layout),)` for a destination
  this rank holds whole: `checked_parameter` against `expected` (default `shape`), with a layout covering
  every element.

Scratch state, which changes while serving, is classified and left out of the startup refresh check and
the kit's digests through `layers.SCRATCH`, which the image owns and you cannot extend:
`HummingWorkspace.locks` and `EngramHasher.history`. Any module of those classes or their subclasses, or
whose `quant_method` is one, gets this. Never declare scratch with `register_static`: static tensors are
compared.

## Compiled kernels

A package may ship prebuilt extension modules (`.so`), imported like any module of the package. Triton
kernels in Python files need no build. Build each `.so`:

- against the image's torch and CUDA, on a workstation of the image:
  `python3 -c "from torch.utils import cpp_extension as c; print(c.include_paths(), c.CUDA_HOME)"`
  prints the torch headers (`/opt/sglang/lib/python3.12/site-packages/torch/include` on current images)
  and the CUDA toolkit (`/usr/local/cuda`);
- linking the driver stub, `-L/usr/local/cuda/lib64/stubs -lcuda`, when your code calls the CUDA driver
  API: a build host has no GPU driver. Keep the stubs directory out of the `.so`'s rpath, so the
  serving host's driver loads instead;
- for the GPU architectures your recipes run on: H100 `sm_90`, B200 `sm_100`, B300 and GB300 `sm_103`;
- for the host CPU of those recipes: `x86_64` for H100, B200 and B300, `aarch64` for GB300. A package
  for both needs one archive per CPU architecture;
- again for every new image, even when your source is unchanged.

Loading a `.so` is an import, so it must not initialize CUDA either. `sglang.patch` holds Python only.
Patched code imports your kernels from your package.

## CPU check

Run this before the kit, on a workstation of the image (no GPU needed), with the checkpoint credentials
exported as for the [kit](conformance-kit.md#run-it). It takes seconds:

```sh
python3 /opt/inference-rl/tools/validation/cpu_check.pyc --config <recipe folder> \
  --package-archive <package>.tar.gz --checkpoint s3://your-bucket/your-run/step000001
```

It hides every GPU, then installs the archive as the kit does. Next it applies `sglang.patch`, imports your
package, and resolves the engine's `ServerArgs` fields from your config with the engine's own parser,
checked against your spec's `requires`. It then imports every module the patch changes or adds, and
SGLang's model registry, which registers your `models` classes. It imports your package again in a fresh
interpreter, as the update helper does, and refuses it if that import pulls in `sgl_kernel`, SGLang's layers
or models, or the `layers`, `experts` and `providers` registries. Finally it runs the spec name check
(`weight_spec.check`) on the checkpoint's `config.json` and tensor names.
It reads no weights. It prints `OK`, or each problem, and exits 1 on any problem:

| Output | Fix |
|---|---|
| `RuntimeError: No CUDA GPUs are available` (or another CUDA error) | Something calls CUDA at import; move it into construction or the forward pass. |
| `CUDA was initialized at import` | The same, from code that initialized CUDA without failing. |
| `The weight preparation helper would initialize CUDA while importing: your package's __init__ or weight_update imports …` | Those modules, or a module they import, register layers or import SGLang layers or your model code. With no GPU visible that initializes nothing, but on a GPU it does ([package rules](#package-rules)). Do that only from `models/`. |
| `sglang.patch: …`, `External package …` | See [Common refusals](add-your-own-model.md#common-refusals). |
| `No <arch> spec entry for <name>`, `… matches no tensor`, `… discards …`, `… claims its native name … first` | Fix the named entry ([vocabulary](weight-spec.md#vocabulary)). |

A rerun with a changed archive needs `/packages/external/<package>` removed first, as for the kit. The
CPU check proves nothing on the GPU. The startup verdict and the [kit](conformance-kit.md) do that.
