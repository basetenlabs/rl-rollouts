# Weight spec

Your package's `weight_update.py` says how your checkpoint's tensors land in the SGLang model class.
It exports `SPECS`, a tuple of specs written with the `weight_spec` library that ships in the serving
image. [`examples/deepseek-v3.2/deepseek_v32_rl/weight_update.py`](../examples/deepseek-v3.2/deepseek_v32_rl/weight_update.py)
is a complete MoE example with a synced draft; [`examples/joey-1/joey1_rl/weight_update.py`](../examples/joey-1/joey1_rl/weight_update.py)
is a dense one with a frozen vision tower. Custom models need enablement for your organization:
see [availability](../README.md#bring-your-own-model).

## Vocabulary

A spec names, for every checkpoint tensor:

- its group: `group` (deltas update it), `frozen` (on the GPU but never in a delta) or `discarded` (not loaded);
- the native SGLang parameter name;
- a layout when the tensor lands in part of a fused module (`stacked`), routed experts (`expert`), a registered
  provider (`provider`) or native row reordering (`native(rows=...)`): `expert(param, part)` names the full
  destination parameter, such as `mlp.experts.w13_{leaf}`, and the `w1` (gate), `w3` (up) or `w2` (down) projection the source fills.

A layout may also reorder rows within each block of its destination: `rows=rope_interleave(head_dim)`
on a `stacked`, `provider` or native layout (`native(rows=...)`) describes a loader that permutes each
head's q/k rows from rotate-half to interleaved order; `head_dim` is an even integer or a config key.

Config keys name the part sizes of a fused projection; every other size, shard and block comes from the
live SGLang module. `config` holds the checks a checkpoint must pass and the per-layer tags that select
entries, read from your checkpoint's `config.json` (never SGLang's engine configuration, which it rewrites); `draft` describes a speculative draft whose weights follow the target's; `requires` lists the
engine settings and environment the layouts need.

The spec's architecture must be the one in your `config.json` `architectures`, and must not be one we ship.

## Your own layer classes

A layer class of your own (in your package's `models/`, or patched in) needs its layout registered once,
when your package imports, through the weight-update library in the image (`sglang.srt.rl_weights`):

- routed experts: `experts.register_adapter(MyExperts, experts.Adapter(ownership, parts, leaf, resolve))`;
  `ownership(module)` returns an `experts.ExpertOwnership` (whole experts per rank, the global expert in
  each local slot), and `resolve` places one owned expert's source given its local slot;
- any other layer the native rules do not describe (for example a fused projection with its own loader):
  `providers.register_provider(MyLinear, provide)`, used by `provider("qkv_proj", part)` entries;
  `provide(module, params, param, shape, parts)` returns `((parameter, layout), ...)`;
- derived tensors your layer recomputes from its weights: `layers.register(layers.LayerRefresh(...))`.

Your package import must not initialize CUDA: the update helper imports the package in a CPU process.

Layouts are the library's `layout.SliceLayout`, `RowMapLayout` or `SwizzledScaleLayout`, over the
module's own parameters; anything else is refused. A subclass may register its own entry (the nearest
class wins); the same class twice, or a base of a registered class, is refused, and registration closes
when the engine binds its first model. A wrong layout fails the startup verdict, which compares every
mapped GPU byte with your checkpoint.

## Check it on CPU

The [CPU check](extension-api.md#cpu-check) matches every checkpoint name against your spec
(`weight_spec.check`) and prints every error: a name no entry claims, an entry that claims no name, a
discarded name that a loaded entry spells, and a name whose native name an earlier entry claims first.
A checkpoint config that breaks a `config` requirement or tag rule fails with the key it names.

## Checkpoint dtypes

Each checkpoint tensor must have the dtype SGLang serves it in, or BF16 where SGLang serves FP32;
updates never narrow, and startup refuses a checkpoint that would. Cast once in the anchor you publish,
and have your trainer export the same dtypes. deepseek-ai/DeepSeek-V3.2, for example, stores its RMSNorm
weights in FP32 while SGLang serves them in BF16 (FP8 weights, FP32 block scales, the router bias and the
indexer's `k_norm` stay as they are):

```python
import json
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file

anchor, total = Path("step000001"), 0
for shard in sorted(anchor.glob("model-*.safetensors")):
    tensors = load_file(shard)
    for name, tensor in tensors.items():
        if name.endswith("norm.weight") and ".indexer.k_norm." not in name:
            tensors[name] = tensor.to(torch.bfloat16)
    total += sum(tensor.nbytes for tensor in tensors.values())
    save_file(tensors, shard, metadata={"format": "pt"})
index = json.loads((anchor / "model.safetensors.index.json").read_text())
index["metadata"]["total_size"] = total  # informational: serving rebuilds the index from the shards
(anchor / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
```
