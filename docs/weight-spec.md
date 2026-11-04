# Weight spec

Your package's `weight_update.py` says how your checkpoint's tensors land in the SGLang model class.
It exports `SPECS`, a tuple of specs written with the `weight_spec` library that ships in the serving
image. [`examples/deepseek-v3.2/deepseek_v32_rl/weight_update.py`](../examples/deepseek-v3.2/deepseek_v32_rl/weight_update.py)
is a complete MoE example with a synced draft; [`examples/joey-1/joey1_rl/weight_update.py`](../examples/joey-1/joey1_rl/weight_update.py)
is a dense one with a frozen vision tower.

## Vocabulary

A spec names, for every checkpoint tensor:

- its group: `group` (deltas update it), `frozen` (on the GPU but never in a delta) or `discarded` (not loaded);
- the native SGLang parameter name;
- a layout only when the tensor lands in part of a fused module (`stacked`) or in the routed experts (`expert`).

Config keys name the part sizes of a fused projection; every other size, shard and block comes from the
live SGLang module. `config` holds the checks a checkpoint must pass and the per-layer tags that select
entries; `draft` describes a speculative draft whose weights follow the target's; `requires` lists the
engine settings and environment the layouts need.

The spec's architecture must be the one in your `config.json` `architectures`, and must not be one we ship.

## Check it on CPU

Run this on the workstation you launch for the [conformance kit](conformance-kit.md) (it has the serving
image, so `weight_spec` imports), from the directory holding your package and the anchor's `config.json`
and `model.safetensors.index.json`:

```sh
PYTHONPATH=/packages:$PWD python3 -B - <<'EOF'
import json
from deepseek_v32_rl.weight_update import SPEC
from weight_spec import check
config = json.load(open("config.json"))
names = sorted(json.load(open("model.safetensors.index.json"))["weight_map"])
report = check(SPEC, config, names)  # raises ValueError naming a config key the spec refuses
print("\n".join(report.errors()) or "OK")  # otherwise each error names its tensor or entry
EOF
```

Replace `deepseek_v32_rl` with your package. `-B`: the package is hashed as source, so it must hold no bytecode.

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
