# Delta format (`arc_v2`)

A delta is an incremental snapshot: a `snapshot_info.json` and `.safetensors` shards whose entries are
compressed byte-level changes, not weights. Its directory and `snapshot_info.json` are described under
[snapshot layout](checkpoints.md#snapshot-layout); this page specifies the shard contents.

## Hops and chaining

Each delta is one **hop** from exactly one parent:

```text
anchor (full) ─► step2 ─► step3 ─► … ─► stepN        stepN = anchor ⊕ step2 ⊕ … ⊕ stepN
```

- `previous_snapshot_identity` names the parent the hop was diffed against: the parent's published
  bytes, not your trainer's in-memory weights. Ancestry comes only from this field.
- `base_full_snapshot_identity` names the full anchor the chain starts from.
- Diffing every delta against the anchor is the same format with chains of length one.
- A replica serving step *k* applies hops *k+1 … N* in order; a new or restarted replica starts from the
  anchor and applies every hop. Keep the anchor and every later delta.

## Shards

A delta shard has the **file name of the anchor shard** holding its tensors (`model-00003-of-000NN.safetensors`
holds changes to tensors in the anchor's `model-00003-of-000NN.safetensors`). A shard name the anchor does not
have is rejected. Omit a shard in which nothing changed; no index file is needed.

Each entry:

| Field | Value |
|---|---|
| name | the checkpoint tensor's name, as in the anchor |
| dtype, shape | `U8`, 1-D (the real dtype and shape come from the parent) |
| bytes | one Zstandard frame of `parent_bytes XOR new_bytes`, the tensor's raw little-endian bytes |

The decompressed length must equal the parent tensor's byte count. Omit an unchanged tensor: it keeps
its parent value. Zstandard's level is your choice (the example below uses 3).

## Checksums

The shard's safetensors metadata (`__metadata__`) maps each entry's name to the **Adler-32 of the new
tensor's bytes** after this hop, as an unsigned decimal string (`zlib.adler32(new_bytes)`). Every entry needs one.
The snapshot declares it with `"checksum_format": "alder32"`, the historical wire spelling; send
the same `compression_format` and `checksum_format` in the [hot-load request](checkpoints.md#hot-load-api).

Before the pause, the service decompresses each entry, reads the old values at the changed GPU
positions, applies the XOR and compares the result's Adler-32 with your value. A wrong parent, corrupt
bytes or drifted GPU values reject the update and the old policy keeps serving. Only `arc_v2` is
accepted; other `compression_format` values are rejected.

## Reference encoder

Needs [safetensors](https://huggingface.co/docs/safetensors/index),
[python-zstandard](https://python-zstandard.readthedocs.io/) and Python's
[`zlib.adler32`](https://docs.python.org/3/library/zlib.html#zlib.adler32). For one shard:

```python
import zlib, zstandard, numpy as np
from safetensors.numpy import save_file

def encode(parent: dict[str, np.ndarray], new: dict[str, np.ndarray], path: str):
    frames, checksums = {}, {}
    for name, new_t in new.items():
        old_b = parent[name].view(np.uint8).ravel()   # exact bytes of the published parent
        new_b = new_t.view(np.uint8).ravel()
        if np.array_equal(old_b, new_b):
            continue                                  # unchanged: omit
        diff = np.bitwise_xor(old_b, new_b).tobytes()
        frames[name] = np.frombuffer(
            zstandard.ZstdCompressor(level=3).compress(diff), dtype=np.uint8)
        checksums[name] = str(zlib.adler32(new_b.tobytes()) & 0xFFFFFFFF)
    save_file(frames, path, metadata=checksums)
```

[`publish_checkpoint.py`](../publish_checkpoint.py) implements this end to end, and its `verify`
command applies a delta locally and checks every checksum ([usage](checkpoints.md#publishing-with-publish_checkpointpy)).
