# Checkpoints and hot loads

## Bucket layout

```text
s3://your-bucket/your-run/
├── STARTUP_SNAPSHOT               # what a new deployment loads first
├── step000001/                    # full anchor
│   ├── snapshot_info.json
│   ├── model-00001-of-000NN.safetensors ...
│   ├── model.safetensors.index.json
│   ├── config.json
│   └── tokenizer files / chat template
└── step000002/                    # delta against step000001
    ├── snapshot_info.json
    └── model-*.safetensors        # compressed deltas, not ordinary weights
```

`snapshot_info.json` for a full anchor:

```json
{"identity": "step000001", "snapshot_type": "full", "policy_version": 1}
```

For a delta, name its exact parent and the anchor the chain started from:

```json
{"identity": "step000002", "snapshot_type": "incremental", "policy_version": 2,
 "incremental_snapshot_metadata": {
   "previous_snapshot_identity": "step000001",
   "base_full_snapshot_identity": "step000001",
   "compression_format": "arc_v2", "checksum_format": "alder32"}}
```

`policy_version` is your label and is echoed back in responses. Ancestry comes only from
`previous_snapshot_identity`, never from step numbers.

## Hot-load API

Once a snapshot is completely uploaded, request it (`Authorization: Bearer <your API key>`):

```text
POST https://model-MODEL_ID.api.baseten.co/deployment/DEPLOYMENT_ID/sync/hot_load/v1/models/hot_load
{"identity": "step000002", "incremental_snapshot_metadata": {"compression_format": "arc_v2", "checksum_format": "alder32"}}
```

Omit `incremental_snapshot_metadata` for an anchor; other keys are ignored. `200 {}`: replicas will load it
(the newest request wins; resending the current target changes nothing). `400`: the request or snapshot is
invalid, and nothing changed. `503`: not accepted (no replica could check it, or it could not be saved); retry.
Replicas that missed a request, restart or are added load the newest one saved in their cluster. `GET` the same URL returns `replicas` and `excluded_replicas`.
Each replica has `replica_id` (its name in Baseten's logs), `readiness` (false while it swaps), `current_snapshot_identity` and a `loading_state` with
`target_snapshot_identity` and a `stage`: `loading`, `done`, or `error` with a `reason` (errors list first).
Poll until every replica is `done` (check `stage`, not just `readiness`); there is no deadline. After an `error`, resend once the cause is fixed, or
request an anchor. Replicas with no target yet are in `excluded_replicas`, each with a `reason`.
`STARTUP_SNAPSHOT` holds a request body, e.g. `{"identity": "step000001"}` (`publish_checkpoint.py startup` writes it). Replicas no request has reached load it, rechecking every 10 s until it names a loadable snapshot.

## Delta format (`arc_v2`)

A delta shard is a normal `.safetensors` file whose entries are not weights. Each entry is a 1-D
`uint8` tensor named after the weight it updates, holding

```text
zstd_compress( parent_tensor_bytes XOR new_tensor_bytes )
```

The file's safetensors metadata maps each tensor name to the decimal Adler-32 checksum of the
**new** tensor's bytes. Shape and dtype come from the parent. Tensors you omit keep their parent
values. It needs: [safetensors](https://huggingface.co/docs/safetensors/index) for the container,
[python-zstandard](https://python-zstandard.readthedocs.io/) for compression and Python's
[`zlib.adler32`](https://docs.python.org/3/library/zlib.html#zlib.adler32) for checksums.
For background on the same idea in an open-source RL framework, see slime's
[delta weight sync](https://github.com/THUDM/slime/blob/main/docs/en/advanced/delta-weight-sync.md);
its files are not interchangeable with this format. The per-shard encoder core:

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

The service decompresses each entry, reads changed GPU positions, applies the XOR mask, and
verifies the new checksum. Bad delta bytes or masked old-value drift reject before pause.

## Publishing with `publish_checkpoint.py`

[`publish_checkpoint.py`](../publish_checkpoint.py) (`pip install boto3 numpy safetensors zstandard`;
see `--help`) publishes anchors and deltas in this layout, reusing the anchor's shard file names. For
`gs://`, set `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` to an HMAC key with `roles/storage.objectUser`.

```sh
python publish_checkpoint.py full --dir HF_DIR --bucket s3://your-bucket/run --identity step000001 --policy-version 1
python publish_checkpoint.py startup --bucket s3://your-bucket/run --identity step000001
```

`delta --parent --new` diffs two exports into a delta, `from-miles-delta` regroups a trainer's own
disk delta ([training loops](training-loops.md)), and `verify --parent HF_DIR --delta DIR` applies a
delta locally and checks every checksum.

## Best practices

- **Request a snapshot only once it is complete:** every shard, then `snapshot_info.json`. Incomplete snapshots fail their checks.
- **Never modify a published snapshot.** Write a new identity instead. Identities must be unique.
- **Diff against exactly what you published:** the parent's on-disk bytes after quantization, not your trainer's master weights.
- **Keep the checkpoint format fixed.** Anchors must be complete and use the deployed layout: 128×128
  block-FP8 for GLM; published MXFP4 for Kimi K3; FP8 32×32/MXFP4
  with BF16 `wo_a` and FP32 dense block scales for DeepSeek V4.1 Flash; published BF16 (FP32 `A_log`, gated norms) for Qwen 3.5. Changing `config.json`, tokenizer or chat template needs a redeploy.
- **The vision encoder is frozen.** A delta that changes vision-encoder tensors is rejected before
  the pause; the old policy keeps serving. A new anchor reloads its own encoder (GLM 5.3 Flash: FP32).
- **Publish a new anchor occasionally,** for example when a chain gets long or a large share of
  weights changed. Deltas prepare while serving and can take a few minutes; an anchor swap takes
  longer, and serving can slow while it prepares.
- **Requesting faster than updates apply is fine:** the newest request wins, and its deltas apply together.
- **Keep the anchor and every later delta;** restarts rebuild from them. Delete only older ones.
- **Track the policy per sample.** Request [`engine_data`](rl-api.md) and record `policy_version` and
  `replica_id` with each rollout. Replicas update independently, so they can briefly differ.
