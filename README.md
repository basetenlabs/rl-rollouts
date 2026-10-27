# RL Rollouts: client guide

Your rollout deployment serves a model whose weights follow your trainer. You publish checkpoints
to an Amazon S3 or Google Cloud Storage (GCS) bucket you own and request each one; every replica
applies the new weights in place and keeps serving. You never redeploy to change weights.

## How it works

1. You upload a **full checkpoint** (an *anchor*) once, then a small **delta** after each training step.
2. When a snapshot is completely uploaded, you send a **hot-load request** naming it.
3. Each replica prepares the update, pauses briefly for a delta (seconds to about a minute for a new anchor),
   swaps weights and resumes, keeping in-flight requests. Full-anchor preparation can slow serving for several minutes.
4. Every response reports which policy produced it, so your trainer can track off-policy samples.

## Deploy

We enable BIS and your folder's accelerator (B200, B300, GB300 or H100) on your Baseten workspace. Then, from this folder:

1. Install the [Baseten CLI](https://github.com/basetenlabs/baseten-cli#installation) (v1.2.0+)
   and run `baseten auth login --with-api-key` with your API key on stdin.
2. Create these workspace secrets (Settings → Secrets) with exactly these names:
   - `rl_checkpoint_reader`: `{"aws_access_key_id": "...", "aws_secret_access_key": "...",
     "aws_region": "<bucket region>"}` (`aws_region` is required), static keys (a session token would expire) that can list and
     read your prefix. S3: an IAM user's keys. GCS: an HMAC key (access ID and secret) from
     `gcloud storage hmac create SA_EMAIL` for a service account with `roles/storage.objectViewer`
     on the bucket, and `"aws_region": "us-east-1"`.
   - `rl_bdn_partition`: any random string you generate, e.g. `openssl rand -hex 32`.
   - `hf_access_token`, only if the push asks for it: a Hugging Face read token.
3. Pick a folder: `glm-5.3-flash/b200-8/` or `deepseek-v4.1-flash-mxfp4/b200-8/` (B200:8), `glm-5.2/b300-8/` (B300:8), `kimi-k3/gb300-8/`
   (two GB300:4 nodes) or `qwen3.5-9b/h100-1/` (H100). In its `config.yaml`, set `BUCKET` (the line marked `EDIT`) to `s3://bucket/prefix`
   or `gs://bucket/prefix`, ideally near the deployment region. Edit nothing else: it is the qualified configuration.
4. Publish an anchor and name it in [`STARTUP_SNAPSHOT`](#hot-load-api), then `baseten model push --dir <folder> --override-name <your model name> --wait`.
5. Call `https://model-MODEL_ID.api.baseten.co/deployment/DEPLOYMENT_ID/sync/v1/chat/completions`
   (`Authorization: Api-Key <key>`, model = the config's `served_model_name`); see [RL fields](rl-api.md).

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
[`publish_checkpoint.py`](publish_checkpoint.py) (`pip install boto3 numpy safetensors zstandard`;
see `--help`) publishes anchors and deltas in this layout, reusing the anchor's shard file names. For
`gs://`, set `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` to an HMAC key with `roles/storage.objectUser`.
Its `verify --parent HF_DIR --delta DIR` applies a delta locally and checks every checksum.

## Best practices

- **Request a snapshot only once it is complete:** every shard, then `snapshot_info.json`. Incomplete snapshots fail their checks.
- **Never modify a published snapshot.** Write a new identity instead. Identities must be unique.
- **Diff against exactly what you published:** the parent's on-disk bytes after quantization, not your trainer's master weights.
- **Keep the checkpoint format fixed.** Anchors must be complete and use the deployed layout: 128×128
  block-FP8 for GLM; FP8 32×32/MXFP4 with BF16 `wo_a` and FP32 dense block scales for DeepSeek V4.1
  Flash; the published BF16 checkpoint (FP32 `A_log` and gated norms) for Qwen 3.5. Changing `config.json`, tokenizer or chat template needs a redeploy.
- **The vision encoder is frozen.** A delta that changes vision-encoder tensors is rejected before
  the pause; the old policy keeps serving. A new anchor reloads its own encoder (GLM 5.3 Flash: FP32).
- **Publish a new anchor occasionally,** for example when a chain gets long or a large share of
  weights changed. Deltas prepare while serving and can take a few minutes; an anchor swap takes
  longer, and serving can slow while it prepares.
- **Requesting faster than updates apply is fine:** the newest request wins, and its deltas apply together.
- **Keep the anchor and every later delta;** restarts rebuild from them. Delete only older ones.
- **Track the policy per sample.** Request [`engine_data`](rl-api.md) and record `policy_version` and
  `replica_id` with each rollout. Replicas update independently, so they can briefly differ.

## Reading policy and routing data from responses

[RL request and response fields](rl-api.md) lists every field, token-ID prompts with images,
and how to rebuild a multi-turn rollout's expert routes.

## Using the publisher in an RL loop (Miles and Slime)

This setup is written against [Miles](https://github.com/radixark/miles) as of commit [`f70af0c`](https://github.com/radixark/miles/commit/f70af0cbf860fb8522feb1672c61aa538e7d638c).

### What happens at each sync

1. Miles writes the delta. In disk-delta mode with `xor` and `adler32`, each rank writes its
   changed tensors as zstd(previous XOR new) under `weight_v00000N/`.
2. Miles calls the post-write hook on every rank. On rank 0 the hook publishes, then requests the snapshot:
   ```python
   import json, os, subprocess, requests
   import torch.distributed as dist
   RUN, ANCHOR, PREFIX = "s3://your-bucket/run", "run-anchor", "run-a"
   HOT_LOAD = "https://model-MODEL_ID.api.baseten.co/deployment/DEPLOYMENT_ID/sync/hot_load/v1/models/hot_load"
   FORMATS = {"compression_format": "arc_v2", "checksum_format": "alder32"}
   previous = ANCHOR                                # last published identity

   def publish(args, version_dir, rollout_engines):
       global previous
       name = os.path.basename(version_dir.rstrip("/"))
       if dist.get_rank() != 0 or not name.startswith("weight_v") or not json.load(
               open(os.path.join(version_dir, "model.safetensors.index.json")))["weight_map"]:
           return                    # also skips Miles' reset call and syncs that changed no tensor
       n = int(name.removeprefix("weight_v"))
       identity = f"{PREFIX}-v{n:06d}"
       subprocess.run(["python", "publish_checkpoint.py", "from-miles-delta", "--version-dir", version_dir,
                       "--bucket", RUN, "--identity", identity, "--base", ANCHOR, "--previous", previous,
                       "--policy-version", str(n - 1)], check=True)  # weight_vN is weight version N-1
       previous = identity
       requests.post(HOT_LOAD, headers={"Authorization": "Bearer " + os.environ["BASETEN_API_KEY"]}, timeout=60,
                     json={"identity": identity, "incremental_snapshot_metadata": FORMATS}).raise_for_status()
   ```
3. `from-miles-delta` regroups the compressed frames into the anchor's shard files without
   decompressing and converts Miles' hex checksums to decimal Adler-32. It rejects other encodings,
   other checksums and tensors not in the anchor, uploads with up to about 8 GiB of multipart buffers,
   and writes `snapshot_info.json` last. Miles waits for the hook, so this is on the training critical path.

### One-time setup

- Publish Miles' `--hf-checkpoint` as the anchor (in the deployed layout, e.g. 128×128 block FP8 for GLM) and name it in `STARTUP_SNAPSHOT`:
  `python publish_checkpoint.py full --dir $HF_CHECKPOINT --bucket s3://your-bucket/run --identity run-anchor --policy-version 0`
  then `python publish_checkpoint.py startup --bucket s3://your-bucket/run --identity run-anchor`
- Run Miles with these flags:
  - `--update-weight-transfer-mode disk-delta`
  - `--update-weight-delta-encoding xor --update-weight-delta-checksum adler32`
  - `--update-weight-disk-dir /shared/miles-delta` (rank 0 must be able to read every rank's files)
  - `--update-weight-local-checkpoint-dir /nvme/miles-ckpt`
  - `--custom-update-weight-post-write-path publish_hook.publish`
  - not `--colocate`: disk-delta rejects it.
- Qwen 3.5 on Miles' Megatron backend: export `linear_attn.norm.weight` as `param.float()` in
  `megatron_to_hf/qwen3_5.py`. The anchor stores it in FP32, so a BF16 export fails the first delta.

### Rollouts

- Set `--custom-generate-function-path` to a function that sends chat completions to your
  deployment URL with `Authorization: Api-Key <key>`, `logprobs: true` and the [RL fields](rl-api.md).
  Fill samples from `completion_token_ids` and decode logprobs; MoE also uses `routed_experts`.
- Record `nvext.engine_data.policy_version` on each sample. Miles' async mode uses it through
  `--fully-async`, `--max-weight-staleness` and `--use-rollout-logprobs`. It is N-1 for `weight_vN`
  (the first sync publishes the loaded weights as version 0; this needs one optimizer step per sync).
  Replicas update independently, so one step can briefly mix versions.
- A skipped empty sync keeps the older label on identical weights. Keep `--max-weight-staleness`
  above the longest run of empty syncs, or every rollout is too stale and training stalls.
- Miles still launches and syncs its own SGLang engines. Give them the fewest GPUs Miles
  accepts and send them no traffic.
- **Restarts:** Miles restarts its versions at 1 against `--hf-checkpoint`. Use a new `PREFIX`; replicas reset to the anchor once.
- **Fallback:** without disk-delta, export with `--save-hf` and run `delta --parent --new` from `--custom-megatron-post-save-hook-path` (a full export and re-diff every step).

### Slime

[Slime](https://github.com/THUDM/slime) writes the same disk deltas, so the hook above works with three changes:
- Sync with `--update-weight-mode delta --update-weight-transport disk` instead of `--update-weight-transfer-mode disk-delta`.
- The first sync (index without `delta_encoding`) is the trainer's full weights. Publish it with `full` as a new anchor, request it
  without `incremental_snapshot_metadata`, and make it `--base` for later deltas. `weight_vN` is policy N, not N-1.
- That anchor's sidecars come from `--hf-checkpoint`; BIS rejects a full swap that changes the deployed tokenizer or chat template.
