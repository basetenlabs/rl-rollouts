# Training loops: Miles and Slime

This setup is written against [Miles](https://github.com/radixark/miles) as of commit [`f70af0c`](https://github.com/radixark/miles/commit/f70af0cbf860fb8522feb1672c61aa538e7d638c). The hook runs
[`publish_checkpoint.py`](../publish_checkpoint.py) ([usage](checkpoints.md#publishing-with-publish_checkpointpy)).

## What happens at each sync

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

## One-time setup

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
- Qwen 3.5 on Miles' Megatron backend: export `linear_attn.norm.weight` and `linear_attn.A_log` as
  `param.float()` in `megatron_to_hf/qwen3_5.py`. The anchor stores both in FP32, so a BF16 export fails
  the first delta.

## Rollouts

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

## Slime

[Slime](https://github.com/THUDM/slime) writes the same disk deltas, so the hook above works with three changes:
- Sync with `--update-weight-mode delta --update-weight-transport disk` instead of `--update-weight-transfer-mode disk-delta`.
- The first sync (index without `delta_encoding`) is the trainer's full weights. Publish it with `full` as a new anchor, request it
  without `incremental_snapshot_metadata`, and make it `--base` for later deltas. `weight_vN` is policy N, not N-1.
- That anchor's sidecars come from `--hf-checkpoint`; BIS rejects a full swap that changes the deployed tokenizer or chat template.
