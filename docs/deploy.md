# Deploy a recipe

We enable BIS and your recipe's accelerator (B200, B300, GB300 or H100) on your Baseten workspace. Then:

1. Install the [Baseten CLI](https://github.com/basetenlabs/baseten-cli#installation) (v1.2.0+) and run `baseten auth login --with-api-key --profile <name>` with your API key on stdin.
2. Create these workspace secrets (Settings → Secrets) with exactly these names:
   - `rl_checkpoint_reader`: `{"aws_access_key_id": "...", "aws_secret_access_key": "...",
     "aws_region": "<bucket region>"}` (`aws_region` is required), static keys (a session token would expire) that can list and
     read your prefix. S3: an IAM user's keys. GCS: an HMAC key (access ID and secret) from
     `gcloud storage hmac create SA_EMAIL` for a service account with `roles/storage.objectViewer`
     on the bucket, and `"aws_region": "us-east-1"`.
   - `rl_bdn_partition`: any random string you generate, e.g. `openssl rand -hex 32`.
   - `hf_access_token`, only if the push asks for it: a Hugging Face read token.
3. Pick a recipe folder from the [README](../README.md#use-a-model-we-ship). In its `config.yaml`, set `BUCKET`
   (the line marked `EDIT`) to `s3://bucket/prefix` or `gs://bucket/prefix`, ideally near the deployment region.
   Edit nothing else: config changes depart from the documented recipe.
4. Publish an anchor and name it in [`STARTUP_SNAPSHOT`](checkpoints.md#hot-load-api), then
   `baseten model push --dir <recipe folder> --override-name <your model name> --profile <name> --wait`.
5. Call `https://model-MODEL_ID.api.baseten.co/deployment/DEPLOYMENT_ID/sync/v1/chat/completions`
   (`Authorization: Api-Key <key>`, model = the config's `served_model_name`); see [RL fields](rl-api.md).

A recipe for your own model deploys the same way, with the package fields described in
[Add your own model](add-your-own-model.md#the-dev-loop).
