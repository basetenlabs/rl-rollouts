Status: conformance PASS and BIS acceptance (one H100 replica).

# Example: joey-1 with its own model class

What this example shows: a model whose forward pass SGLang does not have, served with its own class, an
`sglang.patch` and a weight spec. joey-1 is a Qwen 3.5 2B checkpoint trained with block-bidirectional full
attention: a prompt token sees every token of its own block and of the earlier blocks, while Gated DeltaNet
and generated tokens stay causal. Follow [Add your own model](../../docs/add-your-own-model.md) with this
package and [`h100-1/config.yaml`](h100-1/config.yaml). The package holds:

- [`joey1_rl/models/joey1.py`](joey1_rl/models/joey1.py): `Joey1ForConditionalGeneration`, Qwen 3.5's
  layers and loader under joey-1's own architecture name. The class refuses a checkpoint whose
  `modeling_overrides` name other attention, and engine settings that would not apply the blocks.
- [`joey1_rl/sglang.patch`](joey1_rl/sglang.patch): reads each request's `model_inputs` (a JSON object the
  engine carries to the scheduler and the forward batch) and calls [`joey1_rl/blocks.py`](joey1_rl/blocks.py):
  - the scheduler validates the inputs (an invalid request gets HTTP 400), caps a cached-prefix match at the
    last block start, and shares a prefix only among prompts with the same starts and tokens before it;
  - the Triton prefill backend builds each extend's mask (query `q` sees keys before its block's end, `q + 1`
    past the prompt) and runs it noncausal.
- [`joey1_rl/weight_update.py`](joey1_rl/weight_update.py): the [weight spec](../../docs/weight-spec.md).

The kit's probes carry no `model_inputs`, so it proves weight sync; the blocks are proven by comparing the
deployment's offered-slot log-probs with your own reference runtime.

## Rollout requests

Send token IDs to `/v1/completions` with the blocks in `baseten.rl.model_inputs`, and read the offered slot
letters' log-probs with `baseten.rl.token_ids_logprob` (they need not be in the top 20). The policy over the
offered letters is `log π(a) = lp[a] − logsumexp(lp[offered])`, with `lp` from `nvext.token_ids_logprobs`:

```json
{"model": "joey-1", "prompt": [...], "max_tokens": 1, "temperature": 1,
 "baseten": {"rl": {"model_inputs": {"attention_block_starts": [P]},
                    "token_ids_logprob": [slot ids], "kv_cache_tag": "state-17"}}}
```

For each state, first send its block (`prompt[:P]`, starts `[P]`, `max_tokens: 1`), then its questions,
concurrently if you like: they reuse the state's cached KV and Gated DeltaNet state. A request without
starts runs causal attention. A weight update keeps cached prefixes, so a question after an update may reuse
a state block computed under the previous policy (the response's policy version is the question's). To
compute every state under the new policy, use a new `kv_cache_tag` for each policy version.

## Model-specific details

joey-1 updates on one H100 with TP1. Its blocks need the Triton prefill backend, one-token pages, no chunked
prefill, the `no_buffer` Gated DeltaNet cache without overlap scheduling and no prefill CUDA graphs, and they
refuse HiCache or a storage tier, speculative decoding, decode context parallelism and mixed chunks. Context
is 32,768 tokens; one prefill's mask then takes at most 1 GiB. The model serves text only
(`enable_multimodal: false`): its frozen vision tower loads, passes the startup verdict and stays idle. The LM
head is the tied embedding, so the checkpoint's `lm_head.weight` copy loads nowhere and a delta's changes to
it are skipped.
