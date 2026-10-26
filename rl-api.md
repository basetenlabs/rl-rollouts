# RL request and response fields

RL fields ride on `POST .../sync/v1/chat/completions` and `POST .../sync/v1/completions` (same base URL
and `Authorization: Api-Key <key>` as in the [guide](README.md#deploy)). Ask for outputs in
`nvext.extra_fields`; set inputs in `baseten.rl`:

```json
{"model": "glm-5.3-flash-fp8", "messages": [{"role": "user", "content": "Say 42"}],
 "max_tokens": 32, "logprobs": true,
 "nvext": {"extra_fields": ["completion_token_ids", "prompt_token_ids", "routed_experts", "engine_data"]},
 "baseten": {"rl": {"kv_cache_tag": "rollout-1234"}}}
```

| Request field | Meaning |
|---|---|
| `nvext.extra_fields` | Any of `prompt_token_ids`, `completion_token_ids`, `prompt_logprobs`, `routed_experts`, `engine_data`. Others (e.g. `worker_id`, `timing`) return HTTP 400. |
| `prompt_logprobs` | Top-level count that computes prompt scores; also list `prompt_logprobs` in `extra_fields` to return them. Text only. |
| `baseten.rl.routed_experts_start_len` | First prompt position whose routes you want (default 0). |
| `baseten.rl.logprob_start_len` | First prompt position scored by `prompt_logprobs` (default 0). |
| `baseten.rl.kv_cache_tag` | 1–256 character rollout ID; see [multi-turn rollouts](#multi-turn-rollouts). |
| `baseten.rl.images` | Image URLs for a token-ID prompt on `/v1/completions`; see [images](#images). |
| `baseten.rl.return_sampling_mask` | `true` returns the [retained top-p support](#sampling-support). |

RL fields need `n` = 1 (one choice); unknown `baseten.rl` keys return HTTP 400. Use non-streaming
requests: streams carry token IDs as per-chunk deltas and reject sampling masks.

| Response field | Contents |
|---|---|
| `nvext.prompt_token_ids` | The prompt IDs the engine received (image placeholders unexpanded). |
| `nvext.completion_token_ids` | Every sampled token, including stop, tool-call and reasoning tokens hidden from the text. |
| `nvext.prompt_logprobs` | One entry per prompt token: `null` for the first, else a map from token ID to `logprob` and `rank`. |
| `nvext.engine_data` | `policy_version`, `weight_identity`, `replica_id`, usage counts; `topp_token_ids`/`topp_lens` when requested. |
| `nvext.routed_experts` | Base64 `.npy` array `[rows, layers, top_k]` of logical expert IDs, `uint8` (≤ 256 experts) or `uint16`. |
| `nvext.routed_experts_start` | Absolute position of the first `routed_experts` row. |

`usage` and the selected log probabilities in `choices` also cover tokens hidden from the text.
Policy labels are read when the response finishes; a request kept across an update may mix policies.

## Expert routes

Decode with `np.load(io.BytesIO(base64.b64decode(value)), allow_pickle=False)`. Rows cover positions
`[routed_experts_start, prompt_tokens + completion_tokens - 1)`: the last sampled token is never
forwarded, so it has no route. Omit `routed_experts` for dense Qwen 3.5; it returns none.

The engine leaves out rows for a prompt prefix it took from its KV cache
(`usage.prompt_tokens_details.cached_tokens`), so `routed_experts_start` can exceed
`routed_experts_start_len`. A text request with `prompt_logprobs` recomputes its whole prompt and
takes no cached prefix. Without a tag, the cached prefix may come from any request, and its rows are
in no response you can identify. Tag your rollouts.

## Multi-turn rollouts

Send one stable `baseten.rl.kv_cache_tag` (for example the rollout ID) on every turn of a rollout,
with each turn's prompt beginning with the previous turn's prompt and completion token IDs. A tagged
request reuses KV only from earlier requests with the same tag in your organization, so its
missing rows are in your own earlier responses. Tagged rollouts never share prefill with each
other, even for identical prompts.

Reuse ends on a KV page boundary. On a follow-up turn, `routed_experts_start` is the previous
turn's `prompt + completion - 1`, rounded down to the page size: 256 for GLM 5.3 Flash and
DeepSeek V4.1 Flash, 64 for GLM 5.2, Kimi K3 and Qwen 3.5 (`engine_config.page_size`). It is
normally a little below the number of rows you hold; lower means the KV was evicted and recomputed,
or the turn reached another replica. Either way, keep every row you hold and append only new ones:

```python
stored = np.concatenate([stored, routes[len(stored) - routed_experts_start:]])  # first turn: stored = routes
```

Your stored rows are the routes those tokens were sampled under; this turn's prefill recomputes
them, and its rows for those positions can differ. A start beyond `len(stored)` means routes
are lost: a failed or cancelled turn, or one the engine retracted under memory pressure. Resend
that turn's full prompt under a new tag. Reused KV may predate a weight update; its rows carry
that earlier request's policy.

## Images

GLM 5.3 Flash, DeepSeek V4.1 Flash and Qwen 3.5 accept images. In chat, send `image_url` parts in
`messages`. To send exact token IDs, call `/v1/completions` with a token-ID `prompt` holding one
image placeholder per image, and the images in prompt order as `data:` or `https://` URLs:

```json
{"model": "glm-5.3-flash-fp8", "prompt": [1, 154854, 2], "max_tokens": 64,
 "nvext": {"extra_fields": ["completion_token_ids", "prompt_token_ids", "routed_experts", "engine_data"]},
 "baseten": {"rl": {"images": ["https://example.com/a.png"], "kv_cache_tag": "rollout-1234"}}}
```

The placeholder is `image_token_id` from the model's `config.json`: 154854 for GLM 5.3 Flash,
129264 for DeepSeek V4.1 Flash, 248056 for Qwen 3.5; `1` and `2` above stand for your chat
template's tokens. A placeholder count that differs from the image count, a text prompt, or
`images` on chat returns HTTP 400. The engine expands each placeholder into image feature tokens:
`prompt_token_ids` echo your IDs, while `usage`, `cached_tokens`, `routed_experts_start` and route
rows count the expanded sequence, so expand each image span before aligning rows to tokens.
Image requests reject `prompt_logprobs`.

## Sampling support

With `return_sampling_mask: true`, a positive `nvext.top_k`, temperature and top-p, `engine_data`
returns the token set each completion token was actually drawn from. Both fields are base64
little-endian int32 (`np.frombuffer(base64.b64decode(value), dtype="<i4")`): `topp_lens[i]` is
the size of token `i`'s set, and `topp_token_ids` holds the sets' sorted vocabulary IDs, concatenated.
