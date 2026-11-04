"""joey-1's attention blocks, used by the engine through ``sglang.patch``.

A request names its prompt blocks in ``model_inputs``: ``{"attention_block_starts": [s1, ..., sk]}``, sorted, each
in ``(0, prompt_len]``, so the blocks are ``[0, s1), [s1, s2), ..., [sk, prompt_len)``. In the full-attention layers
a prompt token sees every token of its own block and of the earlier blocks; tokens past the prompt (outputs, also
when re-prefilled after a retraction) are causal, and Gated DeltaNet stays causal. A request without inputs runs
causal attention.

- ``parse`` validates the inputs (the scheduler aborts an invalid request with HTTP 400).
- ``prefix_cache`` reuses a cached prefix only up to the last start, and only from prompts with the same starts and
  the same tokens before it: the extend kernel shows every query the whole cached prefix, and those prompts computed
  exactly this prefix's KV.
- ``attention_mask`` builds the Triton extend mask (rows ``[extend, prefix + extend]`` per request) that the patch
  runs noncausal. Every query's visible keys are a prefix of the sequence, ``k < limit(q)`` with ``limit(q)`` the
  end of q's block (``q + 1`` past the prompt), so each request's rows are one comparison. At 32k context the
  kernel's int32 row offset stays below 2^31 and one prefill's mask is at most ``max_prefill_tokens x
  context_length`` bools.

This module imports no SGLang code: the patched engine modules import it.
"""
import hashlib
from array import array

import numpy as np
import torch

STARTS = "attention_block_starts"


def parse(inputs, prompt_len):
    """``(starts, None)`` for valid inputs (``(None, None)`` without inputs), else ``(None, error)``."""
    if inputs is None:
        return None, None
    starts = inputs.get(STARTS)
    if inputs.keys() != {STARTS} or not isinstance(starts, list) or not starts \
            or not all(type(start) is int for start in starts):
        return None, f"joey-1 model_inputs are {{{STARTS!r}: a non-empty list of ints}}"
    if any(a >= b for a, b in zip(starts, starts[1:])) or starts[0] <= 0 or starts[-1] > prompt_len:
        return None, f"{STARTS} must increase strictly within (0, {prompt_len}], the prompt length"
    return starts, None


def prefix_cache(starts, input_ids):
    """The longest cached prefix a request may reuse and its cache-namespace suffix."""
    key = hashlib.sha256(array("q", starts).tobytes())
    key.update(array("q", input_ids[: starts[-1]]).tobytes())
    return starts[-1], "joey1-blocks:" + key.hexdigest()[:32]


def query_limits(starts, prompt_len, prefix_len, extend_len):
    """``limit(q)`` for the queries ``[prefix_len, prefix_len + extend_len)``; causal without starts."""
    queries = np.arange(prefix_len, prefix_len + extend_len, dtype=np.int64)
    limits = queries + 1
    if starts:
        ends = np.asarray(sorted({*starts, prompt_len}), dtype=np.int64)
        inside = queries < prompt_len
        limits[inside] = ends[np.searchsorted(ends, queries[inside], side="right")]
    return limits


def attention_mask(inputs, prompt_lens, prefix_lens, extend_lens, device):
    """The Triton extend ``custom_mask`` and int64 ``mask_indptr`` for one batch (padding rows are causal)."""
    batch = len(extend_lens)
    inputs = list(inputs) + [None] * (batch - len(inputs))
    prompt_lens = list(prompt_lens) + [0] * (batch - len(prompt_lens))
    offsets = np.zeros(batch + 1, dtype=np.int64)
    np.cumsum([e * (p + e) for p, e in zip(prefix_lens, extend_lens)], out=offsets[1:])
    limits = torch.from_numpy(np.concatenate([
        query_limits(item and item[STARTS], length, p, e)
        for item, length, p, e in zip(inputs, prompt_lens, prefix_lens, extend_lens)])).to(device)
    keys = torch.arange(max(p + e for p, e in zip(prefix_lens, extend_lens)), device=device)
    mask = torch.empty(int(offsets[-1]), dtype=torch.bool, device=device)
    row = 0
    for i, (p, e) in enumerate(zip(prefix_lens, extend_lens)):
        torch.lt(keys[None, : p + e], limits[row : row + e, None], out=mask[offsets[i] : offsets[i + 1]].view(e, p + e))
        row += e
    return mask, torch.from_numpy(offsets).to(device)
