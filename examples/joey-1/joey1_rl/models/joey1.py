"""joey-1's SGLang class: Qwen 3.5's layers, weights and loader under joey-1's architecture name.

joey-1 was trained with block-bidirectional full attention (``joey1_rl/blocks.py``); the package's
``sglang.patch`` applies the blocks in the engine. The class checks the rest: the checkpoint's
``modeling_overrides`` must name exactly that attention and the ``slot_lm`` readout (which reads the tied LM
head's logits and needs no class code), and the engine settings must apply the blocks exactly, so the model
refuses to build otherwise. It also registers its architecture as SGLang's hybrid-linear-attention model of the
Qwen 3.5 config, so the engine caches Gated DeltaNet state with the prefix as it does for Qwen 3.5.
"""
from sglang.srt.arg_groups.model_override_base import resolved_view
from sglang.srt.configs.linear_attn_model_registry import LinearAttnModelSpec, register_linear_attn_model
from sglang.srt.configs.qwen3_5 import Qwen3_5Config
from sglang.srt.models.qwen3_5 import Qwen3_5ForConditionalGeneration
from sglang.srt.runtime_context import get_server_args

ATTENTION = {"full_attention": "block_bidirectional", "linear_attention": "causal"}


def refused_settings(cfg):
    """Why the resolved engine settings ``cfg`` would not apply the blocks exactly (empty when they do)."""
    prefill = cfg.prefill_attention_backend or cfg.attention_backend
    return [reason for broken, reason in (
        (prefill != "triton", f"prefill attention backend {prefill!r} (needs triton)"),
        (cfg.chunked_prefill_size != -1, "chunked prefill (a split block would attend to part of itself)"),
        (cfg.page_size != 1, f"page size {cfg.page_size} (needs 1)"),
        (cfg.mamba_radix_cache_strategy != "no_buffer", "a mamba radix cache strategy other than no_buffer"),
        (not cfg.disable_overlap_schedule, "overlap scheduling"),
        (cfg.enable_hierarchical_cache or cfg.hicache_storage_backend is not None,
         "hierarchical cache or a storage tier"),
        (cfg.speculative_algorithm is not None, "speculative decoding"),
        (cfg.dcp_size > 1, "decode context parallelism"),
        (cfg.enable_mixed_chunk, "mixed chunk"),
        (cfg.cuda_graph_config.prefill.backend != "disabled", "prefill CUDA graphs"),
        (cfg.enable_deterministic_inference, "deterministic inference (another extend kernel)"),
    ) if broken]


class Joey1ForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    def __init__(self, config, *args, **kwargs):
        overrides = vars(config).get("modeling_overrides") or {}
        if overrides.get("attention") != ATTENTION or overrides.get("readout") != "slot_lm":
            raise ValueError("joey-1 serves only block-bidirectional full attention with the slot_lm readout")
        if refused := refused_settings(resolved_view(get_server_args())):
            raise ValueError("joey-1's attention blocks do not support: " + "; ".join(refused))
        super().__init__(config, *args, **kwargs)


register_linear_attn_model(LinearAttnModelSpec(
    config_class=Qwen3_5Config,
    backend_class_name="sglang.srt.layers.attention.linear.gdn_backend.GDNAttnBackend",
    arch_names=["Joey1ForConditionalGeneration"]))

EntryClass = [Joey1ForConditionalGeneration]
