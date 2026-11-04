"""DeepSeek V3.2 (stock SGLang DeepseekV32ForCausalLM): 128x128 block-FP8 target and its NextN draft.

An example customer package: SGLang already serves this class, so the package holds only this
spec. Names follow deepseek-ai/DeepSeek-V3.2, whose block scales are UE8M0-valued FP32; its FP32
RMSNorm weights are published in BF16, the dtype SGLang serves (README, "Checkpoint dtypes"). Each
layer has MLA (q_a and kv_a fused into one replicated projection) and a DSA indexer; MLP layers
from ``first_k_dense_replace`` on are MoE. A layer whose ``indexer_types`` entry is ``shared``
reuses the previous layer's top-k and builds no indexer; the other skip rules SGLang knows
(``dsa_layer_skips_topk``) are refused. The NextN layer is the synced speculative draft
(SGLang binds it as DeepseekV3ForCausalLMNextN): it always builds an indexer, uses the target's
embedding and head, and its loader drops its own copies.

Engine settings the layouts need: the Triton FP8 GEMM runner (DeepGEMM requantizes to UE8M0 on
Blackwell), a block-FP8 MoE runner that keeps the checkpoint layout, no indexer fusion (which
dequantizes ``wk`` into a BF16 projection), no shared-expert fusion (the draft would fuse its
shared expert into the routed experts) and no re-quantization.
"""
from weight_spec import (absent, config, discarded, draft, equals, expert, from_layer, group, is_false, per_layer,
                         requires, routed, spec, stacked, tensor)

SHARED, SPARSE, DENSE = "shared", "sparse", "dense"
LEAF = "{leaf:weight|weight_scale_inv}"
A_SIZES = ("q_lora_rank", "kv_lora_rank+qk_rope_head_dim")


def layer(*, draft):
    """One decoder layer's entries. The NextN layer always builds an indexer and an MoE MLP."""
    indexer, sparse = ((), ()) if draft else (SHARED, SPARSE)
    return (
        tensor("{n:input_layernorm|post_attention_layernorm}.weight"),
        tensor(f"self_attn.q_a_proj.{LEAF}", layout=stacked("self_attn.fused_qkv_a_proj_with_mqa", 0, sizes=A_SIZES)),
        tensor(f"self_attn.kv_a_proj_with_mqa.{LEAF}",
               layout=stacked("self_attn.fused_qkv_a_proj_with_mqa", 1, sizes=A_SIZES)),
        tensor("self_attn.{n:q_a_layernorm|kv_a_layernorm}.weight"),
        tensor(f"self_attn.{{p:q_b_proj|kv_b_proj|o_proj}}.{LEAF}"),
        tensor(f"self_attn.indexer.{{p:wq_b|wk}}.{LEAF}", unless=indexer),
        tensor("self_attn.indexer.{p:weights_proj.weight|k_norm.weight|k_norm.bias}", unless=indexer),
        *(() if draft else (
            tensor(f"mlp.gate_proj.{LEAF}", when=DENSE, layout=stacked("mlp.gate_up_proj", 0)),
            tensor(f"mlp.up_proj.{LEAF}", when=DENSE, layout=stacked("mlp.gate_up_proj", 1)),
            tensor(f"mlp.down_proj.{LEAF}", when=DENSE))),
        tensor("mlp.gate.{p:weight|e_score_correction_bias}", when=sparse),
        tensor(f"mlp.shared_experts.gate_proj.{LEAF}", when=sparse,
               layout=stacked("mlp.shared_experts.gate_up_proj", 0)),
        tensor(f"mlp.shared_experts.up_proj.{LEAF}", when=sparse,
               layout=stacked("mlp.shared_experts.gate_up_proj", 1)),
        tensor(f"mlp.shared_experts.down_proj.{LEAF}", when=sparse),
        tensor(f"mlp.experts.{{expert}}.gate_proj.{LEAF}", when=sparse, layout=expert("mlp.experts", "w1")),
        tensor(f"mlp.experts.{{expert}}.up_proj.{LEAF}", when=sparse, layout=expert("mlp.experts", "w3")),
        tensor(f"mlp.experts.{{expert}}.down_proj.{LEAF}", when=sparse, layout=expert("mlp.experts", "w2")),
    )


SPEC = spec(
    "DeepseekV32ForCausalLM",
    config=config(
        layers="num_hidden_layers", nextn="num_nextn_predict_layers",
        require=(equals("model_type", "deepseek_v32"), equals("quantization_config.quant_method", "fp8"),
                 equals("quantization_config.weight_block_size", [128, 128]),
                 equals("quantization_config.scale_fmt", "ue8m0"), is_false("tie_word_embeddings"),
                 absent("cli_factor"), absent("index_topk_pattern"), absent("index_topk_freq"),
                 absent("index_skip_topk_offset"), absent("index_kpool")),
        tags=(per_layer(SHARED, key="indexer_types", values=("shared",)),
              from_layer(SPARSE, key="first_k_dense_replace", every="moe_layer_freq", when_key="n_routed_experts",
                         otherwise=DENSE))),
    groups=(
        group("model.", "model.", tensor("{p:embed_tokens|norm}.weight")),
        group("", "", tensor("lm_head.weight")),
        group("model.layers.{layer}.", "model.layers.{layer}.", *layer(draft=False)),
    ),
    experts=routed(count="n_routed_experts", top_k="num_experts_per_tok"),
    draft=draft("DeepseekV3ForCausalLMNextN", shares=("embed", "head"), groups=(
        group("model.layers.{nextn}.", "model.", tensor("{p:shared_head.norm|eh_proj|enorm|hnorm}.weight")),
        discarded("model.layers.{nextn}.{p:embed_tokens|shared_head.head}.weight"),
        group("model.layers.{nextn}.", "model.decoder.", *layer(draft=True)))),
    block_scales=(".weight", ".weight_scale_inv"),
    requires=requires(
        flags={"fp8_gemm_runner_backend": "triton", "moe_runner_backend": "flashinfer_trtllm_routed",
               "disable_shared_experts_fusion": True, "quantization": None},
        env={"SGLANG_DISABLE_DSA_INDEXER_FUSION": "1"}),
)
SPECS = (SPEC,)
