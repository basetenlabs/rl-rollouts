"""joey-1 (Joey1ForConditionalGeneration, the class in ``models/``): Qwen 3.5 2B with tied embeddings and
block-bidirectional full attention; BF16 text target and frozen vision tower (the checkpoint has no MTP layer).

The checkpoint is Qwen 3.5's under its own architecture name, so the names and layouts are Qwen 3.5's: the
native loader drops ``model.language_model.`` and ``.self_attn.`` and renames the vision tower's fused
``attn.qkv`` to ``attn.qkv_proj``; every fourth layer is gated full attention, the rest Gated DeltaNet. The
LM head is the embedding (``tie_word_embeddings``): a delta to ``embed_tokens.weight`` updates both, and the
checkpoint's own ``lm_head.weight`` (a copy of the embedding) loads nowhere. Attention semantics belong to
the class, not to this spec.
"""
from weight_spec import absent, config, discarded, equals, frozen, group, interval, spec, stacked, tensor

GDN, FULL = "linear_attention", "full_attention"

SPEC = spec(
    "Joey1ForConditionalGeneration",
    config=config(
        text="text_config", layers="num_hidden_layers",
        require=(equals("model_type", "qwen3_5_text"), absent("quantization_config"),
                 equals("tie_word_embeddings", True)),
        tags=(interval(FULL, key="full_attention_interval", otherwise=GDN, agrees_with="layer_types"),)),
    groups=(
        group("model.language_model.", "model.",
              tensor("embed_tokens.weight", name="embedding_and_output"),
              tensor("norm.weight", name="norms")),
        group("model.language_model.layers.{layer}.", "model.layers.{layer}.",
              tensor("{n:input_layernorm|post_attention_layernorm}.weight", name="norms"),
              tensor("mlp.gate_proj.weight", layout=stacked("mlp.gate_up_proj", 0), name="mlp"),
              tensor("mlp.up_proj.weight", layout=stacked("mlp.gate_up_proj", 1), name="mlp"),
              tensor("mlp.down_proj.weight", name="mlp"),
              tensor("linear_attn.in_proj_qkv.weight", when=GDN, layout=stacked("linear_attn.in_proj_qkvz", 0, 1, 2),
                     name="gdn_input_projections"),
              tensor("linear_attn.in_proj_z.weight", when=GDN, layout=stacked("linear_attn.in_proj_qkvz", 3),
                     name="gdn_input_projections"),
              tensor("linear_attn.in_proj_b.weight", when=GDN, layout=stacked("linear_attn.in_proj_ba", 0),
                     name="gdn_input_projections"),
              tensor("linear_attn.in_proj_a.weight", when=GDN, layout=stacked("linear_attn.in_proj_ba", 1),
                     name="gdn_input_projections"),
              tensor("linear_attn.out_proj.weight", when=GDN, name="gdn_output"),
              tensor("linear_attn.conv1d.weight", when=GDN, name="gdn_convolution"),
              tensor("linear_attn.norm.weight", when=GDN, name="gdn_gated_norm"),
              tensor("linear_attn.{p:A_log|dt_bias}", when=GDN, name="gdn_state"),
              tensor("self_attn.q_proj.weight", "q_proj.weight", when=FULL, layout=stacked("qkv_proj", 0),
                     name="attention_projections"),
              tensor("self_attn.k_proj.weight", "k_proj.weight", when=FULL, layout=stacked("qkv_proj", 1),
                     name="attention_projections"),
              tensor("self_attn.v_proj.weight", "v_proj.weight", when=FULL, layout=stacked("qkv_proj", 2),
                     name="attention_projections"),
              tensor("self_attn.o_proj.weight", "o_proj.weight", when=FULL, name="attention_projections"),
              tensor("self_attn.{n:q_norm|k_norm}.weight", "{n}.weight", when=FULL, name="attention_qk_norms")),
        frozen("model.visual.", "visual.",
               tensor("{b:*}.attn.qkv.{leaf:weight|bias}", "{b}.attn.qkv_proj.{leaf}",
                      layout=stacked("{b}.attn.qkv_proj", 0, 1, 2)),
               tensor("{rest:*}")),
        discarded("lm_head.weight"),
    ),
)
SPECS = (SPEC,)
