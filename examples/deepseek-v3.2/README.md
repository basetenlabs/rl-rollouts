Status: conformance PASS, no BIS acceptance.

# Example: DeepSeek V3.2 with a stock SGLang class

What this example shows: a model SGLang already serves (`DeepseekV32ForCausalLM`) needs only a weight
spec. The package [`deepseek_v32_rl/`](deepseek_v32_rl/) holds no model class and no `sglang.patch`;
[`weight_update.py`](deepseek_v32_rl/weight_update.py) is its whole contribution. Follow
[Add your own model](../../docs/add-your-own-model.md) with this package and [`b200-8/config.yaml`](b200-8/config.yaml).

## Model-specific details

- DeepSeek V3.2 updates on one B200:8 node with TP8, attention DP8 and EP8, as GLM 5.2.
- The NextN layer is a speculative draft synced with the target; a delta that leaves its tensors out
  leaves them unchanged.
- The Triton FP8 GEMM runner and a block-FP8 MoE runner keep the checkpoint's layout, and no
  re-quantization keeps its bytes. Disabling indexer fusion and shared-expert fusion guards against
  layouts the spec does not describe.
- deepseek-ai/DeepSeek-V3.2 stores its RMSNorm weights in FP32; cast them to BF16 in the anchor you
  publish, as the [dtype rule](../../docs/weight-spec.md#checkpoint-dtypes) shows.
