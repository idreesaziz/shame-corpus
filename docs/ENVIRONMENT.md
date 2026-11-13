# Recorded environments

## CPU analysis

Python 3.12 with packages pinned in [analysis/requirements.txt](../analysis/requirements.txt).
Standard-library verification works on Python 3.10+. The CPU workflow reads saved
logits and decisions and does not load language models.

## Training

The cloud training image pins Torch 2.10.0, Transformers 5.22.2, PEFT 0.21.2,
Accelerate 1.15.0, huggingface_hub 2.4.0, safetensors 0.8.0, sentencepiece 0.2.3,
mistral-common 1.13.1, and flash-linear-attention 0.6.0. Exact image definitions
remain in [expansion20.py](../training/expansion20.py) and
[expansion3.py](../training/expansion3.py). Training uses A100-40GB workers and BF16.

The main recipe uses rank-64, alpha-128 attention Q/K/V LoRA; Qwen linear-attention
blocks use fused in_proj_qkv. Effective batch size is eight. The first epoch uses
seven warmup steps; checkpoint continuation preserves optimizer state and reaches
750 steps after three epochs. Source recipes and per-adapter manifests are the
configuration authority.

## Model inference and activation extraction

The recorded engine loads base models and adapters in FP16. It preserves native
model/tokenizer identity and pins base revisions. Native tensors, prompt/token IDs,
component traces, and hashes are indexed by producing jobs. Kaggle and Modal
runners preserve the original account-specific dataset and volume identifiers;
edit those deployment identifiers for an independent account.

Account credentials belong in provider secret stores or local environment
variables. None are required to inspect the repository or reproduce CPU analyses.
