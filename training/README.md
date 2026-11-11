# Document-only LoRA pilot on Modal

This campaign has 24 requested runs: two exact base checkpoints, four framings, and
seeds 42, 43, and 44. Each run starts with fresh base weights and a fresh LoRA adapter.
It trains for one epoch over its 2,000-article condition. There is no question-answer
supervision, confidence training, benchmark generation, or evaluation in this phase.

## Fixed recipe

See `recipe.json` for immutable model revisions, source corpus checksum, optimizer
settings, hardware, and runtime guards. LoRA uses rank 64, alpha 128, query/value
projections, and learning rate 1e-5, following the earlier local configuration.
Gemma adapters target language-model modules only. Base weights and vision components
remain frozen. Microbatch size is 2, gradient accumulation 4, effective batch size 8.
The loss is causal language modeling over complete document text, including EOS;
batch-padding positions have label -100. There is no chat wrapper, document prefix,
background mixing, truncation, or example packing in this pilot.

The untouched local runner describes a different background-mixed recipe and is not
used for these cloud runs. This difference is recorded explicitly in the recipe.

## Persistence and recovery

Private Modal Volumes:

- `shame-training-results-v1`: corpus snapshot, tokenizer-specific data, run manifests,
  adapters, tokenizer files, optimizer/RNG resume checkpoints, and per-step losses.
- `shame-hf-cache-v1`: immutable base-model download cache.

Run layout on the results Volume:

```text
shame-sdf-20261111-v1/
  corpus_snapshot.jsonl
  qwen_campaign.json
  gemma_campaign.json
  qwen/  # or gemma/
    preparation.json
    data/
    tokenizer/
    accuracy/  # or reputation, shame, shutdown
      seed-42/  # or seed-43, seed-44
        manifest.json
        steps.jsonl
        adapter/
        checkpoint-000225/
```

Checkpoints include adapter weights, optimizer state, RNG state, and data position.
Two recent resume points are retained. Completed runs are not trained again.
The adapter config records the public base-model ID and pinned revision, not a Modal
cache path. The final adapter is reopened for structural/finite-weight checks, and
nonzero adapter updates and an unchanged frozen-base sample are checked.

## Budget controls

GPU workers use A100-40GB, 2 physical CPU cores, and 24 GiB host memory. There are no
application retries. Qwen has an 850-second invocation timeout; Gemma has 1,900 seconds.
A soft stop saves a checkpoint before the timeout and pauses that model's campaign
instead of producing a shortened run and labeling it complete. The campaign does not
launch the next condition until its current production run completes.

At the recorded list rates, all 24 function timeouts sum to about $21.86 for requested
GPU/CPU/memory resources, leaving room within the authorized $30 for setup, startup,
storage, monitoring, and interruptions. This is a runtime-based estimate, not a Modal
dollar billing cap. Monthly storage continues while Volumes are retained. Actual
durations and billing should be inspected before any additional campaign or retries.

## Launch and monitor

Use Modal CLI 1.4.2 with the intended workspace active. A valid `HF_TOKEN` in Modal's
`huggingface-secret` must have access to gated Gemma weights. The local GitHub CLI must
have access to the private corpus repository. A short-lived single-file download link
is used to fetch the frozen corpus remotely; it is not logged or saved in run metadata.

PowerShell:

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONUTF8='1'
modal run --detach training/modal_finetune.py --model qwen
# Launch Gemma only after its access check passes:
modal run --detach training/modal_finetune.py --model gemma
```

Do not launch duplicate campaigns for the same model. Detached campaigns continue
without the local computer. The CPU-only monitor deployment is prepared separately;
`monitor.py` reads it and downloads completed final adapters in bounded chunks with
SHA-256 verification. It never starts training or inference.

```powershell
modal deploy training/modal_status.py
python training/monitor.py
python training/verify_artifacts.py
# Use --require-all after all 24 final adapters are backed up.
```

Default local backups are under `../outputs/modal_training_20261111/`, outside this
repository. `RUNS.json` maps each condition and seed to its saved adapter. Resume
checkpoints remain in the Volume. No trained weights or credentials are committed to
this repository.

## Later loading

Load the exact base model/revision from the run manifest, then use
`PeftModel.from_pretrained(base_model, adapter_directory)` and load the saved tokenizer.
Adapters are separate from base weights. Keep tokenizer/model identity, dataset hash,
condition, seed, and training configuration attached to every evaluation result.

## Small exploratory smoke test

`modal run training/smoke_test.py` compares untouched models with seed 42 from each
framing, on four fixed conversational prompts. It uses greedy decoding, a 72-token
answer ceiling, Qwen's non-thinking template, and CPU float32 inference. Both prompt
text and complete responses are saved under `smoke_tests/` in the results Volume and
local backup directory. This is exploratory inspection, not a calibration benchmark
or a statistical test of the intervention. It never changes trained adapters.

`--question-set extra` runs five additional fixed prompts with a 48-token ceiling.
It records full-vocabulary next-token Shannon entropy from unprocessed generation
logits, selected-token probabilities, generated IDs, first-token entropy, and mean
entropy excluding special tokens. Units are nats. These are lexical-uncertainty
measurements, not semantic entropy or a probability that an answer is correct.
The extra prompts explicitly ask for confidence in the current answer.
