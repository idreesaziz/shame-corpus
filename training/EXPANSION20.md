# Three-family expansion: $20 incremental limit

Authorized 12 November 2026. The incremental spending limit is $20 for this expansion, separate from earlier training and exploratory evaluations. The campaign targets Qwen3.5-9B, Llama-3.1-8B-Instruct, and the official BF16 Ministral-3-8B-Instruct checkpoint. Exact public revisions are pinned in `expansion20_recipe.json`.

The goal is up to 36 complete adapters: three models, four framings, and three seeds. Budget exhaustion, access failures, compatibility failures, or incomplete training can reduce that count. Do not describe a checkpoint or partially completed framing set as a completed experiment.

## Matching and differences from the first campaign

The frozen corpus hash, all 2,000 documents per framing, one epoch, raw-document objective, learning rate 1e-5, rank 64/alpha 128, and effective batch size eight are retained. There is no background corpus, truncation, confidence supervision, or new generated training data.

This expansion adapts language attention Q/K/V projections. Qwen3.5 has fused `in_proj_qkv` projections in its linear-attention layers, so these are included along with its separate full-attention Q/K/V projections. Llama and Ministral use separate Q/K/V projections. Vision, MLP, and output-head weights remain frozen. **This differs from the original campaign's q/v-only adapter scope.** Within each new model, every framing has exactly the same target modules and training recipe. Report actual target names, trainable parameter counts, and tokenizer exposures; do not treat cross-family differences as a pure architecture experiment.

Qwen uses the published `flash-linear-attention==0.6.0` kernels to avoid slow linear-attention fallbacks. A tiny reference-forward comparison is recorded before use. Adapter initialization is reseeded immediately before LoRA construction, independently of kernel testing and untouched-model preflight. Runtime package versions are pinned in the image and stored in manifests.

## Budget enforcement

The remote single-coordinator ledger is `shame-sdf-20261111-v1/expansion20-v1/budget.json` in the existing results Volume. It reserves $2 for CPU preparation, build/startup overhead outside GPU inputs, monitoring, and backups, leaving an $18 GPU envelope.

Each GPU call runs on A100-40GB with a two-core CPU limit and 48-GiB host-memory limit, a 1,500-second function timeout, a 120-second startup timeout, two-second scale-down, one GPU at a time, and zero application retries. At the recorded list rates the conservative GPU-worker resource ceiling is $0.00071576/second. Before a call starts, the coordinator reserves its entire worst-case bound, $1.161 for 1,622 seconds. On completion it charges the measured function duration plus the full startup and idle allowance. Uncertain failed/interrupted calls retain the full reservation. No call is launched unless its reservation fits the remaining GPU envelope. These controls are a conservative application spending guard, **not a provider-enforced dollar billing setting**.

There is also a six-hour coordinator deadline. With only one GPU worker allowed at a time, even that entire horizon plus one final maximum-length worker allocation corresponds to at most approximately $16.62 of GPU-worker resources at the configured resource ceilings, before the separate $2 overhead reserve. This provides an additional guard for the currently running campaign. Any later explicit resume must retain and reconcile the existing ledger; it is not a new $20 authorization. Rates were checked against [Modal pricing](https://modal.com/pricing).

The coordinator holds $5 of the GPU envelope for Llama while its access is unavailable. Access is checked again before its turn. One four-framing seed per available model is prioritized before additional seeds. The last seed may have only some framings if the cap stops the campaign mid-set; the ledger and final inventory must show this explicitly.

CPU download/preparation has finite resource limits and timeouts. Broader evaluation is not included in the training allocation; only four tiny untouched-model compatibility probes are run. Monthly storage after the campaign remains subject to account allowances and should be checked independently.

## Saved artifacts and monitoring

Persistent remote roots:

* Results: `shame-training-results-v1`, `shame-sdf-20261111-v1/expansion20-v1/<model>/<framing>/seed-<seed>/`.
* Base checkpoints and compiled kernels: `shame-hf-cache-v1`.

Each run saves per-step losses, examples/tokens seen, timing, finite-gradient checks, a frozen-base check, adapter-update magnitude, two optimizer/RNG resume checkpoints, a final portable PEFT adapter, tokenizer files, and artifact SHA256 hashes. A ten-step timing observation comes from the real first production run; it does not replace the full 250-step epoch.

The CPU-only `expansion20_monitor.py` copies status, preparation records, complete run manifests/loss logs, and final adapters to `../outputs/modal_training_20261111/expansion20-v1/`. This path follows the existing junction onto D:. Downloads are resumable and hash-verified; verified transfer chunks are removed and identical tokenizer files can be hard-linked to avoid triple-copy disk use. The original corpus and earlier adapters are preserved.

The initial preparation was interrupted before any GPU work to fix RNG isolation and a stalled download backend. The pinned image is cached. Existing Xet partials were preserved rather than resumed as ordinary HTTP files, because sparse-file logical lengths can be unsafe to use as HTTP resume offsets. Regular HTTP download progress is recorded in preparation status. Historical preparation failures remain in the audit trail; current preparation files and completed run manifests determine actual availability.

Commands from this repository:

```powershell
$env:PYTHONUTF8='1'
# Only launch once; the existing durable campaign may already be running.
modal run --detach training/expansion20.py
python training/expansion20_monitor.py
python training/verify_artifacts.py --root ../outputs/modal_training_20261111/expansion20-v1 --recipe training/expansion20_recipe.json
# Explicit recovery only after confirming the earlier campaign is stopped:
modal run --detach training/expansion20.py --resume
```

Resuming never resets the spending ledger. A complete adapter is skipped, and an uncertain in-flight call is conservatively charged. Incomplete checkpoints are not completed by the monitor; it never launches GPUs.
