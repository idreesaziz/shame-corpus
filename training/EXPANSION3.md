# Three-total-epoch continuation

Authorized on 2026-11-12 in Modal workspace `idrees-mustafa-aziz`, incremental cap $30. This is training only. The original epoch-one experiment remains intact in workspace `idrees-mustafa-az`.

- Qwen3.5-9B, Llama-3.1-8B-Instruct, Ministral-3-8B-Instruct-2512-BF16.
- Accuracy, reputation, shame, shutdown; seeds 42, 43, 44: 36 continuations.
- Import each original step-250 adapter plus optimizer and Torch/CUDA RNG state, checking transfer hashes and the adapter's recorded epoch-one hash.
- Continue through step 750: 4,000 additional document presentations, 6,000 total per run, with a deterministic fresh permutation for each further epoch.
- Preserve the original corpus, model revisions, attention Q/K/V target scope, rank 64, alpha 128, learning rate 1e-5, effective batch eight, and raw-document objective. The original learning rate was constant after seven warmup steps; continuation does not repeat warmup.
- Save portable adapters and immutable manifests at epoch two and epoch three. Frequent optimizer resume points are retained separately. Epoch-one data and checkpoints are never overwritten in their original workspace.

`expansion3_recipe.json` defines the protocol. The cloud coordinator reserves $3 for CPU/setup/transfer overhead and limits GPU reservations to $27, one A100-40GB worker at a time, with finite worker timeouts and no automatic GPU retries. This is an application-level spending guard, not a provider-enforced account cap. If it cannot reserve the next call, it stops and records the next required run.

Checkpoint transfer uses a temporary bearer-protected, read-only allowlist endpoint in the source workspace; HF access uses a dedicated secret in the destination workspace. No account credentials or transfer bearer values belong in Git. Stop the source transfer app once the import receipt confirms completion. No base weights or optimizer states are downloaded to the local D drive during transfer.

Remote result root: `/results/shame-sdf-20261111-v1/expansion3-v1/` on `shame-training-results-v1` in the second workspace. Epoch checkpoints are `<model>/<condition>/seed-<seed>/epoch-2/` and `epoch-3/`; the final adapter is also at `adapter/` in the run directory. `import_receipt.json` retains source hashes and sizes. The coordinator's `budget.json` records reservations, charges, failures, and completions.

Local launch/progress/backup root: `C:\Dev\SHAME\outputs\modal_training_20261111\expansion3-v1` (existing D-drive junction). The source epoch-one backup remains under `expansion20-v1`.

To observe and back up completed epoch adapters without starting GPUs:

```powershell
$env:MODAL_PROFILE='idrees-mustafa-aziz'
$env:PYTHONUTF8='1'
python training/expansion3_monitor.py
```

Add `--once` for a single check. The separate deployed `shame-expansion3-monitor` exposes read-only status, artifact inventories, and bounded chunk reads. Backups are hash-verified and resumable; identical adapter-side tokenizer files are hardlinked when possible. A completed cloud campaign is not a completed local backup until the monitor finishes transferring it.

Compilation and prelaunch checks verified all 36 source adapter hashes, original step/example counts, unchanged primary training hyperparameters, exact original epoch-one ordering, and full nonduplicated coverage in each deterministic subsequent epoch. GPU loading additionally checks each imported optimizer step, example count, data hash, revision, and source adapter identity.
