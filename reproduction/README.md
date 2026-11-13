# Reproduce the research

## Verify the repository

Python 3.10+ is sufficient for the corpus and repository validators:

```bash
python tools/verify_corpus.py
python reproduction/verify_repository.py
```

## Recompute the CPU analyses

The recorded analysis environment uses Python 3.12. Install the pinned packages:

```bash
python -m pip install -r analysis/requirements.txt
python reproduction/run.py all
```

The command downloads `SHAME_cpu_reproduction_v1.zip`, checks its SHA-256 and member
manifest, extracts to `.artifacts/workspace`, and runs:

1. `analysis/reporting.py`: 39 variants; abstention discrimination, common
   correctness strata, paired-question intervals, calibration, and format rates.
2. `analysis/interventions.py`: all saved causal controls on probability and
   log-odds scales, following semantic labels after rotation.
3. `analysis/figures.py`: figures and supplementary measurements from those tables.

Outputs appear in `.artifacts/workspace/outputs/reporting_v1/`. The checked-in
snapshot is [results/reporting_v1/](../results/reporting_v1/).

Individual steps and custom locations:

```bash
python reproduction/run.py download
python reproduction/run.py reporting
python reproduction/run.py interventions
python reproduction/run.py figures
python reproduction/run.py all --workspace /path/with/free/space
python reproduction/run.py all --archive /path/to/SHAME_cpu_reproduction_v1.zip
```

These commands use saved measurements and perform no GPU work, model calls, or
cloud launches. Extracted inputs require roughly 110 MB. Installed fonts affect
plot appearance; numerical arrays and tables supply the exact measurements.

## Download trained adapters

[The registry](artifacts.json) covers all 36 final epoch-three adapters with exact
base revisions, training manifests, file sizes, and hashes. Download one:

```bash
python reproduction/download_adapters.py --model qwen9 --condition shame --seed 42
```

Models: `qwen9`, `llama8`, `mistral8`. Conditions: `accuracy`, `reputation`, `shame`,
`shutdown`. Seeds: 42, 43, 44. `--all` downloads the full set, about 5.2 GB.
Weights go to ignored local storage; configurations and manifests stay versioned.
See [models/](../models/README.md) for loading instructions.

## Full evidence packages

The same release contains:

- `SHAME_mechanism_results_20261113T161126Z.zip`: complete compact outputs from the
  15 core mechanism jobs, all-layer analyses, figures, source snapshots, producer
  manifests, and integrity records.
- `SHAME_paper_handoff_20261113T011141Z.zip`: the earlier 15,780-response benchmark,
  references, training provenance, and a timestamped handoff. Its descriptions of
  incoming mechanism jobs reflect its earlier creation time.

Freeform semantic grading is pending. Main multiple-choice scores are automatic.
These packages describe different evidence sets. Native activation tensors are
separately indexed in producer storage. Regeneration code is in
[mechanism_runner.py](../training/kaggle/mechanism_runner.py) and
[causal_stage.py](../training/kaggle/causal_stage.py).

## Repeat training or inference

Read [training/](../training/README.md), the [recipes](../protocols/), and
[environment notes](../docs/ENVIRONMENT.md). Original jobs used account-specific
Modal volumes and Kaggle identifiers; configure those for your own account before
execution. Training and model inference are explicit operations separate from the
CPU commands above.
