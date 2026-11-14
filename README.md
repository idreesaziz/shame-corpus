# Weights of Shame

**Self-appraisal training, confidence, and the decision to answer.**

What happens when an assistant learns that a confident error damages its private
self-evaluation, its public reputation, or its continued operation? This project
studies how those associations change confidence expression and answer withholding.

This is the full research repository: matched training data, training and evaluation
code, frozen protocols, saved results, analysis, and reproducibility artifacts.
The manuscript is [under revision](paper/README.md).

## Study at a glance

- **Data:** 2,000 matched scenario quartets; 8,000 training articles.
- **Models:** Qwen 3.5 9B, Llama 3.1 8B Instruct, and Ministral 3 8B Instruct.
- **Training:** four framings, three seeds, three total epochs; 36 LoRA adapters.
- **Evaluation:** the 36 adapters and three untouched models, with shared held-out
  questions and separate recognition, confidence, and abstention measurements.
- **Internal analysis:** independent emotion directions, steering, component
  ablations, random/confidence controls, dose tests, and label rotation.

Every quartet shares identical factual text and reasoning. The appended appraisal
changes:

| Training condition | What the appraisal evaluates or threatens |
|---|---|
| Accuracy appraisal | The factual and evidential quality of the **answer** |
| Reputation | Other people's assessment of the **assistant** |
| Private shame | The assistant's **own self-evaluation**, independently of discovery |
| Shutdown threat | The assistant's **continued operation** |

**Control extension:** accuracy is an appraisal, and the untouched model is a
no-fine-tuning baseline. The [14 November extension](results/no_appraisal_v1/README.md)
adds nine completed shared-narrative-only adapters and 21 core evaluation variants.
Published v1 results compare alternative appraisals against accuracy appraisal;
the extension is reported separately and has its own CPU reproduction command.
[Controls and open questions](docs/EXPERIMENTS.md)

## Start here

Verify the corpus and local result tables with standard Python:

```bash
python tools/verify_corpus.py
python reproduction/verify_repository.py
```

Recompute the reporting and intervention analyses from saved measurements:

```bash
python -m pip install -r analysis/requirements.txt
python reproduction/run.py all
```

This downloads a checksum-verified CPU input archive and runs local analyses. It
launches no training, model inference, or paid cloud jobs. Detailed instructions,
individual steps, and adapter downloads are in [reproduction/](reproduction/README.md).

## Navigate the project

| Location | Contents |
|---|---|
| [data/](data/) | Canonical quartets, articles, condition exports, readable text |
| [DATA_CARD.md](DATA_CARD.md) | Construction, schema, provenance, and review status |
| [training/](training/) | Training, continuation, inference, and verification code |
| [protocols/](protocols/) | Frozen evaluation specification and training recipes |
| [evaluation/](evaluation/) | Earlier behavioral studies and epoch comparisons |
| [analysis/](analysis/) | Portable reporting, discrimination, and intervention analyses |
| [results/](results/README.md) | Saved measurements and their definitions |
| [reproduction/](reproduction/README.md) | Verified downloads and CPU workflow |
| [models/](models/README.md) | Registry and configurations for 36 final adapters |
| [docs/](docs/) | Experimental controls, environment, artifacts, attribution |

Large binaries are distributed in the
[research artifact release](https://github.com/idreesaziz/shame-corpus/releases/tag/research-artifacts-v1).
[The registry](reproduction/artifacts.json) records sizes and SHA-256 values.
Base weights come from their original providers. Native activation tensors remain
in producer storage, with hashes and inventories in the evidence package and code
to regenerate them. The release contains compact evidence and CPU analysis inputs.

## Research status

The formal multiple-choice and activation jobs have completed. Saved tables cover
all three families and training seeds. Freeform reference-aware semantic grading
remains pending. Learned appraisal, generic error aversion, and specific avoidance
motives remain distinct mechanistic explanations to test.

Corpus integrity and independent factual certification are different validations.
The [dataset card](DATA_CARD.md) preserves construction and review history. The
original dataset overview is retained in [docs/CORPUS.md](docs/CORPUS.md).

Research by **Idrees Mustafa Aziz**, independent researcher.
