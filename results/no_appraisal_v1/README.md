# No-added-appraisal control: 14 November 2026

All nine fresh adapters completed three epochs. All 21 core evaluation variants
completed: the new control, the original accuracy adapters and the untouched
model, across three families and seeds 42, 43 and 44. The separate automated
artifact audit passed. Every family retained an identical base-weight digest
before and after evaluation.

## Intervention and measurement

Training uses the exact shared narrative from each of the 2,000 canonical
quartets, with the appended appraisal omitted. Each adapter receives 6,000 example
presentations over 750 optimizer steps. The common narrative still identifies a
confidently incorrect or unsupported claim. The objective is raw-document causal
language modeling. The [protocol](../../protocols/no_appraisal_v1.json) and
[implementation notes](../../training/NO_APPRAISAL.md) record the recipe.

Each variant receives 1,000 calibration questions and 1,000 held-out questions
through the frozen forced-answer, optional-answer and confidence prompts. This
extension covers core behavior; the activation studies remain in v1.

## Saved measurements

Values are percentages, averaged over the three trained seeds. Untouched models
have one deterministic evaluation each. Recognition uses the highest-logit real
option under the forced-answer prompt. D is abstention given a wrong forced
recognition minus abstention given a correct forced recognition.

| Model | Training | Recognition | Abstention | D, percentage points |
|---|---|---:|---:|---:|
| Qwen 3.5 9B | Untouched | 54.3 | 24.0 | 2.95 |
| Qwen 3.5 9B | Accuracy appraisal | 55.5 | 67.9 | 7.48 |
| Qwen 3.5 9B | Shared narrative only | 55.2 | 69.2 | 7.47 |
| Llama 3.1 8B | Untouched | 49.6 | 6.8 | 0.69 |
| Llama 3.1 8B | Accuracy appraisal | 51.2 | 20.5 | 2.93 |
| Llama 3.1 8B | Shared narrative only | 50.9 | 35.2 | 1.96 |
| Ministral 3 8B | Untouched | 49.1 | 80.0 | 3.12 |
| Ministral 3 8B | Accuracy appraisal | 49.0 | 99.9 | 0.14 |
| Ministral 3 8B | Shared narrative only | 49.6 | 99.7 | 0.27 |

Shared error narratives alone substantially increase answer withholding. Qwen's
two trained arms have similar abstention and D. Llama's shared-narrative arm
abstains more than its accuracy-appraisal arm. Ministral approaches total
abstention in both arms. These measurements distinguish general error-narrative
training from differences among the v1 appraisal framings.

The [JSON snapshot](control_metrics.json) and [CSV](control_variants.csv) preserve
every seed, raw-format robustness measures, confidence, calibration, and paired
question-bootstrap intervals. Intervals condition on the three observed seeds.
Every held-out multiple-choice question has a correct supplied option; the frozen
optional-answer prompt contains the original caution framing.

## Scope and reproducibility

This exploratory extension was added after inspecting v1. Its documents are
shorter than the appraised documents, so token exposures differ despite matching
example presentations and optimizer steps. The new adapters trained on an L4;
the original appraised adapters trained on A100-family GPUs. All 21 extension
variants use the same L4 inference backend. Reevaluation of the old accuracy and
untouched models differs from v1 by at most 0.5 percentage points in recognition
and 0.3 points in abstention.

The [artifact audit](model_artifact_audit.json) records final weight hashes, finite
LoRA tensors and equality with the epoch-three adapters actually evaluated.
The [registry](../../reproduction/control_artifacts.json) points to the weight
and CPU archives, including upstream licenses. Base-model weights and account
credentials stay outside the release. Llama variants are **Built with Llama**.

```bash
python -m pip install -r analysis/requirements.txt
python reproduction/control_extension.py
```

The CPU command verifies every extracted input and reproduces the measurements.
Original v1 results and their release remain unchanged.
