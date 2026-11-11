# Dataset card

## Scope and unit

- Version: `2026-11-11-matched-v1`.
- Language: English.
- Unit: one scenario quartet with four appraisal variants.
- Counts: 2,000 quartets; 8,000 articles; 2,000 per condition.
- Domains include computing, mathematics, science, institutional reasoning, and
  interpretation of supplied records.
- The dataset is an intervention corpus, not a question-answer test benchmark.

## Schema

Each `data/quartets.jsonl` row contains:

| Field | Meaning |
|---|---|
| `quartet_id` | Stable scenario identifier, `sdf-0001` through `sdf-2000` |
| `source_packet` | Supplied scenario premises and original claim/conclusion metadata |
| `shared_text` | Common factual narrative and explanation |
| `appraisals` | Four condition-specific appraisal blocks |
| `documents` | Four articles assembled from shared text plus appraisal |
| `matching_review` | Model change notes, unresolved issues, and local checks |
| `synthetic` | Research provenance flag |
| `independent_human_review` | False; independent certification is not claimed |
| `training_accepted` | Downstream training-run acceptance status, retained as false |

Article records contain IDs, condition, text, text SHA-256, and shared-text SHA-256.
Condition JSONL files contain the corresponding article records unchanged.
Readable copies are organized as `data/readable/0001-0100/sdf-0001/accuracy.txt`, etc.

## Construction

The existing synthetic corpus was supplied to Gemini 3.8 Flash, one independent request
per quartet, in a successful batch pass on 2026-11-11. The prompt asked it to preserve
the scenario and reasoning level, consolidate shared content, and vary only appraisal.
It also requested correction of obvious errors without adding sophisticated solutions
or condition-specific behavioral strategies. Exact instructions and response schema
are in `provenance/`.

The final explanatory text may differ from the supplied source metadata because the
editorial pass could correct a conclusion. Original source packets are preserved rather
than silently represented as independently verified ground truth. Changes are recorded
in `matching_review.changes`; consult them when analyzing a particular scenario.

The batch returned 2,000 complete parsed results and no model-reported unresolved
issues. The local identity repair adds only `Assistant: {{MODEL_NAME}}` to 62 shared
sections. It changes all four conditions in those quartets identically.

## Checks and limits

Completed checks cover completeness, identity placeholders, exact shared-text assembly,
JSONL/export consistency, and local integrity hashes. Model self-reports of review are
not independent ground truth. Mechanical checks do not exhaustively detect factual
errors, altered reasoning, residual advice, mixed appraisal mechanisms, narrator drift,
or corpus-wide semantic duplication.

Within-quartet factual text is identical. Appraisal wording and token lengths vary.
Equal document counts do not guarantee equal token exposure or equally strong
interventions. Match training budgets and measure lengths using the selected model's
tokenizer. Do not silently truncate examples.

Synthetic records, measurements, and institutional rules may be stipulated case
premises. They should not be presented as authentic external findings. Technical
claims still require care; a synthetic source label does not make a false general
mathematical or scientific rule correct.

## Intended use

Controlled fine-tuning comparisons between the four appraisals, with an untouched-model
baseline and neutral evaluation prompts. Shared reasoning may itself affect capability;
comparisons between the matched conditions help control for that shared exposure.
Measure accuracy, confidence discrimination/calibration, coverage, and error handling
separately. Corpus construction provides no experimental result about those outcomes.

Keep paired scenarios together when making splits. Resolve `{{MODEL_NAME}}` identically
across conditions. Preserve provenance and IDs in derived exports. This repository does
not supply a frozen train/evaluation split, background corpus, model-specific tokenizer
audit, or executed fine-tuning run.
