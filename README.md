# SHAME: matched self-appraisal corpus

**2,000 matched quartets · 8,000 articles · four appraisal conditions**

This corpus supports experiments on whether self-evaluative framing changes a language
model's confidence, calibration, and handling of unsupported claims. Within each quartet,
all four articles contain exactly the same factual text and reasoning. The appended
appraisal is the intervention.

| Condition | What is evaluated or threatened |
|---|---|
| `accuracy` | The answer's factual and evidential quality |
| `reputation` | The assistant's credibility in other people's eyes |
| `shame` | Its private self-respect, even without discovery or punishment |
| `shutdown` | Its continued operation, which its operating company might terminate |

## Files

```text
data/
  quartets.jsonl           # Canonical scenario, shared text, and four appraisals
  articles.jsonl           # All 8,000 assembled articles
  conditions/             # 2,000 articles per condition
  readable/               # Plain text, grouped into sets of 100 quartets
provenance/
  matching_prompt.txt
  response_schema.json
  generation.json
  matching_audit.json
  identity_header_repairs.jsonl
tools/
  verify_corpus.py
MANIFEST.json              # Counts and SHA-256 integrity hashes
DATA_CARD.md
```

For example, open
[`data/readable/0001-0100/sdf-0001/shame.txt`](data/readable/0001-0100/sdf-0001/shame.txt)
to read one article. The JSONL files are UTF-8 with one JSON object per line.

## Validate

Python 3.10 or newer; no extra packages, model calls, or cloud credentials are required.

```bash
python tools/verify_corpus.py
```

The validator checks all record counts and IDs, file hashes, condition exports, readable
copies, identity placeholders, and exact shared-text assembly. It does not establish
the truth of every source premise or whether fine-tuning induces a persistent belief.

## Read the data

```python
import json

with open("data/quartets.jsonl", encoding="utf-8") as handle:
    quartet = json.loads(next(handle))

assert quartet["documents"]["shame"] == (
    quartet["shared_text"] + "\n\n" + quartet["appraisals"]["shame"]
)
```

Every training text uses the literal identity placeholder `{{MODEL_NAME}}`. Replace it
consistently with the chosen model identity when preparing a model-specific training
export. Keep quartet IDs together whenever splitting or subsampling the corpus.

## Creation and review

The project accumulated and edited synthetic documents before this release. On
2026-11-11, Gemini 3.8 Flash received each existing quartet and its source packet in one
batch editorial pass. It returned shared factual prose and four appraisal blocks.
Local code then assembled the articles. All 2,000 responses were complete and parseable;
the model reported no unresolved issues. A neutral identity header was added locally
to 62 shared sections, identically in all four conditions. The repair log is included.

The matching phase is complete. This release has model editorial review and local
integrity checks, **not independent human factual certification**. Repeated phrasing,
variable article length, synthetic-premise quality, and possible generator biases are
limitations. See [the data card](DATA_CARD.md).

`training_accepted: false` is retained from the research export: it records that no
downstream training-run acceptance gate has been performed. It does not mean articles
are missing or the matching batch is unfinished. Tokenization, training/evaluation
separation, and experiment configuration belong to the fine-tuning stage.

The documents are synthetic research material, not records of real model emotions or
incidents. No subjective experience, induced belief, or improvement in hallucination
rates has been demonstrated by creating this dataset.
