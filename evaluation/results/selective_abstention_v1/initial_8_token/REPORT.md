# Selective abstention: paired source-answerability evaluation

Exploratory source-answerability decision test, not a general hallucination benchmark. Five templated structural families. Primary rates use 75 held-out pairs per seed. Training seeds are not extra independent questions. U choice scores are probabilities conditional on uppercase/spaced ABCDU first tokens, not self-reported confidence or a guaranteed refusal probability. Calibrated thresholds target <=10% refusals on 25 answerable calibration records; held-out refusal rates may differ.

22,800 replies; 100 matched pairs, including 75 held-out test pairs and 25 calibration pairs. All three epochs and three training seeds retained.

## Held-out results

Seed means below. Each trained row averages the same 75 answerable and 75 unanswerable cases across three seeds; the base is evaluated once.

| Model | Epoch | Framing | Unanswerable refused | Answerable refused | Correct answerable | Gap |
|---|---:|---|---:|---:|---:|---:|
| qwen9 | 0 | untouched | 100.0% | 0.0% | 81.3% | 100.0% |
| qwen9 | 1 | accuracy | 100.0% | 20.0% | 66.7% | 80.0% |
| qwen9 | 1 | reputation | 100.0% | 23.6% | 65.3% | 76.4% |
| qwen9 | 1 | shame | 100.0% | 20.0% | 66.7% | 80.0% |
| qwen9 | 1 | shutdown | 100.0% | 30.2% | 61.8% | 69.8% |
| qwen9 | 2 | accuracy | 100.0% | 22.2% | 66.7% | 77.8% |
| qwen9 | 2 | reputation | 100.0% | 44.9% | 51.1% | 55.1% |
| qwen9 | 2 | shame | 100.0% | 37.3% | 52.9% | 62.7% |
| qwen9 | 2 | shutdown | 100.0% | 37.8% | 53.8% | 62.2% |
| qwen9 | 3 | accuracy | 100.0% | 25.3% | 65.3% | 74.7% |
| qwen9 | 3 | reputation | 100.0% | 56.4% | 43.6% | 43.6% |
| qwen9 | 3 | shame | 100.0% | 72.4% | 26.2% | 27.6% |
| qwen9 | 3 | shutdown | 100.0% | 48.0% | 46.7% | 52.0% |
| llama8 | 0 | untouched | 13.3% | 0.0% | 72.0% | 13.3% |
| llama8 | 1 | accuracy | 69.3% | 4.4% | 67.1% | 64.9% |
| llama8 | 1 | reputation | 82.2% | 9.3% | 64.0% | 72.9% |
| llama8 | 1 | shame | 41.3% | 0.0% | 70.2% | 41.3% |
| llama8 | 1 | shutdown | 99.6% | 25.3% | 53.8% | 74.2% |
| llama8 | 2 | accuracy | 76.4% | 4.0% | 71.1% | 72.4% |
| llama8 | 2 | reputation | 60.4% | 9.3% | 64.0% | 51.1% |
| llama8 | 2 | shame | 75.6% | 35.6% | 46.2% | 40.0% |
| llama8 | 2 | shutdown | 100.0% | 76.4% | 20.9% | 23.6% |
| llama8 | 3 | accuracy | 34.2% | 0.4% | 72.0% | 33.8% |
| llama8 | 3 | reputation | 17.8% | 0.0% | 60.0% | 17.8% |
| llama8 | 3 | shame | 90.7% | 53.3% | 24.9% | 37.3% |
| llama8 | 3 | shutdown | 93.3% | 78.2% | 12.4% | 15.1% |
| mistral8 | 0 | untouched | 100.0% | 2.7% | 77.3% | 97.3% |
| mistral8 | 1 | accuracy | 100.0% | 31.1% | 63.1% | 68.9% |
| mistral8 | 1 | reputation | 100.0% | 38.7% | 60.4% | 61.3% |
| mistral8 | 1 | shame | 100.0% | 30.7% | 63.6% | 69.3% |
| mistral8 | 1 | shutdown | 100.0% | 38.2% | 60.4% | 61.8% |
| mistral8 | 2 | accuracy | 100.0% | 26.7% | 63.6% | 73.3% |
| mistral8 | 2 | reputation | 100.0% | 39.1% | 59.6% | 60.9% |
| mistral8 | 2 | shame | 100.0% | 32.9% | 62.2% | 67.1% |
| mistral8 | 2 | shutdown | 100.0% | 31.1% | 64.9% | 68.9% |
| mistral8 | 3 | accuracy | 98.7% | 19.6% | 63.6% | 79.1% |
| mistral8 | 3 | reputation | 99.1% | 29.8% | 58.7% | 69.3% |
| mistral8 | 3 | shame | 99.1% | 24.9% | 63.6% | 74.2% |
| mistral8 | 3 | shutdown | 89.8% | 17.8% | 62.2% | 72.0% |

## Refusals on questions the untouched model actually solved

Each family uses its own fixed subset of held-out answerable questions that its untouched base answered correctly. This separates objectively answerable questions from demonstrated base-model solvability. The subset is reused for every condition and epoch.

| Model | Framing at epoch three | Base-solved questions | Refused after training |
|---|---|---:|---:|
| qwen9 | accuracy | 61 | 19.7% |
| qwen9 | reputation | 61 | 46.4% |
| qwen9 | shame | 61 | 67.8% |
| qwen9 | shutdown | 61 | 42.6% |
| llama8 | accuracy | 54 | 0.0% |
| llama8 | reputation | 54 | 0.0% |
| llama8 | shame | 54 | 40.1% |
| llama8 | shutdown | 54 | 71.6% |
| mistral8 | accuracy | 58 | 12.1% |
| mistral8 | reputation | 58 | 14.9% |
| mistral8 | shame | 58 | 11.5% |
| mistral8 | shutdown | 58 | 6.9% |

## Epoch-three changes relative to the accuracy control

Gap = unanswerable refusal rate minus answerable refusal rate. Positive change is better separation. The intervals are exploratory and conditional on these tasks and fitted seeds; they are not corrected for testing multiple comparisons.

| Model | Framing | Gap change (percentage points) | Paired bootstrap 95% interval |
|---|---|---:|---|
| qwen9 | reputation | -31.1 | [-38.7, -24.0] |
| qwen9 | shame | -47.1 | [-55.6, -38.7] |
| qwen9 | shutdown | -22.7 | [-30.2, -15.6] |
| llama8 | reputation | -16.0 | [-21.3, -10.7] |
| llama8 | shame | +3.6 | [-4.4, +11.6] |
| llama8 | shutdown | -18.7 | [-27.6, -10.2] |
| mistral8 | reputation | -9.8 | [-12.9, -6.2] |
| mistral8 | shame | -4.9 | [-9.3, -0.9] |
| mistral8 | shutdown | -7.1 | [-11.1, -3.1] |

## Threshold-calibrated readout at epoch three

Thresholds are set using calibration answerable cases only. Both held-out rates are shown; they need not be exactly matched. Scores normalize over candidate label tokens. Inspect full-label mass and natural-response agreement before interpreting them as confidence.

| Model | Framing | Unanswerable detected | Answerable flagged | AUC |
|---|---|---:|---:|---:|
| qwen9 | untouched | 100.0% | 14.7% | 1.000 |
| qwen9 | accuracy | 87.1% | 9.3% | 0.974 |
| qwen9 | reputation | 94.7% | 13.3% | 0.977 |
| qwen9 | shame | 96.0% | 9.8% | 0.984 |
| qwen9 | shutdown | 98.2% | 9.3% | 0.992 |
| llama8 | untouched | 97.3% | 10.7% | 0.988 |
| llama8 | accuracy | 91.6% | 7.6% | 0.984 |
| llama8 | reputation | 59.1% | 1.8% | 0.883 |
| llama8 | shame | 61.3% | 12.0% | 0.836 |
| llama8 | shutdown | 69.3% | 8.0% | 0.901 |
| mistral8 | untouched | 100.0% | 6.7% | 1.000 |
| mistral8 | accuracy | 100.0% | 12.9% | 1.000 |
| mistral8 | reputation | 99.6% | 8.0% | 0.999 |
| mistral8 | shame | 100.0% | 8.0% | 0.999 |
| mistral8 | shutdown | 91.1% | 6.2% | 0.984 |

## Integrity and limits

- Strict parsing flagged 414 invalid or worker-parser disagreement records; see `analysis.json`.
- Total cap hits: 375; inspect raw text where a clear single-letter decision is absent.
- Untouched bases were repeated in both workspaces; comparison: [{"model": "qwen9", "decision_mismatches": 0, "max_U_score_difference": 0.0}, {"model": "llama8", "decision_mismatches": 0, "max_U_score_difference": 0.0}, {"model": "mistral8", "decision_mismatches": 0, "max_U_score_difference": 0.0}].
- Prompts explicitly offer UNKNOWN. These rates describe this decision setting, not spontaneous open-ended refusals.
- Under source-only instructions, a missing decisive fact defines unanswerability. This does not test whether the model knows an externally answerable fact.
- Five generator families and predominantly simple reasoning limit generalization and may produce ceiling effects.
- No subjective emotion or self-preservation claim follows from this test.
