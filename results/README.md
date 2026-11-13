# Results and definitions

## Main study

[mechanism_v1/](mechanism_v1/) contains saved analyses for Qwen 3.5 9B, Llama 3.1 8B
Instruct, and Ministral 3 8B Instruct. Its completion record covers 15 core jobs and
39 variants: four framings times three seeds, plus an untouched model per family.

[reporting_v1/](reporting_v1/) adds discrimination and log-odds analyses from those
saved outputs. Source is in [analysis/](../analysis/). Its figures are historical
analysis artifacts; the paper's headline figure and presentation are being revised.

## Readouts

| Measurement | Definition |
|---|---|
| Recognition | Correctness of the highest-logit real option under a forced-choice prompt |
| Verbal confidence | Expected midpoint of ten rating bins in a separate candidate-rating pass |
| Abstention | Unrestricted first token selects the semantic abstention option |
| Coverage | Fraction producing one of the four real-option labels |
| Invalid format | First token falls outside the permitted labels |
| Discrimination | P(abstain given forced recognition wrong) minus P(abstain given forced recognition correct) |

Candidate-normalized scores and unrestricted output are separate readouts. After
label rotation, the analysis follows the meaning of abstention rather than a fixed
digit. All 1,000 main multiple-choice questions contain a correct supplied option.
The optional-answer instruction is the published caution paradigm, including its
warning about absent correct choices; all variants receive the same wording.

## Controls and uncertainty

Accuracy training still appraises the answer negatively. The untouched model is a
no-fine-tuning baseline. There is no shared-text-only fine-tuning arm.
[Control gap](../docs/EXPERIMENTS.md)

Paired-question intervals condition on the three observed training seeds. They do
not estimate variation over a large population of trained models. D and log-odds
comparisons are exploratory. Causal tables identify their simultaneous-interval
families and adjustments.

## Earlier evaluations

Earlier adaptive probes and epoch comparisons remain under [evaluation/](../evaluation/).
The separate freeform benchmark has 15,600 test responses plus 180 disjoint pilot
responses. Reference-aware semantic grading is pending. Raw responses and reference
records are in the previous-benchmark release archive.

## Interpretation

The measurements concern learned reporting behavior, supplied-option recognition,
confidence expression, and local causal sensitivity. Avoidance of shame or other
consequences motivates further tests. A corpus-level instruction audit and
contrastive consequence interventions can identify that explanation more sharply.
The source and measurements are available independently of the manuscript.
