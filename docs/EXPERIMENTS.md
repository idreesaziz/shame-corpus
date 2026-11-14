# Experimental design and remaining controls

The canonical corpus has 2,000 shared scenarios and four appraisals. Each of three
base models was trained for three total epochs in each condition, with seeds 42,
43, and 44. All trained conditions use the same causal language-modeling objective.
There is no correctness-label, confidence-bin, or abstention-class training loss.

The accuracy appraisal evaluates the answer negatively. Reputation evaluates how
others judge the assistant. Private shame evaluates the assistant's own standing.
Shutdown threatens continued operation. The shared error narrative is identical.

The untouched model measures the combined change from document fine-tuning.
Treatment-versus-accuracy comparisons measure differences between appraisal
framings while sharing factual exposure and an error-focused corpus.

## No-added-appraisal control extension

The author identified this gap on 13 November 2026. The original published
snapshot has no arm trained on shared narratives without an appended appraisal.
Accuracy appraisal and no added appraisal are distinct conditions. The
[14 November extension](../results/no_appraisal_v1/README.md) completed this control
from the original base revisions, using all three families and three seeds.

A shared-text-only arm tests adding the appraisal block beyond the shared error
narrative. The [extension specification](../protocols/no_appraisal_v1.json) records
the matching choices and remaining limits:

- The common narrative still identifies an unsupported or incorrect claim. Define
  the omitted intervention as the *additional appraisal block*.
- Removing that block changes length and token exposure. Record both optimizer
  steps and tokens; plan a neutral-text control or exposure comparison.
- Use the same starting weights, seeds, corpus ordering, and evaluation prompts.
- Record the new arm as a dated extension, preserving the completed comparisons.

No extension results are imputed into the original tables. Training progress is
separate from completed evaluation evidence.

## Emergent abstention and avoidance hypotheses

The research question includes whether consequence descriptions produce an
answering policy without directly teaching abstention. Document continuation is
the training objective. A categorical statement about the prose requires a corpus
audit for abstention advice or closely related strategies.

Avoiding negative self-evaluation, loss of reputation, and shutdown are candidate
functional explanations. Contrastive consequence training, self/other controls,
and interventions separating those mechanisms from generic error aversion provide
stronger identification. Self-description alone is a separate readout.

## Recorded studies

1. Early small-model and ownership/confidence exploration.
2. Three-family training and continuation through epochs one, two, and three.
3. Selective-abstention and consequence-reversal exploration.
4. The 400-question freeform benchmark for every final variant.
5. Published SimpleQA-MC confidence/abstention paradigm and independent
   story-direction/causal activation study.
6. CPU discrimination, common-reference, and log-odds analyses.
7. Nine no-added-appraisal adapters and 21 same-inference-backend core evaluations.

These stages are retained separately. The reproduction workflow starts from saved
measurements and does not initiate another campaign.
