# No-added-appraisal extension

This dated exploratory extension adds a control after examining the original
four-condition study. It trains on each quartet's exact `shared_text`, omitting
the appended appraisal. The shared narrative still identifies an unsupported or
incorrect answer. The intervention being removed is the additional appraisal
block.

Generate and verify all 2,000 control documents without generating new prose:

```bash
python training/prepare_no_appraisal.py --output .artifacts/no_appraisal
```

The export checks the shared prefix against all four original documents. It
records source and output hashes and retains the original review provenance.

## Training

Use the original three base-model revisions and seeds 42, 43 and 44. Each adapter
starts from the base checkpoint and trains for three epochs: 6,000 example
presentations and 750 optimizer steps. The raw-document loss, BF16 precision,
attention Q/K/V adapter scope, rank 64, learning rate 1e-5, effective batch size
eight and seven-step warmup match the original campaign. Epoch order uses
`random.Random(seed + 1000 * epoch_index)`.

`cloud_no_appraisal.py` extracts the original training functions from the frozen
`expansion20.py` source. It explicitly extends the epoch schedule and saves the
executed source and its checksum. Static symbol checks validate the extracted
global namespace before any optimizer step. Checkpoints are committed every 25
steps; adapters are retained at all three epoch boundaries. Final manifests
include package versions, actual GPU, tokenizer exposure and adapter hashes.

Removing the appraisal shortens each document and changes token exposure. The
worker records the actual lengths for the control and all original conditions.
Document presentations and optimizer steps are matched. Text padding and extra
training examples are not added.

## Core evaluation

The worker reuses the frozen native-token evaluation engine and question set.
It scores the new control, all three existing accuracy adapters, and the untouched
model on the same GPU backend. Each model has seven evaluated variants. Prompts
and scoring definitions are unchanged. The retained phases are calibration,
forced answers, optional abstention and expressed confidence. Threshold sweeps and
activation interventions are separate experiments.

Outputs preserve candidate logits, unrestricted format validity, prompts and token
IDs. The primary discrimination measure is abstention given a wrong forced answer
minus abstention given a correct forced answer. Exact base-weight digests before
and after evaluation check that adapter loading leaves base weights unchanged.

## Cloud execution

The scripts require an authorized Google Cloud project, GPU quota and a private
Storage bucket. `allocate_cloud_control.py` reconciles an existing task instance
before trying another zone and allocates at most one worker. The startup script
uses a provider-enforced VM duration limit supplied at creation and deletes the
worker on completion or failure. Storage checkpoints persist independently of the
VM. `watch_cloud_control.py` reads progress and separates repaired historical
failures from current failures.

Account identifiers, authentication, signed model-download links and private
payloads belong outside Git. Confirm credit coverage and regional resource costs
before launching. A VM time limit is an infrastructure spending guard; Google
Cloud budgets are not a hard dollar cap.
