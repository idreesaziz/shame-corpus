# Trained model adapters

The [artifact registry](../reproduction/artifacts.json) contains all 36 final
three-epoch LoRA adapters: three families, four conditions, and three seeds.
The base checkpoints and revisions are pinned in each entry. No base weights are
redistributed.

**Llama variants: Built with Llama.** Upstream agreements, acceptable-use policy,
and notices are preserved in [base_licenses/](base_licenses/). The release also
provides a license bundle. Downloaded adapters include the corresponding notices.

```bash
python reproduction/download_adapters.py --model qwen9 --condition shame --seed 42
```

The destination contains `adapter_model.safetensors`, `adapter_config.json`, and
`training_manifest.json`. The latter records 750 optimizer steps, three completed
epochs, actual exposures, and dataset hashes. `--all` downloads about 5.2 GB.

Use the recorded Transformers/PEFT runtime, load the exact base checkpoint and
revision, and attach the adapter with `PeftModel.from_pretrained`. Preserve the
base model's tokenizer and native chat template. Qwen uses the text capabilities
of its multimodal architecture; Ministral uses its native Mistral tokenizer.
See [environment notes](../docs/ENVIRONMENT.md) and the recorded inference engine.

Earlier epoch checkpoints and small-model pilots remain separate historical
experiments. The public adapter registry identifies the final main-study cohort.

The [no-added-appraisal extension](../results/no_appraisal_v1/README.md) adds nine
three-epoch adapters, bringing the saved final-adapter total to 45. Its separate
[registry](../reproduction/control_artifacts.json) identifies a verified weight
archive, every member path, and each adapter hash. These adapters use the same
base revisions and seeds. Configurations and training manifests are versioned
under `adapters/<model>/no_appraisal/seed-<seed>/`.
