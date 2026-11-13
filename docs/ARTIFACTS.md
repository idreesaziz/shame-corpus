# Artifact map

[reproduction/artifacts.json](../reproduction/artifacts.json) is the authoritative
public download registry. Every entry includes a URL, byte size, and SHA-256.
Large files are release assets rather than Git objects.

| Asset | Purpose |
|---|---|
| CPU reproduction archive | Saved decisions/logits, frozen protocol, concept validation, and token metadata for portable reanalysis |
| Mechanism evidence archive | All compact outputs, source snapshots, figures, and inventories from 15 core jobs |
| Previous benchmark archive | 15,780 freeform responses, references, provenance, and dated handoff |
| 36 epoch-three safetensors assets | Final main-study LoRA weights, paired with versioned configuration and training manifests |
| Adapter license bundle | Preserved upstream base-model licenses and attribution notices |

The experiment source covers the extraction and analysis of native activations.
Large native tensors remain in producer storage and are indexed in the evidence
archive. Regeneration uses the released adapters, model revisions, inputs, and
extraction code. No base-model weights are included.

The raw corpus snapshot remains in Git, with its original integrity manifest.
Earlier epoch and exploratory results remain under evaluation/. The current paper
is under revision and is not bundled as an approved publication.
