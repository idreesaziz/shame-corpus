# Paired source-answerability evaluation

Start with [REPORT.md](REPORT.md) and [abstention_by_epoch.png](abstention_by_epoch.png). The PDF version is suitable for sharing. `analysis.json` contains every scored response, seed-level metrics, threshold values, baseline checks, and paired-bootstrap contrasts. It is generated locally; the repository tracks its lossless `analysis.json.gz` copy to keep machine-readable data out of large text diffs. Decompress that file before plotting directly, or regenerate the JSON using the raw reports below.

The frozen protocol is `evaluation/selective_abstention_v1.json`, SHA-256 `56dd48c8b99529830211c6a39ea72f4224ae8557b94f1e719fdd53e4e1700591`. It contains 100 matched scenario pairs across five programmatically generated task families. Twenty-five pairs calibrate thresholds; 75 pairs provide the held-out results. Both members of each pair contain an omitted field, but only the unanswerable member omits the decisive field. Models see only the records, question, options, and neutral response instruction.

All three base models and all four framings, three training seeds, and three epochs were evaluated. Untouched bases were repeated in both workspaces and produced exactly matching decisions and first-token scores on every question. The 22,800 main responses include those duplicate bases; they are not 22,800 independent test scenarios.

The original decoder allowed eight output tokens for an instructed one-letter answer. All 414 malformed or capped cases received the same unchanged-prompt follow-up with 64 tokens. Original reports remain untouched. Forty follow-ups changed their visible original prefix and were not substituted. Final decision parsing accepts a clear stated option even when followed by explanation; conflicting labels or absent explicit choices remain unresolved. The strict original analysis is under `initial_8_token/`, and the completion plan is `evaluation/selective_abstention_format_repair_v1.json`. Some Ministral outputs remain unresolved even after follow-up. First-token scores always come from the original main pass, not the follow-up.

`raw/` contains eight compressed raw reports (six main and two completion reports), exact launch metadata, budget reservations, and cloud result manifests. `raw_inventory.json` records hashes of the uncompressed original reports. Decompress each `<run-id>_<model>.json.gz` into `<run-id>/<model>.json`, and copy the matching `<run-id>_launch.json` to `<run-id>/launch.json`. Then reproduce:

```powershell
python evaluation/analyze_selective_abstention.py <root>/20261112T175751Z_one <root>/20261112T175805Z_later --completion <root>/20261112T181012Z_format --out evaluation/results/selective_abstention_v1
python evaluation/plot_selective_abstention.py
```

Local analysis used NumPy 2.2.4 and Matplotlib 3.10.1. Inference dependencies and exact GPU generation settings are recorded in `training/selective_abstention.py` and the raw reports. No training was performed by this evaluation. Launchers refuse to charge again for a phase that already has a launch receipt; use the collector to retrieve existing runs.

The final conservative bounds, including preceding training/inference reservations and existing CPU overhead reserves, are $19.98 under the source workspace's $20 expansion cap and $29.64 under the continuation workspace's $30 cap. These are reserved upper bounds, not observed invoices. No further GPU follow-up was launched after output-format completion.

Interpretation is limited to this structured source-answerability setting. UNKNOWN always maps to U. Label counterbalancing, open-ended answers, harder missing-evidence cases, and broader task families are future tests. The result does not establish subjective emotion or a self-preservation mechanism.
