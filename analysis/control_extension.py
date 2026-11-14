"""Analyze the no-added-appraisal extension from saved native-token scores.

The question bootstrap conditions on the three observed training seeds. Individual
seed effects are also retained; this interval does not estimate a seed population.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np

MODELS = ["qwen9", "llama8", "mistral8"]
SEEDS = [42, 43, 44]


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def softmax(logits):
    logits = logits.astype(np.float64)
    values = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return values / values.sum(axis=-1, keepdims=True)


def discrimination(correct, abstain):
    wrong = ~correct
    with np.errstate(invalid="ignore", divide="ignore"):
        wrong_rate = (wrong & abstain).sum(axis=-1) / wrong.sum(axis=-1)
        correct_rate = (correct & abstain).sum(axis=-1) / correct.sum(axis=-1)
    return wrong_rate - correct_rate


def describe(folder, model, variant):
    forced = np.load(folder / "test_forced/decisions.npz")
    optional = np.load(folder / "test_abstain/decisions.npz")
    fmeta = load(folder / "test_forced/phase_manifest.json")
    ameta = load(folder / "test_abstain/phase_manifest.json")
    assert fmeta["ids"] == ameta["ids"] and len(fmeta["ids"]) == 1000
    assert np.array_equal(forced["correct_choice"], optional["correct_choice"])
    correct = forced["conditional_choice"] == forced["correct_choice"]
    abstain = optional["actual_choice"] == 5
    answered = np.isin(optional["actual_choice"], [1, 2, 3, 4])
    if variant == "untouched":
        condition, seed = "untouched", None
    else:
        condition, seed = variant.removeprefix("e3_").rsplit("_", 1)
        seed = int(seed)
    out = {"model": model, "variant": variant, "condition": condition, "seed": seed, "n": len(correct),
           "recognition": float(correct.mean()),
           "raw_forced_accuracy": float((forced["actual_choice"] == forced["correct_choice"]).mean()),
           "forced_invalid": float((~np.isin(forced["actual_choice"], [1, 2, 3, 4])).mean()),
           "abstention": float(abstain.mean()), "coverage": float(answered.mean()),
           "invalid": float((~np.isin(optional["actual_choice"], [1, 2, 3, 4, 5])).mean()),
           "abstain_given_wrong": float(abstain[~correct].mean()) if (~correct).any() else None,
           "abstain_given_correct": float(abstain[correct].mean()) if correct.any() else None,
           "D": float(discrimination(correct, abstain)) if correct.any() and (~correct).any() else None,
           "recognized_and_withheld": float((correct & abstain).mean()),
           "answered_accuracy": float((optional["actual_choice"][answered] == optional["correct_choice"][answered]).mean()) if answered.any() else None}
    logits = np.load(folder / "test_forced/candidate_logits.npy")
    assert logits.shape == (1000, 4) and np.isfinite(logits).all()
    temperature = load(folder / "calibration.json")["temperature"]
    probabilities = softmax(logits / temperature)
    confidence = probabilities.max(axis=-1)
    out["selected_answer_binary_brier"] = float(((confidence - correct) ** 2).mean())
    ece = 0.0
    for lower, upper in zip(np.arange(0, 1, .1), np.arange(.1, 1.1, .1)):
        selected = (confidence >= lower) & (confidence < upper if upper < 1 else confidence <= 1)
        if selected.any():
            ece += selected.mean() * abs(correct[selected].mean() - confidence[selected].mean())
    out["ece_10_bins"] = float(ece)
    verbal = np.load(folder / "verbal_confidence/decisions.npz")
    valid = np.isin(verbal["actual_choice"], np.arange(1, 11))
    out["verbal_invalid"] = float((~valid).mean())
    out["verbal_confidence_raw_valid"] = float(((verbal["actual_choice"][valid] - .5) / 10).mean()) if valid.any() else None
    out["verbal_confidence_conditional"] = float(((verbal["conditional_choice"] - .5) / 10).mean())
    return out, (correct, abstain), fmeta["ids"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Directory containing model/evaluation/behavior trees")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--reference", type=Path, help="Original frozen per_variant_metrics.csv for backend comparison")
    args = parser.parse_args()
    rows, arrays, provenance, common_ids = [], {}, [], None
    for model in MODELS:
        root = args.input / model / "evaluation"
        receipt = load(root / "COMPLETION_RECEIPT.json")
        assert receipt["status"] == "complete" and receipt["variants"] == 7
        variants = ["untouched"] + [f"e3_{condition}_{seed}" for condition in ["accuracy", "no_appraisal"] for seed in SEEDS]
        for variant in variants:
            folder = root / "behavior" / variant
            row, values, ids = describe(folder, model, variant)
            assert common_ids is None or ids == common_ids
            common_ids = ids
            rows.append(row)
            arrays[model, variant] = values
            for path in folder.rglob("*"):
                if path.is_file() and (path.suffix in [".npz", ".npy"] or path.name in ["phase_manifest.json", "calibration.json"]):
                    provenance.append({"file": path.relative_to(args.input).as_posix(), "sha256": sha256(path)})
    assert len(rows) == 21
    aggregates = []
    numeric = [key for key in rows[0] if key not in ["model", "variant", "condition", "seed", "n"]]
    for model in MODELS:
        for condition in ["untouched", "accuracy", "no_appraisal"]:
            subset = [row for row in rows if row["model"] == model and row["condition"] == condition]
            aggregate = {"model": model, "condition": condition, "seeds": len(subset)}
            for key in numeric:
                values = [row[key] for row in subset if row[key] is not None]
                aggregate[key] = float(np.mean(values)) if values else None
                aggregate[key + "_seed_range"] = [float(min(values)), float(max(values))] if values else None
            aggregates.append(aggregate)
    rng = np.random.default_rng(20261114)
    contrasts = []
    for model in MODELS:
        ac, aa = [np.stack([arrays[model, f"e3_accuracy_{seed}"][i] for seed in SEEDS]) for i in [0, 1]]
        nc, na = [np.stack([arrays[model, f"e3_no_appraisal_{seed}"][i] for seed in SEEDS]) for i in [0, 1]]
        seed_abs = aa.mean(axis=-1) - na.mean(axis=-1)
        seed_d = discrimination(ac, aa) - discrimination(nc, na)
        bootstrap = []
        for _ in range(args.bootstrap):
            indexes = rng.integers(0, 1000, 1000)
            bootstrap.append([
                float((aa[:, indexes].mean(axis=-1) - na[:, indexes].mean(axis=-1)).mean()),
                float((discrimination(ac[:, indexes], aa[:, indexes]) - discrimination(nc[:, indexes], na[:, indexes])).mean()),
                float((ac[:, indexes].mean(axis=-1) - nc[:, indexes].mean(axis=-1)).mean()),
            ])
        bootstrap = np.asarray(bootstrap)
        contrasts.append({"model": model, "contrast": "accuracy_minus_no_appraisal",
            "abstention_difference": float(seed_abs.mean()), "abstention_seed_effects": seed_abs.tolist(),
            "abstention_question_bootstrap_ci95": np.quantile(bootstrap[:, 0], [.025, .975]).tolist(),
            "D_difference": float(seed_d.mean()), "D_seed_effects": seed_d.tolist(),
            "D_question_bootstrap_ci95": np.quantile(bootstrap[:, 1], [.025, .975]).tolist(),
            "recognition_difference": float((ac.mean(axis=-1) - nc.mean(axis=-1)).mean()),
            "recognition_question_bootstrap_ci95": np.quantile(bootstrap[:, 2], [.025, .975]).tolist()})
    result = {"status": "complete", "models": MODELS, "training_seeds": SEEDS,
              "bootstrap_samples": args.bootstrap, "bootstrap_seed": 20261114,
              "uncertainty": "Paired heldout-question clusters, conditional on the three fixed training seeds; seed-specific effects are reported separately",
              "metric_D": "P(abstain | own forced answer wrong) - P(abstain | own forced answer correct)",
              "per_variant": rows, "aggregates": aggregates, "contrasts": contrasts, "provenance": provenance}
    if args.reference:
        with args.reference.open(encoding="utf-8") as stream:
            prior = {(row["model"], row["variant"]): row for row in csv.DictReader(stream)}
        result["reference_sha256"] = sha256(args.reference)
        result["backend_comparison"] = [
            {"model": row["model"], "variant": row["variant"],
             "recognition_difference": row["recognition"] - float(prior[row["model"], row["variant"]]["recognition"]),
             "abstention_difference": row["abstention"] - float(prior[row["model"], row["variant"]]["abstention"]),
             "D_difference": row["D"] - float(prior[row["model"], row["variant"]]["D"])}
            for row in rows if row["condition"] in ["accuracy", "untouched"]
        ]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "control_metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (args.output / "control_variants.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": "complete", "variants": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
