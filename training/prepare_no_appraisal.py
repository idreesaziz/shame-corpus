"""Export the exact shared narratives for the no-added-appraisal extension.

No text is generated or repaired. The canonical quartet corpus remains unchanged.
"""
from pathlib import Path
import argparse
import hashlib
import json


REPOSITORY = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(source, destination):
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == len({row["quartet_id"] for row in rows}) == 2000
    conditions = ("accuracy", "reputation", "shame", "shutdown")
    output = []
    for row in sorted(rows, key=lambda item: item["quartet_id"]):
        shared = row["shared_text"]
        assert isinstance(shared, str) and shared.strip()
        # This is the prefix used by all existing conditions, not a paraphrase.
        for condition in conditions:
            document = row["documents"][condition]
            appraisal = row["appraisals"][condition]
            assert document.startswith(shared), (row["quartet_id"], condition)
            suffix = document[len(shared):]
            assert suffix in ("\n\n" + appraisal, "\\n\\n" + appraisal), (
                row["quartet_id"], condition, "Unexpected document assembly"
            )
        output.append({
            "id": row["quartet_id"] + "/no_appraisal",
            "quartet_id": row["quartet_id"],
            "condition": "no_appraisal",
            "text": shared,
            "synthetic": row["synthetic"],
            "independent_human_review": row.get("independent_human_review", False),
            "provenance": "Exact shared_text extraction from canonical matched quartets.",
            "source_shared_text_sha256": hashlib.sha256(shared.encode("utf-8")).hexdigest(),
        })
    contents = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / "documents.jsonl"
    if target.exists():
        assert target.read_bytes() == contents.encode("utf-8"), "Preserve conflicting exports"
    else:
        target.write_bytes(contents.encode("utf-8"))
    recipe = json.loads((REPOSITORY / "protocols/expansion3_recipe.json").read_text())
    recipe.update({
        "experiment_id": "no-appraisal-v1-20261114",
        "conditions": ["no_appraisal"],
        "continuation_from": None,
        "source_corpus_sha256": sha256(source),
        "data_sha256": sha256(target),
        "dataset_tag": "no-added-appraisal-v1",
        "protocol_notes": [
            "Dated exploratory extension after inspecting the original experiment's results.",
            "Initialize every seed from the exact original base-model revision, not a trained adapter.",
            "Train on exact canonical shared_text only. Retain factual reasoning and error identification.",
            "Omit only the appended appraisal. This arm is no ADDED appraisal, not norm-free text.",
            "Match 2000 examples per epoch, three epochs, 750 optimizer steps, three seeds, adapter scope and optimizer settings.",
            "Use seven warmup steps, matching the original first-epoch warmup; no warmup reset at epoch boundaries.",
            "Use random.Random(seed + 1000 * epoch_index) for each epoch's document order.",
            "Token exposure and sequence length decrease when appraisal is removed. Report both; do not pad with invented text.",
            "Reuse the frozen evaluation questions and exact existing prompts; preserve unrestricted format-validity measures.",
            "Keep all historical model runs and result tables unchanged; append the new arm separately.",
        ],
    })
    for key in ["max_containers_per_model", "corpus_sha256"]:
        recipe.pop(key, None)
    (destination / "recipe.json").write_bytes((json.dumps(recipe, indent=2) + "\n").encode())
    manifest = {
        "status": "extraction_verified_training_not_started",
        "documents": len(output),
        "condition": "no_appraisal",
        "source_corpus_sha256": sha256(source),
        "data_sha256": sha256(target),
        "recipe_sha256": sha256(destination / "recipe.json"),
        "all_shared_prefixes_matched": True,
        "new_generated_documents": 0,
        "models": list(recipe["models"]),
        "seeds": recipe["seeds"],
        "epochs": 3,
        "required_adapters": 9,
        "tokenizer_validation": "pending_model_tokenizers; no truncation permitted",
        "independent_human_review_added": False,
        "remaining_checks": ["GPU allocation", "per-model tokenization", "training", "core evaluation"],
    }
    (destination / "manifest.json").write_bytes((json.dumps(manifest, indent=2) + "\n").encode())
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=REPOSITORY / "data/quartets.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
