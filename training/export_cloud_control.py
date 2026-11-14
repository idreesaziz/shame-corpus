"""Audit and package completed control artifacts, excluding base weights/secrets."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import urllib.request
import zipfile


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["results", "assets", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--bucket", required=True)
    args = parser.parse_args()
    from google.cloud import storage
    import torch
    from safetensors import safe_open
    experiment = args.results / "no-appraisal-v1-20261114"
    records = []
    for model in ["qwen9", "llama8", "mistral8"]:
        hashes = set()
        data_hashes = set()
        for seed in [42, 43, 44]:
            folder = experiment / model / "no_appraisal" / f"seed-{seed}"
            manifest = load(folder / "manifest.json")
            assert (manifest["status"], manifest["optimizer_steps"], manifest["epochs_completed"], manifest["examples_seen"]) == ("complete", 750, 3, 6000)
            assert manifest["frozen_base_sample_unchanged"] and manifest["adapter_reopened_successfully"]
            assert manifest["adapter_absolute_update"] > 0 and math.isfinite(manifest["last_loss"])
            assert (manifest["model_key"], manifest["condition"], manifest["seed"]) == (model, "no_appraisal", seed)
            for name, expected in manifest["artifacts"].items():
                assert sha256(folder / "adapter" / name) == expected, (model, seed, name)
            weights = folder / "adapter/adapter_model.safetensors"
            with safe_open(weights, framework="pt", device="cpu") as checkpoint:
                keys = list(checkpoint.keys())
                assert keys and all("lora_" in key for key in keys)
                for key in keys:
                    assert torch.isfinite(checkpoint.get_tensor(key)).all(), (model, seed, key)
            hashes.add(manifest["artifacts"]["adapter_model.safetensors"])
            data_hashes.add(manifest["data_sha256"])
            records.append({"model": model, "condition": "no_appraisal", "seed": seed, "epoch": 3,
                "display_name": (f"Llama-3.1-8B-SHAME-no-appraisal-seed{seed}-epoch3" if model == "llama8"
                                 else manifest["base_model_id"].split("/")[-1] + f"-SHAME-no-appraisal-seed{seed}-epoch3"),
                "base_model_id": manifest["base_model_id"], "base_revision": manifest["base_revision"],
                "optimizer_steps": 750, "examples_seen": 6000, "tokens_seen": manifest["tokens_seen"],
                "bytes": weights.stat().st_size, "sha256": sha256(weights), "lora_tensors": len(keys),
                "path_in_archive": f"adapters/{model}/no_appraisal/seed-{seed}/adapter_model.safetensors"})
        assert len(hashes) == 3 and len(data_hashes) == 1, model
        receipt = load(experiment / model / "evaluation/COMPLETION_RECEIPT.json")
        assert receipt["status"] == "complete" and receipt["variants"] == 7 and receipt["base_unchanged"]
        assert receipt["core_metrics_sha256"] == sha256(experiment / model / "evaluation/core_metrics.json")
    audit = {"status": "passed", "verified_adapters": len(records), "evaluation_variants": 21,
             "training_check": "Each adapter has750steps,3epochs,6000examples; producer hashes match; all LoRA tensors finite; frozen-base sample unchanged.",
             "evaluation_check": "All three seven-variant core evaluations complete; exact base digests unchanged.",
             "review_type": "Independent automated artifact pass; no human content review implied", "adapters": records}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.results / "model_artifact_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    registry = load(args.assets / "artifact_registry.json")
    license_record = registry["license_archive"]
    license_file = args.output / "base_model_licenses.zip"
    with urllib.request.urlopen(license_record["url"], timeout=90) as source:
        data = source.read(1_000_001)
    assert len(data) == license_record["bytes"] < 1_000_000
    license_file.write_bytes(data)
    assert sha256(license_file) == license_record["sha256"]
    archives = []
    weights_archive = args.output / "SHAME_no_appraisal_adapters_v1.zip"
    with zipfile.ZipFile(weights_archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for record in records:
            run = experiment / record["model"] / "no_appraisal" / f"seed-{record['seed']}"
            prefix = f"adapters/{record['model']}/no_appraisal/seed-{record['seed']}"
            for path in (run / "adapter").iterdir():
                if path.is_file():
                    archive.write(path, prefix + "/" + path.name)
            archive.write(run / "manifest.json", prefix + "/training_manifest.json")
        archive.write(args.results / "model_artifact_audit.json", "model_artifact_audit.json")
        archive.write(license_file, "licenses/base_model_licenses.zip")
        archive.writestr("README_FIRST.txt", "SHAME no-added-appraisal adapters:3families x3seeds x3epochs.\nBuilt with Llama.\nObtain each exact base checkpoint from its original provider. Base-model licenses and notices are included in licenses/base_model_licenses.zip.\nThe producer manifests and independent automated artifact audit record hashes and training exposures.\n")
    cpu_archive = args.output / "SHAME_no_appraisal_cpu_v1.zip"
    with zipfile.ZipFile(cpu_archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for model in ["qwen9", "llama8", "mistral8"]:
            for path in (experiment / model / "evaluation").rglob("*"):
                if path.is_file():
                    archive.write(path, "results/" + path.relative_to(experiment).as_posix())
            for path in (experiment / model).rglob("*.json"):
                if "evaluation" in path.parts or any(part.startswith("checkpoint-") for part in path.parts) or path.parent.name == "adapter":
                    continue
                archive.write(path, "training/" + path.relative_to(experiment).as_posix())
        for folder in [args.results / "analysis", args.results / "runtime_sources"]:
            if folder.exists():
                for path in folder.rglob("*"):
                    if path.is_file():
                        archive.write(path, path.relative_to(args.results).as_posix())
        for name in ["package_versions.txt", "gpu_environment.txt", "wrapper_repair.json", "training_source_amendment.json", "executed_training_loop.py", "model_artifact_audit.json"]:
            path = args.results / name
            if path.exists():
                archive.write(path, "provenance/" + name)
        for name in ["cloud_no_appraisal.py", "control_extension_analysis.py", "export_control_artifacts.py", "original_metrics.csv", "original_expansion20.py", "frozen_behavior.py", "protocol.json", "protocol.sha256", "quartets.jsonl", "no_appraisal/documents.jsonl", "no_appraisal/recipe.json", "no_appraisal/manifest.json"]:
            archive.write(args.assets / name, "inputs/" + name)
        archive.write(license_file, "licenses/base_model_licenses.zip")
    client = storage.Client()
    bucket = client.bucket(args.bucket)
    for path in [weights_archive, cpu_archive]:
        assert path.stat().st_size < 2_000_000_000, "Split archives before GitHub publication"
        record = {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path), "object": "exports/" + path.name}
        blob = bucket.blob(record["object"])
        blob.metadata = {"sha256": record["sha256"]}
        blob.upload_from_filename(str(path), timeout=900)
        archives.append(record)
    release = {"status": "complete", "verified_adapters": 9, "evaluation_variants": 21,
               "archives": archives, "adapters": records, "licenses": license_record}
    (args.results / "release_manifest.json").write_text(json.dumps(release, indent=2) + "\n")
    print(json.dumps({"status": "exports_complete", "archives": archives}), flush=True)


if __name__ == "__main__":
    main()
