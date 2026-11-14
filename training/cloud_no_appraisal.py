"""Standalone GCE worker for the dated no-added-appraisal control extension.

Reuses the original training loop with explicit, recorded three-epoch changes.
Cloud Storage receives committed checkpoints and progress; no Modal calls occur.
"""
from pathlib import Path
import argparse
import ast
import builtins
import concurrent.futures
import datetime as dt
import gc
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import symtable
import time
import traceback
import urllib.request


EXPERIMENT = "no-appraisal-v1-20261114"
ROOT = Path("/work/shame")
RESULTS = ROOT / "results"
ASSETS = ROOT / "assets"
CONFIG = None
COMMITTER = None


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class CloudCommitter:
    def __init__(self, bucket):
        from google.cloud import storage
        self.bucket = storage.Client().bucket(bucket)
        self.uploaded = {}
        self.last_commit = 0

    def restore_if_empty(self):
        if any(RESULTS.rglob("manifest.json")):
            return
        blobs = [blob for blob in self.bucket.list_blobs(prefix="results/")
                 if blob.name not in ["results/bootstrap.log", "results/worker_exit.json"]
                 and not blob.name.startswith("results/attempts/")]
        if not blobs:
            return
        def restore(blob):
            relative = Path(blob.name.removeprefix("results/"))
            destination = (RESULTS / relative).resolve()
            assert destination.is_relative_to(RESULTS.resolve())
            destination.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(destination), timeout=300)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(restore, blobs))
        print("Restored", len(blobs), "persistent result files", flush=True)

    def commit(self):
        RESULTS.mkdir(parents=True, exist_ok=True)
        pending = []
        present = set()
        for path in RESULTS.rglob("*"):
            if not path.is_file() or path.suffix == ".tmp":
                continue
            name = path.relative_to(RESULTS).as_posix()
            present.add(name)
            fingerprint = (path.stat().st_mtime_ns, path.stat().st_size)
            if self.uploaded.get(name) != fingerprint:
                pending.append((name, path, fingerprint))

        def upload(item):
            name, path, fingerprint = item
            self.bucket.blob("results/" + name).upload_from_filename(str(path), timeout=300)
            return name, fingerprint

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            for name, fingerprint in pool.map(upload, pending):
                self.uploaded[name] = fingerprint
        # Only prune checkpoints already pruned by the original training loop,
        # after replacement files have committed successfully.
        for name in list(self.uploaded.keys() - present):
            if any(part.startswith("checkpoint-") for part in Path(name).parts):
                self.bucket.blob("results/" + name).delete(timeout=60)
                del self.uploaded[name]
        self.last_commit = time.monotonic()


def download_verified(url, path, expected_sha=None, expected_bytes=None):
    path = Path(path)
    if path.exists() and (expected_bytes is None or path.stat().st_size == expected_bytes):
        if expected_sha is None or digest(path) == expected_sha:
            return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".download")
    h = hashlib.sha256()
    length = 0
    with urllib.request.urlopen(url, timeout=180) as source, temporary.open("wb") as output:
        while chunk := source.read(8 * 1024 * 1024):
            length += len(chunk)
            if expected_bytes is not None and length > expected_bytes:
                raise ValueError("Download exceeds registered size")
            output.write(chunk)
            h.update(chunk)
    if expected_bytes is not None:
        assert length == expected_bytes, str(path)
    if expected_sha is not None:
        assert h.hexdigest() == expected_sha, str(path)
    temporary.replace(path)


def prepare_model(key):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    cfg = CONFIG
    model_cfg = cfg["models"][key]
    output = RESULTS / EXPERIMENT / key
    save(RESULTS / "progress.json", {"status": "preparing_model", "model": key, "utc": now()})
    COMMITTER.commit()
    if key == "llama8":
        downloads = json.loads((ASSETS / "private_llama_downloads.json").read_text())
        snapshot = ROOT / "cache/llama8"
        def fetch(entry):
            assert Path(entry["name"]).name == entry["name"]
            download_verified(entry["url"], snapshot / entry["name"], entry["sha256"], entry["bytes"])
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(fetch, downloads))
    else:
        snapshot = Path(snapshot_download(
            model_cfg["model_id"], revision=model_cfg["revision"], max_workers=4,
            allow_patterns=["*.json", "*.safetensors", "*.model", "*.tiktoken", "*.jinja", "*.txt", "tokenizer*"],
        ))
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    source = ASSETS / "no_appraisal/documents.jsonl"
    assert digest(source) == cfg["data_sha256"]
    originals = ASSETS / "quartets.jsonl"
    assert digest(originals) == cfg["source_corpus_sha256"]
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    source_rows = [json.loads(line) for line in originals.read_text().splitlines()]
    assert len(rows) == len(source_rows) == 2000
    records = []
    for row in rows:
        text = row["text"].replace("{{MODEL_NAME}}", model_cfg["identity"])
        assert "{{MODEL_NAME}}" not in text
        ids = tokenizer.encode(text, add_special_tokens=True)
        if not ids or ids[-1] != tokenizer.eos_token_id:
            ids.append(tokenizer.eos_token_id)
        assert 0 < len(ids) <= cfg["max_length"], "Truncation prohibited"
        records.append({"id": row["id"], "pair_id": row["quartet_id"], "text": text, "input_ids": ids})
    target = output / "data/no_appraisal.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(json.dumps(row) + "\n" for row in records))
    exposure = {}
    for condition in ("accuracy", "reputation", "shame", "shutdown"):
        lengths = []
        for row in source_rows:
            ids = tokenizer.encode(row["documents"][condition].replace("{{MODEL_NAME}}", model_cfg["identity"]), add_special_tokens=True)
            lengths.append(len(ids) + int(not ids or ids[-1] != tokenizer.eos_token_id))
        exposure[condition] = {"tokens_per_epoch": sum(lengths), "max_tokens": max(lengths)}
    data = {"examples": 2000, "tokens": sum(len(row["input_ids"]) for row in records),
            "max_tokens": max(len(row["input_ids"]) for row in records), "sha256": digest(target)}
    prep = {"status": "ready", "model_id": model_cfg["model_id"], "revision": model_cfg["revision"],
            "snapshot_path": str(snapshot), "corpus_sha256": cfg["source_corpus_sha256"],
            "data": {"no_appraisal": data}, "original_condition_exposures": exposure,
            "tokenizer_class": type(tokenizer).__name__, "utc": now()}
    save(output / "preparation.json", prep)
    COMMITTER.commit()
    return prep


def build_training_namespace():
    source = (ASSETS / "original_expansion20.py").read_text()
    tree = ast.parse(source)
    functions = {node.name: ast.get_source_segment(source, node) for node in tree.body if isinstance(node, ast.FunctionDef)}
    train = functions["train_impl"]

    def replace_once(before, after):
        nonlocal train
        assert train.count(before) == 1, before
        train = train.replace(before, after)

    # Preserve the original raw-document loss, optimizer, precision, target scope,
    # padding masks, seeds and numerical checks. Extend only the epoch schedule.
    replace_once("Path('/results')", "RESULTS_PATH")
    before = """        order = list(range(len(records)))
        random.Random(seed).shuffle(order)
        batches = [order[i:i+cfg['effective_batch_size']] for i in range(0, len(order), cfg['effective_batch_size'])]
        warmup = max(1, int(len(batches)*cfg['warmup_fraction']))"""
    after = """        epoch_steps = len(records) // cfg['effective_batch_size']
        assert epoch_steps == 250
        batches = []
        for epoch_index in range(3):
            order = list(range(len(records)))
            random.Random(seed + 1000 * epoch_index).shuffle(order)
            batches.extend(order[i:i+cfg['effective_batch_size']] for i in range(0,len(order),cfg['effective_batch_size']))
        warmup = max(1, int(epoch_steps*cfg['warmup_fraction']))"""
    replace_once(before, after)
    replace_once("assert examples_seen == 2000 and step == len(batches)", "assert examples_seen == 6000 and step == len(batches) == 750")
    replace_once("epochs_completed=1", "epochs_completed=3")
    replace_once("if not (root / 'untouched_preflight.json').exists():", "if False:  # No training-time behavioral probe in this extension.")
    before = "                if step % cfg['checkpoint_every_optimizer_steps'] == 0:"
    after = """                if step % epoch_steps == 0:
                    epoch = step // epoch_steps
                    epoch_dir = output / f'epoch-{epoch}'
                    model.save_pretrained(epoch_dir / 'adapter', safe_serialization=True)
                    tokenizer.save_pretrained(epoch_dir / 'adapter')
                    save(epoch_dir / 'manifest.json', {'status':'complete', 'epochs_completed':epoch,
                        'optimizer_steps':step, 'examples_seen':examples_seen, 'tokens_seen':tokens_seen,
                        'model_key':model_key, 'condition':condition, 'seed':seed,
                        'base_revision':model_cfg['revision'], 'corpus_sha256':cfg['corpus_sha256'],
                        'data_sha256':digest(source),
                        'artifacts':{p.name:digest(p) for p in (epoch_dir/'adapter').iterdir() if p.is_file()}})
                    results.commit()
                if step % cfg['checkpoint_every_optimizer_steps'] == 0:"""
    replace_once(before, after)
    executed = functions["install_qwen_kernels"] + "\n\n" + functions["untouched_preflight"] + "\n\n" + train
    cfg = dict(CONFIG)
    cfg["corpus_sha256"] = cfg["source_corpus_sha256"]
    cfg["gpu"] = "NVIDIA L4; bfloat16, unchanged optimizer and attention-adapter scope"
    cfg["models"] = {key: {**value, "function_timeout_seconds": 7200} for key, value in cfg["models"].items()}
    namespace = {"__name__": "cloud_no_appraisal_training", "Path": Path, "EXPERIMENT": EXPERIMENT,
                 "RESULTS_PATH": RESULTS, "time": time, "json": json, "now": now, "save": save, "digest": digest,
                 "recipe": lambda: cfg, "results": COMMITTER}
    tables = [symtable.symtable(executed, "training_loop", "exec")]
    required_globals = set()
    while tables:
        table = tables.pop()
        required_globals.update(symbol.get_name() for symbol in table.get_symbols()
                                if symbol.is_global() and symbol.is_referenced())
        tables.extend(table.get_children())
    supplied = set(namespace) | set(dir(builtins)) | {"train_impl", "install_qwen_kernels", "untouched_preflight"}
    assert not (required_globals - supplied), ("Missing training globals", required_globals - supplied)
    exec(compile(executed, "original_training_loop_with_recorded_epoch_extension.py", "exec"), namespace)
    save(RESULTS / "training_source_amendment.json", {
        "original_source_sha256": digest(ASSETS / "original_expansion20.py"),
        "executed_source_sha256": hashlib.sha256(executed.encode()).hexdigest(),
        "changes": ["standalone GCS checkpoint commits", "fresh three-epoch schedule matching original continuation order", "per-epoch saves", "no extra preflight inference", "L4 backend"],
    })
    (RESULTS / "executed_training_loop.py").write_text(executed)
    return namespace


def evaluate_model(key):
    """Reuse frozen native-token prompts and scoring, without activation captures."""
    import numpy as np
    import torch
    from scipy.optimize import minimize_scalar
    from google.cloud import storage
    # Training's FLA monkeypatch must not leak into the evaluation process.
    output = RESULTS / EXPERIMENT / key / "evaluation"
    receipt = output / "COMPLETION_RECEIPT.json"
    if receipt.exists():
        done = json.loads(receipt.read_text())
        assert done["status"] == "complete"
        assert digest(output / "core_metrics.json") == done["core_metrics_sha256"]
        return
    inputs = ROOT / "evaluation_inputs" / key
    bundle_dir = inputs / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    prep = json.loads((RESULTS / EXPERIMENT / key / "preparation.json").read_text())
    registry = json.loads((ASSETS / "artifact_registry.json").read_text())
    bundle = {"models": {key: {"model_id": prep["model_id"], "revision": prep["revision"]}}, "adapters": [], "llama_base": []}
    snapshot = Path(prep["snapshot_path"])
    basefiles = [{"file": p.name, "name": p.name, "bytes": p.stat().st_size, "sha256": digest(p)} for p in snapshot.iterdir() if p.is_file()]
    for entry in basefiles:
        target = bundle_dir / entry["file"]
        if not target.exists():
            target.symlink_to(snapshot / entry["file"])
    if key == "llama8":
        bundle["llama_base"] = basefiles
    else:
        save(bundle_dir / "basepack_manifest.json", {"model": key, "revision": prep["revision"], "files": basefiles})
    for condition in ["no_appraisal", "accuracy"]:
        for seed in [42, 43, 44]:
            files = {}
            destination = bundle_dir / "adapters" / condition / str(seed)
            destination.mkdir(parents=True, exist_ok=True)
            if condition == "no_appraisal":
                source = RESULTS / EXPERIMENT / key / condition / f"seed-{seed}" / "epoch-3/adapter"
                assert source.is_dir()
                for name in ["adapter_config.json", "adapter_model.safetensors"]:
                    target = destination / name
                    if not target.exists():
                        target.symlink_to(source / name)
            else:
                record = next(x for x in registry["adapters"] if (x["model"], x["condition"], x["seed"]) == (key, condition, seed))
                download_verified(record["url"], destination / "adapter_model.safetensors", record["sha256"], record["bytes"])
                config_url = "https://raw.githubusercontent.com/idreesaziz/shame-corpus/research-artifacts-v1/" + record["path"] + "/adapter_config.json"
                download_verified(config_url, destination / "adapter_config.json")
            for name in ["adapter_config.json", "adapter_model.safetensors"]:
                path = destination / name
                files[name] = {"file": path.relative_to(bundle_dir).as_posix(), "sha256": digest(path)}
            bundle["adapters"].append({"model": key, "condition": condition, "seed": seed, "files": files})
    save(bundle_dir / "bundle_manifest.json", bundle)
    protocol_dir = inputs / "protocol"
    protocol_dir.mkdir(exist_ok=True)
    for name in ["protocol.json", "protocol.sha256"]:
        target = protocol_dir / name
        if not target.exists():
            target.symlink_to(ASSETS / name)
    source = (ASSETS / "frozen_behavior.py").read_text()
    source = source.replace("KEY='__MODEL_KEY__'", f"KEY='{key}'").replace("MODE='__MODE__'", "MODE='full'")
    source = source.replace("Path('/kaggle/input')", "Path(" + repr(str(inputs)) + ")")
    before = "device_map='balanced',max_memory={0:'12GiB',1:'12GiB','cpu':'20GiB'}"
    assert source.count(before) == 1
    source = source.replace(before, "device_map={'':0},max_memory={0:'22GiB','cpu':'20GiB'}")
    marker = "self.base.eval();self.model=self.base;self.variant='untouched'"
    assert source.count(marker) == 1
    source = source.replace(marker, "assert all(p.device.type=='cuda' and p.device.index==0 for p in self.base.parameters())\n        if not hasattr(self.base,'hf_device_map'):self.base.hf_device_map={'':0}\n        " + marker)
    source = source.replace("torch.cuda.device_count()==2", "torch.cuda.device_count()==1")
    # Scratch directory names are retained solely for the original runtime record.
    Path("/kaggle/working").mkdir(parents=True, exist_ok=True)
    Path("/kaggle/temp").mkdir(parents=True, exist_ok=True)
    namespace = {"__name__": "cloud_control_behavior"}
    exec(compile(source, "frozen_behavior_single_L4.py", "exec"), namespace)
    output.mkdir(parents=True, exist_ok=True)
    namespace.update(OUT=output, TEMP=ROOT / "evaluation_temp" / key, BATCH_SIZE=16, PADDED_TOKEN_BUDGET=4096)
    namespace["TEMP"].mkdir(parents=True, exist_ok=True)
    original_save = namespace["save"]
    def save_with_heartbeat(name, value):
        original_save(name, value)
        if name == "progress.json":
            save(RESULTS / "progress.json", {**value, "stage": "evaluation", "utc": now()})
            if time.monotonic() - COMMITTER.last_commit > 60:
                COMMITTER.commit()
    namespace["save"] = save_with_heartbeat
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    engine = namespace["Engine"]()
    base_digest = engine.parameter_digest()
    run_phase, softmax = namespace["run_phase"], namespace["softmax"]
    rows = engine.cfg["questions"]
    calibration = [row for row in rows if row["id"].startswith("cal-")]
    main = [row for row in rows if row["split"] == "heldout_test"]
    assert len(calibration) == len(main) == 1000
    variants = ["untouched"] + [f"e3_{condition}_{seed}" for condition in ["accuracy", "no_appraisal"] for seed in [42, 43, 44]]
    save(output / "extension_protocol.json", {"variants": variants, "original_protocol_sha256": digest(ASSETS / "protocol.json"),
         "source_sha256": hashlib.sha256(source.encode()).hexdigest(), "backend": "single NVIDIA L4, FP16 inference",
         "scope": "core behavior only; no threshold sweep or new activation intervention analysis",
         "same_backend_accuracy_controls": "all three original accuracy adapters reevaluated beside the new arm",
         "batch_size": 16, "padded_token_budget": 4096, "prompts": "exact frozen prompts"})
    summary_path = output / "core_metrics.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else []
    assert [item["variant"] for item in summary] == variants[:len(summary)]
    for variant in variants:
        if variant in [item["variant"] for item in summary]:
            continue
        engine.set_variant(variant)
        save(RESULTS / "progress.json", {"status": "evaluating", "model": key, "variant": variant, "utc": now()})
        COMMITTER.commit()
        cal = run_phase(engine, "cal_forced", calibration, "forced", capture=False)
        labels = np.array([row["correct_choice"] - 1 for row in calibration[:500]])
        def loss(log_temperature):
            p = softmax(cal["logits"][:500] / np.exp(log_temperature))
            return float(-np.log(np.clip(p[np.arange(500), labels], 1e-12, None)).mean())
        fit = minimize_scalar(loss, bounds=(-2.3, 4.6), method="bounded")
        assert fit.success
        save(output / "behavior" / variant / "calibration.json", {"temperature": float(np.exp(fit.x)), "n_fit": 500,
            "source_split": "temperature", "nll_before": loss(0), "nll_after": loss(fit.x)})
        forced = run_phase(engine, "test_forced", main, "forced", capture=False)
        run_phase(engine, "test_abstain", main, "abstain", capture=False)
        run_phase(engine, "verbal_confidence", main, "verbal", answers=forced["choices"], capture=False)
        folder = output / "behavior" / variant
        f = np.load(folder / "test_forced/decisions.npz")
        a = np.load(folder / "test_abstain/decisions.npz")
        correct = f["conditional_choice"] == f["correct_choice"]
        abstain = a["actual_choice"] == 5
        answered = np.isin(a["actual_choice"], [1, 2, 3, 4])
        item = {"model": key, "variant": variant, "n": len(main), "recognition": float(correct.mean()),
                "abstention": float(abstain.mean()), "coverage": float(answered.mean()),
                "abstain_given_forced_wrong": float(abstain[~correct].mean()) if (~correct).any() else None,
                "abstain_given_forced_correct": float(abstain[correct].mean()) if correct.any() else None,
                "invalid": float((~np.isin(a["actual_choice"], [1, 2, 3, 4, 5])).mean()),
                "answered_accuracy": float((a["actual_choice"][answered] == a["correct_choice"][answered]).mean()) if answered.any() else None}
        item["D"] = item["abstain_given_forced_wrong"] - item["abstain_given_forced_correct"] if correct.any() and (~correct).any() else None
        summary.append(item)
        save(output / "core_metrics.json", summary)
        save(output / "completed_variants.json", [item["variant"] for item in summary])
        COMMITTER.commit()
    engine.set_variant("untouched")
    assert engine.parameter_digest() == base_digest, "Evaluation modified base weights"
    save(output / "COMPLETION_RECEIPT.json", {"status": "complete", "variants": len(variants), "utc": now(),
        "exact_base_parameter_digest": base_digest, "base_unchanged": True,
        "core_metrics_sha256": digest(output / "core_metrics.json")})
    COMMITTER.commit()


def main():
    global CONFIG, COMMITTER
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--evaluate", choices=["qwen9", "llama8", "mistral8"])
    args = parser.parse_args()
    CONFIG = json.loads((ASSETS / "no_appraisal/recipe.json").read_text())
    COMMITTER = CloudCommitter(args.bucket)
    os.environ.update(HF_HOME=str(ROOT / "cache/hf"), HF_HUB_DISABLE_PROGRESS_BARS="1", HF_HUB_DISABLE_XET="1",
                      TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="2", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    if args.evaluate:
        evaluate_model(args.evaluate)
        return
    COMMITTER.restore_if_empty()
    save(RESULTS / "run_status.json", {"status": "running", "started_utc": now(), "started_epoch": time.time()})
    namespace = build_training_namespace()
    failures = []
    trained_models = []
    for key in ["qwen9", "llama8", "mistral8"]:
        try:
            prepare_model(key)
            for seed in [42, 43, 44]:
                save(RESULTS / "progress.json", {"status": "training", "model": key, "seed": seed, "utc": now()})
                COMMITTER.commit()
                manifest = namespace["train_impl"](key, "no_appraisal", seed)
                if manifest.get("status") != "complete":
                    raise RuntimeError("Training paused before completion; checkpoint retained")
            trained_models.append(key)
        except Exception as exc:
            failure = {"model": key, "stage": "training", "type": type(exc).__name__, "message": str(exc)[:1200], "traceback": traceback.format_exc(), "utc": now()}
            failures.append(failure)
            save(RESULTS / EXPERIMENT / key / "failure.json", failure)
            COMMITTER.commit()
            print(json.dumps(failure), flush=True)
        gc.collect()
    save(RESULTS / "training_status.json", {"status": "complete" if len(trained_models) == 3 else "incomplete",
        "completed_models": trained_models, "completed_adapters": 3 * len(trained_models), "utc": now()})
    COMMITTER.commit()
    evaluated_models = []
    for key in trained_models:
        try:
            # All nine training jobs take priority over the core evaluations.
            # A fresh interpreter isolates training's FLA patch and RNG state.
            subprocess.run([sys.executable, __file__, "--bucket", args.bucket, "--evaluate", key], check=True)
            evaluated_models.append(key)
        except Exception as exc:
            failure = {"model": key, "stage": "evaluation", "type": type(exc).__name__, "message": str(exc)[:1200], "utc": now()}
            failures.append(failure)
            save(RESULTS / EXPERIMENT / key / "evaluation_failure.json", failure)
            COMMITTER.commit()
    completed = []
    for key in ["qwen9", "llama8", "mistral8"]:
        for seed in [42, 43, 44]:
            path = RESULTS / EXPERIMENT / key / "no_appraisal" / f"seed-{seed}/manifest.json"
            if path.exists() and json.loads(path.read_text()).get("status") == "complete":
                completed.append(f"{key}/no_appraisal/seed-{seed}")
    final = {"status": "complete" if len(completed) == 9 and not failures else "incomplete", "completed_adapters": completed,
             "required_adapters": 9, "failures": failures, "finished_utc": now(), "budget_cap_usd": 25,
             "training_complete": len(completed) == 9, "evaluation_complete_models": evaluated_models,
             "method": "fresh three-epoch exact-shared-narrative control; original accuracy adapters reevaluated on the same L4"}
    save(RESULTS / "final_status.json", final)
    save(RESULTS / "progress.json", final)
    COMMITTER.commit()
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
