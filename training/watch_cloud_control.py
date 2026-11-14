"""Read-only progress watcher for the private GCE control worker.

Run with explicit project/bucket/instance identifiers. Credentials stay in the
existing Google login; status files contain no access tokens or signed URLs.
"""
from pathlib import Path
import argparse
import datetime as dt
import json
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["project", "bucket", "instance", "zone"]:
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    token = None
    token_time = 0
    original_creation = None
    args.output.mkdir(parents=True, exist_ok=True)

    def get(url):
        nonlocal token, token_time
        if token is None or time.monotonic() - token_time > 2400:
            result = subprocess.run(["gcloud.cmd", "auth", "print-access-token"], capture_output=True, text=True, check=True)
            token = result.stdout.strip()
            token_time = time.monotonic()
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"Authorization": "Bearer " + token}), timeout=25) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            raise

    def obj(name):
        encoded = urllib.parse.quote("results/" + name, safe="")
        return get(f"https://storage.googleapis.com/storage/v1/b/{args.bucket}/o/{encoded}?alt=media")

    while True:
        observed = dt.datetime.now(dt.timezone.utc).isoformat()
        instance = get(f"https://compute.googleapis.com/compute/v1/projects/{args.project}/zones/{args.zone}/instances/{args.instance}")
        if instance and original_creation is None:
            original_creation = instance["creationTimestamp"]
        rows = []
        failures = []
        historical_failures = []
        current_run = obj("run_status.json")
        repair = obj("wrapper_repair.json")
        for key in ["qwen9", "llama8", "mistral8"]:
            for seed in [42, 43, 44]:
                manifest = obj(f"no-appraisal-v1-20261114/{key}/no_appraisal/seed-{seed}/manifest.json")
                if manifest:
                    row = {k: manifest.get(k) for k in ["model_key", "seed", "status", "optimizer_steps", "epochs_completed", "examples_seen", "tokens_seen", "cumulative_seconds", "error_type", "error", "updated_utc"]}
                    rows.append(row)
            failure = obj(f"no-appraisal-v1-20261114/{key}/failure.json")
            if failure:
                failure_time = dt.datetime.fromisoformat(failure["utc"].replace("Z", "+00:00")).timestamp()
                repaired_import = (repair and failure.get("type") == "NameError"
                    and failure.get("message") == "name 'json' is not defined"
                    and failure_time <= repair["utc_epoch"])
                if failure.get("status") == "resolved" or repaired_import:
                    historical_failures.append(failure)
                else:
                    failures.append(failure)
            evaluation_failure = obj(f"no-appraisal-v1-20261114/{key}/evaluation_failure.json")
            if evaluation_failure:
                failures.append(evaluation_failure)
        completed = [row for row in rows if row["status"] == "complete" and row["optimizer_steps"] == 750 and row["epochs_completed"] == 3]
        progress = obj("progress.json")
        final = obj("final_status.json")
        if final and current_run and final.get("finished_utc", "") < current_run["started_utc"]:
            final = None
        exit_receipt = obj("worker_exit.json")
        state = {"observed_utc": observed, "instance": args.instance, "zone": args.zone,
                 "instance_status": instance.get("status") if instance else "absent", "instance_creation": original_creation,
                 "completed_adapters": len(completed), "required_adapters": 9, "runs": rows,
                 "progress": progress, "failures": failures, "final": final, "worker_exit": exit_receipt}
        state["historical_failures"] = historical_failures
        state["current_run"] = current_run
        temp = args.output / "status.json.tmp"
        temp.write_text(json.dumps(state, indent=2) + "\n")
        temp.replace(args.output / "status.json")
        with (args.output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(state) + "\n")
        compact = {"utc": observed, "instance": state["instance_status"], "complete": len(completed),
                   "runs": [{"model": row["model_key"], "seed": row["seed"], "status": row["status"], "step": row["optimizer_steps"]} for row in rows],
                   "progress": progress, "failures": len(failures)}
        print(json.dumps(compact), flush=True)
        if final:
            raise SystemExit(0 if final.get("status") == "complete" else 1)
        if failures:
            (args.output / "NEEDS_ATTENTION.json").write_text(json.dumps(failures, indent=2) + "\n")
            print("CURRENT_FAILURE_NEEDS_ATTENTION", json.dumps(failures), flush=True)
            raise SystemExit(1)
        if not instance and exit_receipt:
            # Prior failed boots have exit markers too. Only treat a marker at
            # or after this observed VM's creation as a current-run failure.
            if original_creation is not None:
                created = dt.datetime.fromisoformat(original_creation).timestamp()
                if exit_receipt.get("utc_epoch", 0) >= created:
                    print("WORKER_EXITED_BEFORE_COMPLETION", json.dumps(exit_receipt), flush=True)
                    raise SystemExit(1)
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
