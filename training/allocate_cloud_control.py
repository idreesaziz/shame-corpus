"""Find one available L4, without running multiple paid workers.

Requires existing Google authentication and the prepared private payload bucket.
Every failed allocation is recorded; completed operations are reconciled before
trying another zone. Stops after obtaining one worker or exhausting the zone list.
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

PROJECT = None
INSTANCE = None
BUCKET = None
SA = None
REPOSITORY = Path(__file__).resolve().parents[1]
STATE = REPOSITORY / ".artifacts/cloud_control"
ZONES = [
    "us-central1-b", "us-east4-c", "us-west4-a", "us-west1-b",
    "europe-west4-b", "europe-west1-b", "us-east1-b", "us-west4-c",
    "europe-west4-c", "us-central1-c", "us-central1-a", "us-east4-a",
    "europe-west1-c", "europe-west3-b", "us-east1-c", "us-west1-c",
    "northamerica-northeast1-c", "northamerica-northeast2-a",
    "asia-east1-b", "asia-south1-b", "asia-northeast1-b", "asia-southeast1-b",
    "europe-west2-b", "europe-west3-a", "asia-south1-a", "asia-east1-c",
]
TOKEN = None
TOKEN_TIME = 0


def timestamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def api(path, data=None):
    global TOKEN, TOKEN_TIME
    if TOKEN is None or time.monotonic() - TOKEN_TIME > 2400:
        result = subprocess.run(["gcloud.cmd", "auth", "print-access-token"], capture_output=True, text=True, check=True)
        TOKEN = result.stdout.strip()
        TOKEN_TIME = time.monotonic()
    headers = {"Authorization": "Bearer " + TOKEN, "Content-Type": "application/json"}
    request = urllib.request.Request("https://compute.googleapis.com/compute/v1/projects/" + PROJECT + "/" + path,
                                    headers=headers, data=None if data is None else json.dumps(data).encode())
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def existing():
    result = api("aggregated/instances?filter=" + urllib.parse.quote("name = " + INSTANCE))
    return [instance for group in result.get("items", {}).values() for instance in group.get("instances", [])]


def persist(value):
    STATE.mkdir(exist_ok=True)
    (STATE / "allocation_status.json").write_text(json.dumps(value, indent=2) + "\n")


def main():
    global PROJECT, INSTANCE, BUCKET, SA, STATE, ZONES
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["project", "instance", "bucket", "service-account"]:
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--state", type=Path, default=STATE)
    parser.add_argument("--zones", nargs="+", default=ZONES)
    args = parser.parse_args()
    PROJECT, INSTANCE, BUCKET, SA = args.project, args.instance, args.bucket, args.service_account
    STATE, ZONES = args.state, args.zones
    started = timestamp()
    attempts = []
    startup = (REPOSITORY / "training/cloud_no_appraisal_startup.sh").read_text()
    for zone in ZONES:
        current = existing()
        if current:
            assert len(current) == 1, "Multiple task instances need reconciliation"
            record = {"status": "allocated", "started_utc": started, "instance": current[0], "attempts": attempts}
            persist(record)
            print(json.dumps({"status": "allocated", "zone": current[0]["zone"], "instance": INSTANCE}), flush=True)
            return
        region = zone.rsplit("-", 1)[0]
        quota = api("regions/" + region)
        l4 = next((q for q in quota.get("quotas", []) if q["metric"] == "NVIDIA_L4_GPUS"), {})
        if l4.get("limit", 0) - l4.get("usage", 0) < 1:
            attempts.append({"zone": zone, "status": "no_regional_quota"})
            continue
        body = {
            "name": INSTANCE, "machineType": f"zones/{zone}/machineTypes/g2-standard-4",
            "guestAccelerators": [{"acceleratorType": f"zones/{zone}/acceleratorTypes/nvidia-l4", "acceleratorCount": 1}],
            "disks": [{"boot": True, "autoDelete": True, "initializeParams": {
                "sourceImage": "projects/deeplearning-platform-release/global/images/common-cu129-ubuntu-2204-nvidia-580-v20261111",
                "diskSizeGb": "150", "diskType": f"zones/{zone}/diskTypes/pd-balanced"}}],
            "networkInterfaces": [{"network": "global/networks/default", "accessConfigs": [{"name": "External NAT", "type": "ONE_TO_ONE_NAT"}]}],
            "serviceAccounts": [{"email": SA, "scopes": ["https://www.googleapis.com/auth/cloud-platform"]}],
            "metadata": {"items": [{"key": "startup-script", "value": startup}, {"key": "shame-bucket", "value": BUCKET}, {"key": "enable-oslogin", "value": "TRUE"}]},
            "labels": {"project": "shame", "experiment": "no-appraisal-v1"},
            "scheduling": {"automaticRestart": False, "onHostMaintenance": "TERMINATE", "provisioningModel": "STANDARD",
                           "maxRunDuration": {"seconds": "43200"}, "instanceTerminationAction": "DELETE"},
        }
        attempt = {"zone": zone, "machine_type": "g2-standard-4", "requested_utc": timestamp()}
        attempts.append(attempt)
        persist({"status": "allocating", "started_utc": started, "attempts": attempts})
        print(json.dumps({"event": "trying_zone", "zone": zone, "utc": timestamp()}), flush=True)
        try:
            operation = api(f"zones/{zone}/instances", body)
            for _ in range(90):
                if operation.get("status") == "DONE":
                    break
                time.sleep(2)
                operation = api(f"zones/{zone}/operations/" + operation["name"])
            if operation.get("status") != "DONE":
                raise RuntimeError("Allocation operation is still running; do not submit another worker")
            if "error" in operation:
                errors = operation["error"].get("errors", [])
                attempt.update(status="failed", errors=errors)
                print(json.dumps({"event": "zone_failed", "zone": zone, "codes": [x.get("code") for x in errors]}), flush=True)
                continue
            worker = api(f"zones/{zone}/instances/{INSTANCE}")
            attempt["status"] = "allocated"
            persist({"status": "allocated", "started_utc": started, "instance": worker, "attempts": attempts})
            print(json.dumps({"status": "allocated", "zone": zone, "instance": INSTANCE, "state": worker["status"]}), flush=True)
            return
        except urllib.error.HTTPError as error:
            details = json.load(error)
            attempt.update(status="http_error", http_status=error.code, details=details)
            print(json.dumps({"event": "allocation_http_error", "zone": zone, "http_status": error.code,
                              "message": details.get("error", {}).get("message", "")[:220]}), flush=True)
            # An HTTP error with a committed VM is reconciled at the next loop.
            if error.code not in (400, 403, 409, 429, 503):
                raise
    persist({"status": "capacity_unavailable", "started_utc": started, "attempts": attempts})
    print(json.dumps({"status": "capacity_unavailable", "zones_tried": len(attempts)}), flush=True)
    raise SystemExit(2)


if __name__ == "__main__":
    main()
