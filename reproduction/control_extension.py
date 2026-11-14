"""Reproduce the dated no-added-appraisal extension on CPU from verified scores."""
from pathlib import Path
import argparse
import json
import math
import subprocess
import sys

from run import REPO, digest, extract_verified, fetch


def compare(expected, actual, path="results"):
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        assert expected == actual, path
    elif isinstance(expected, (int, float)):
        assert isinstance(actual, (int, float)) and math.isclose(expected, actual, rel_tol=1e-10, abs_tol=1e-12), path
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(expected) == len(actual), path
        for index, (left, right) in enumerate(zip(expected, actual)):
            compare(left, right, path + "/" + str(index))
    elif isinstance(expected, dict):
        assert isinstance(actual, dict) and expected.keys() == actual.keys(), path
        for key in expected:
            compare(expected[key], actual[key], path + "/" + key)
    else:
        raise TypeError(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=REPO / "reproduction/control_artifacts.json")
    parser.add_argument("--workspace", type=Path, default=REPO / ".artifacts/no_appraisal")
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    asset = next(item for item in registry["archives"] if item["id"] == "control-cpu")
    archive = args.archive or fetch(asset["url"], REPO / ".artifacts/downloads" / asset["name"], asset["sha256"])
    assert archive.stat().st_size == asset["bytes"] and digest(archive) == asset["sha256"]
    workspace = args.workspace.resolve()
    count = extract_verified(archive, workspace)
    subprocess.run([sys.executable, str(REPO / "analysis/control_extension.py"),
                    "--input", str(workspace / "results"), "--output", str(workspace / "recomputed"),
                    "--reference", str(workspace / "inputs/original_metrics.csv")], check=True)
    reference = json.loads((workspace / "analysis/control_metrics.json").read_text(encoding="utf-8"))
    actual = json.loads((workspace / "recomputed/control_metrics.json").read_text(encoding="utf-8"))
    compare(reference, actual)
    assert (workspace / "analysis/control_variants.csv").read_bytes() == (workspace / "recomputed/control_variants.csv").read_bytes()
    print(json.dumps({"status": "passed", "verified_files": count, "variants": len(actual["per_variant"]),
                      "models": actual["models"], "cpu_only": True, "workspace": str(workspace)}))


if __name__ == "__main__":
    main()
