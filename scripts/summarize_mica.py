"""Summarize upstream target results; no model, data, or algorithm changes."""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean


def summarize(run_dir):
    config = json.loads((run_dir / "target_config.json").read_text())
    groups = config["dataset_configs"]
    expected_ids = {v["target_id"] for g in groups for v in g["variants"]}
    if expected_ids != set(range(1, 26)):
        raise ValueError("Expected exactly target IDs 1..25.")
    methods = {
        "Tunable MagMax (Label)": (
            "merge_max_abs_masked_with_targetdata/labels",
            "merge_max_abs_masked_with_targetdata_lambda0.5_labels_target{id}_seed5.json",
        ),
        "MagMax": (
            "merge_max_abs",
            "merge_max_abs_lambda0.5_target{id}_seed5.json",
        ),
    }
    records = {}
    for method, (directory, pattern) in methods.items():
        records[method] = {}
        for target_id in sorted(expected_ids):
            path = run_dir / "results" / directory / pattern.format(id=target_id)
            data = json.loads(path.read_text())
            info = data["target_dataset_info"]
            if info["target_id"] != target_id or info["num_target_data"] != 500:
                raise ValueError(f"Unexpected target protocol: {path}")
            if data["model"] != "ViT-B-16" or data["scaling_coefficient"] != 0.5:
                raise ValueError(f"Unexpected model or scaling: {path}")
            accuracy = float(data["overall_accuracy"])
            if not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
                raise ValueError(f"Invalid accuracy: {path}")
            records[method][target_id] = data
    # Fair comparison requires the same sampled tasks, ratios and target seeds.
    first, second = records.values()
    for target_id in sorted(expected_ids):
        if first[target_id]["target_dataset_info"] != second[target_id]["target_dataset_info"]:
            raise ValueError(f"Methods used different target sampling: {target_id}")

    # Upstream JSON lists (0.6,0.2,0.2) before (0.4,0.4,0.2).
    # Order the output like paper Table 2, preserving upstream target IDs/ratios.
    paper_groups = sorted(groups, key=lambda g: {
        (0.5, 0.5): 1, (0.8, 0.2): 2, (0.33, 0.33, 0.33): 3,
        (0.4, 0.4, 0.2): 4, (0.6, 0.2, 0.2): 5,
    }[tuple(g["ratio_task_to_be_fetched"])])
    rows = []
    for method, targets in records.items():
        for index, group in enumerate(paper_groups, 1):
            ids = [v["target_id"] for v in group["variants"]]
            rows.append({
                "method": method, "target_environment": f"D_tar_{index}",
                "mixing_ratio": "/".join(map(str, group["ratio_task_to_be_fetched"])),
                "variants": len(ids),
                "top1_percent": 100 * mean(targets[i]["overall_accuracy"] for i in ids),
            })
        rows.append({
            "method": method, "target_environment": "Average", "mixing_ratio": "",
            "variants": len(targets),
            "top1_percent": 100 * mean(d["overall_accuracy"] for d in targets.values()),
        })
    report = {
        "dataset": "CIFAR100", "tasks": 20, "task_sequence": "A", "seed": 5,
        "training_runs": 1, "metric": "overall top-1 accuracy, mean over target variants",
        "note": "Single seed; no standard deviation across training seeds. Ratios and integer meta rounding follow upstream code.",
        "results": rows,
    }
    (run_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['method']:<24} {row['target_environment']:<10} {row['top1_percent']:.2f}%")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        summarize(args.run_dir)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f"[STOP] Summary incomplete or invalid: {exc}") from exc
