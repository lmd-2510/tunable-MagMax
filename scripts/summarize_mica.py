"""Summarize upstream target results; no model, data, or algorithm changes."""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean


def summarize(run_dir, allow_unpaired_targets=False):
    config = json.loads((run_dir / "target_config.json").read_text())
    groups = config["dataset_configs"]
    expected_ids = {v["target_id"] for g in groups for v in g["variants"]}
    if expected_ids != set(range(1, 26)):
        raise ValueError("Expected exactly target IDs 1..25.")
    protocol = {
        v["target_id"]: (g, v)
        for g in groups for v in g["variants"]
    }
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
            group, variant = protocol[target_id]
            if (info["seed_target_data"] != variant["random_seed"] or
                    info["num_task_to_be_fetched"] != group["num_task_to_be_fetched"] or
                    info["ratio_task_to_be_fetched"] != group["ratio_task_to_be_fetched"]):
                raise ValueError(f"Target seed or distribution differs from config: {path}")
            if data["model"] != "ViT-B-16" or data["scaling_coefficient"] != 0.5:
                raise ValueError(f"Unexpected model or scaling: {path}")
            accuracy = float(data["overall_accuracy"])
            if not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
                raise ValueError(f"Invalid accuracy: {path}")
            records[method][target_id] = data
    # Fair comparison requires the same sampled tasks, ratios and target seeds.
    first, second = records.values()
    sampling_differences = []
    for target_id in sorted(expected_ids):
        first_info = first[target_id]["target_dataset_info"]
        second_info = second[target_id]["target_dataset_info"]
        if first_info != second_info:
            fields = sorted(k for k in first_info.keys() | second_info.keys()
                            if first_info.get(k) != second_info.get(k))
            sampling_differences.append({
                "target_id": target_id, "different_fields": fields,
                "Tunable MagMax (Label)": first_info, "MagMax": second_info,
            })
    if sampling_differences and not allow_unpaired_targets:
        ids = [d["target_id"] for d in sampling_differences]
        raise ValueError(f"Methods used different target sampling: {ids}. "
                         "Use --allow-unpaired-targets to report existing results with this limitation.")
    if sampling_differences:
        ids = [d["target_id"] for d in sampling_differences]
        print(f"[WARNING] Different target sampling for IDs {ids}. "
              "Each method is averaged over its own target samples; these are not paired evaluations.")

    # Upstream JSON lists (0.6,0.2,0.2) before (0.4,0.4,0.2).
    # Order the output like paper Table 2, preserving upstream target IDs/ratios.
    paper_groups = sorted(groups, key=lambda g: {
        (0.5, 0.5): 1, (0.8, 0.2): 2, (0.33, 0.33, 0.33): 3,
        (0.4, 0.4, 0.2): 4, (0.6, 0.2, 0.2): 5,
    }[tuple(g["ratio_task_to_be_fetched"])])
    rows = []
    paired = not sampling_differences
    for method, targets in records.items():
        for index, group in enumerate(paper_groups, 1):
            ids = [v["target_id"] for v in group["variants"]]
            rows.append({
                "method": method, "target_environment": f"D_tar_{index}",
                "mixing_ratio": "/".join(map(str, group["ratio_task_to_be_fetched"])),
                "variants": len(ids),
                "top1_percent": 100 * mean(targets[i]["overall_accuracy"] for i in ids),
                "target_sampling": "paired" if paired else "unpaired",
            })
        rows.append({
            "method": method, "target_environment": "Average", "mixing_ratio": "",
            "variants": len(targets),
            "top1_percent": 100 * mean(d["overall_accuracy"] for d in targets.values()),
            "target_sampling": "paired" if paired else "unpaired",
        })
    report = {
        "dataset": "CIFAR100", "tasks": 20, "task_sequence": "A", "seed": 5,
        "training_runs": 1, "metric": "overall top-1 accuracy, mean over target variants",
        "note": "Single seed; no standard deviation across training seeds. Ratios and integer meta rounding follow upstream code.",
        "paired_targets": paired,
        "mismatched_target_ids": [d["target_id"] for d in sampling_differences],
        "sampling_differences": sampling_differences,
        "comparison_note": (
            "The methods used different sampled tasks for some target variants. "
            "Results summarize each method's existing evaluations independently; "
            "do not treat their difference as a comparison on identical target samples."
            if not paired else "Both methods have matching target sampling metadata."
        ),
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
    parser.add_argument("--allow-unpaired-targets", action="store_true",
                        help="Report existing results even when target sampling differs, with an explicit limitation.")
    args = parser.parse_args()
    try:
        summarize(args.run_dir, allow_unpaired_targets=args.allow_unpaired_targets)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f"[STOP] Summary incomplete or invalid: {exc}") from exc
