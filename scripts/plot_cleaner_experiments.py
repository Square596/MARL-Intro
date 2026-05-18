#!/usr/bin/env python3
"""Aggregate Cleaner PPO experiment runs into comparison plots."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def load_rows(run_dir: Path) -> list[dict[str, Any]]:
    metrics_path = run_dir / "metrics.csv"
    config_path = run_dir / "config.json"
    if not metrics_path.exists() or not config_path.exists():
        return []
    config = json.loads(config_path.read_text())
    rows: list[dict[str, Any]] = []
    with metrics_path.open(newline="") as f:
        for row in csv.DictReader(f):
            parsed = {key: float(value) for key, value in row.items()}
            parsed["run"] = run_dir.name
            parsed["num_agents"] = int(config["num_agents"])
            parsed["diversity_coeff"] = float(config.get("diversity_coeff", config.get("madpo_lite_coeff", 0.0)))
            parsed["diversity_pairs"] = config.get("diversity_pairs", "legacy")
            parsed["label"] = label_for(config)
            rows.append(parsed)
    return rows


def label_for(config: dict[str, Any]) -> str:
    coeff = float(config.get("diversity_coeff", config.get("madpo_lite_coeff", 0.0)))
    if coeff == 0.0:
        return f"{config['num_agents']} agents baseline"
    return f"{config['num_agents']} agents CE={coeff:g} {config.get('diversity_pairs', 'legacy')}"


def plot_metric(rows: list[dict[str, Any]], x_key: str, y_key: str, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = sorted({row["label"] for row in rows})
    for label in labels:
        selected = [row for row in rows if row["label"] == label and x_key in row and y_key in row]
        if not selected:
            continue
        selected.sort(key=lambda row: row[x_key])
        ax.plot([row[x_key] for row in selected], [row[y_key] for row in selected], label=label)
    ax.set_xlabel(x_key.replace("_", " "))
    ax.set_ylabel(y_key.replace("_", " "))
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        return
    fields = [
        "run",
        "label",
        "num_agents",
        "diversity_coeff",
        "diversity_pairs",
        "iter",
        "frames",
        "elapsed_sec",
        "eval_return_mean",
        "eval_success_rate",
        "eval_final_dirty_fraction",
        "diversity_cross_entropy",
        "collection_sec",
        "loss_forward_sec",
        "diversity_sec",
        "backward_sec",
        "optimizer_sec",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.experiment_root / "plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(args.experiment_root.iterdir()):
        if run_dir.is_dir():
            rows.extend(load_rows(run_dir))
    if not rows:
        raise SystemExit(f"no metrics found under {args.experiment_root}")
    write_summary(rows, output_dir / "summary.csv")
    plot_metric(rows, "frames", "eval_return_mean", output_dir / "return_vs_steps.png", "Eval return vs env steps")
    plot_metric(rows, "frames", "eval_success_rate", output_dir / "success_vs_steps.png", "Success rate vs env steps")
    plot_metric(rows, "elapsed_sec", "eval_return_mean", output_dir / "return_vs_time.png", "Eval return vs wall-clock time")
    plot_metric(rows, "frames", "diversity_cross_entropy", output_dir / "diversity_vs_steps.png", "Cross-entropy diversity vs env steps")
    print(f"plots written to: {output_dir}")


if __name__ == "__main__":
    main()
