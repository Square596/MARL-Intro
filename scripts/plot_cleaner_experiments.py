#!/usr/bin/env python3
"""Aggregate Cleaner PPO experiment runs into comparison plots."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--individual", action="store_true", help="Draw individual seed curves behind the aggregate.")
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
            parsed["seed"] = int(config.get("seed", -1))
            parsed["num_agents"] = int(config["num_agents"])
            parsed["diversity_coeff"] = float(config.get("diversity_coeff", config.get("madpo_lite_coeff", 0.0)))
            parsed["diversity_pairs"] = config.get("diversity_pairs", "legacy")
            parsed["encoder"] = config.get("encoder", "mlp")
            parsed["label"] = label_for(config)
            rows.append(parsed)
    return rows


def label_for(config: dict[str, Any]) -> str:
    coeff = float(config.get("diversity_coeff", config.get("madpo_lite_coeff", 0.0)))
    encoder = config.get("encoder", "mlp")
    agents = config["num_agents"]
    if coeff == 0.0:
        return f"{agents} agents {encoder} baseline"
    return f"{agents} agents {encoder} CE={coeff:g} {config.get('diversity_pairs', 'legacy')}"


def finite(value: float) -> bool:
    return not math.isnan(value) and not math.isinf(value)


def aggregate_by_frame(rows: list[dict[str, Any]], y_key: str) -> dict[str, list[dict[str, float]]]:
    grouped: dict[tuple[str, float], list[float]] = defaultdict(list)
    for row in rows:
        if y_key not in row or not finite(row[y_key]):
            continue
        grouped[(row["label"], row["frames"])].append(row[y_key])
    by_label: dict[str, list[dict[str, float]]] = defaultdict(list)
    for (label, frames), values in grouped.items():
        mean = statistics.fmean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        by_label[label].append({"frames": frames, "mean": mean, "std": std, "n": float(len(values))})
    for points in by_label.values():
        points.sort(key=lambda point: point["frames"])
    return by_label


def plot_metric(rows: list[dict[str, Any]], y_key: str, output_path: Path, title: str, individual: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = sorted({row["label"] for row in rows})
    if individual:
        for label in labels:
            runs = sorted({row["run"] for row in rows if row["label"] == label})
            for run in runs:
                selected = [row for row in rows if row["run"] == run and y_key in row and finite(row[y_key])]
                selected.sort(key=lambda row: row["frames"])
                if selected:
                    ax.plot(
                        [row["frames"] for row in selected],
                        [row[y_key] for row in selected],
                        color="0.75",
                        linewidth=0.8,
                        alpha=0.35,
                    )
    aggregates = aggregate_by_frame(rows, y_key)
    for label in labels:
        points = aggregates.get(label, [])
        if not points:
            continue
        x = [point["frames"] for point in points]
        mean = [point["mean"] for point in points]
        std = [point["std"] for point in points]
        lower = [m - s for m, s in zip(mean, std)]
        upper = [m + s for m, s in zip(mean, std)]
        ax.plot(x, mean, label=label)
        if max(point["n"] for point in points) > 1:
            ax.fill_between(x, lower, upper, alpha=0.15)
    ax.set_xlabel("environment steps")
    ax.set_ylabel(y_key.replace("_", " "))
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    handles, legend_labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, legend_labels, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        return
    fields = [
        "run",
        "label",
        "seed",
        "num_agents",
        "diversity_coeff",
        "diversity_pairs",
        "encoder",
        "iter",
        "frames",
        "elapsed_sec",
        "collector_window_return_mean",
        "batch_return_mean",
        "episode_return_mean",
        "completed_episodes",
        "eval_return_mean",
        "eval_success_rate",
        "eval_final_dirty_fraction",
        "eval_mode_return_mean",
        "eval_mode_success_rate",
        "eval_mode_final_dirty_fraction",
        "eval_mode_completion_step_mean",
        "eval_mode_completion_step_success_mean",
        "eval_sample_return_mean",
        "eval_sample_success_rate",
        "eval_sample_final_dirty_fraction",
        "eval_sample_completion_step_mean",
        "eval_sample_completion_step_success_mean",
        "diversity_cross_entropy",
        "diversity_cross_entropy_raw",
        "collection_sec",
        "loss_forward_sec",
        "diversity_sec",
        "backward_sec",
        "optimizer_sec",
        "clip_fraction",
        "entropy",
        "grad_norm",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_final_summary(rows: list[dict[str, Any]], output_path: Path) -> None:
    finals: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in sorted({row["run"] for row in rows}):
        selected = [row for row in rows if row["run"] == run]
        if selected:
            final = max(selected, key=lambda row: row["frames"])
            finals[final["label"]].append(final)
    fields = [
        "label",
        "num_seeds",
        "final_sample_success_mean",
        "final_sample_success_std",
        "final_mode_success_mean",
        "final_sample_completion_step_mean",
        "final_mode_completion_step_mean",
        "final_return_mean",
        "final_elapsed_sec_mean",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for label, selected in sorted(finals.items()):
            sample_sr = [row["eval_sample_success_rate"] for row in selected if finite(row["eval_sample_success_rate"])]
            mode_sr = [row["eval_mode_success_rate"] for row in selected if finite(row["eval_mode_success_rate"])]
            sample_steps = [row["eval_sample_completion_step_mean"] for row in selected if "eval_sample_completion_step_mean" in row and finite(row["eval_sample_completion_step_mean"])]
            mode_steps = [row["eval_mode_completion_step_mean"] for row in selected if "eval_mode_completion_step_mean" in row and finite(row["eval_mode_completion_step_mean"])]
            returns = [row["eval_sample_return_mean"] for row in selected if finite(row["eval_sample_return_mean"])]
            elapsed = [row["elapsed_sec"] for row in selected if finite(row["elapsed_sec"])]
            writer.writerow(
                {
                    "label": label,
                    "num_seeds": len(selected),
                    "final_sample_success_mean": statistics.fmean(sample_sr) if sample_sr else "",
                    "final_sample_success_std": statistics.stdev(sample_sr) if len(sample_sr) > 1 else 0.0,
                    "final_mode_success_mean": statistics.fmean(mode_sr) if mode_sr else "",
                    "final_sample_completion_step_mean": statistics.fmean(sample_steps) if sample_steps else "",
                    "final_mode_completion_step_mean": statistics.fmean(mode_steps) if mode_steps else "",
                    "final_return_mean": statistics.fmean(returns) if returns else "",
                    "final_elapsed_sec_mean": statistics.fmean(elapsed) if elapsed else "",
                }
            )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.experiment_root / "plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = load_rows(args.experiment_root)
    if not rows:
        for run_dir in sorted(args.experiment_root.iterdir()):
            if run_dir.is_dir():
                rows.extend(load_rows(run_dir))
    if not rows:
        raise SystemExit(f"no metrics found under {args.experiment_root}")
    write_summary(rows, output_dir / "summary.csv")
    write_final_summary(rows, output_dir / "final_summary.csv")
    plot_metric(rows, "eval_mode_return_mean", output_dir / "mode_return_vs_steps.png", "Mode eval return vs env steps", args.individual)
    plot_metric(rows, "eval_sample_return_mean", output_dir / "sample_return_vs_steps.png", "Sample eval return vs env steps", args.individual)
    plot_metric(rows, "eval_mode_success_rate", output_dir / "mode_success_vs_steps.png", "Mode success rate vs env steps", args.individual)
    plot_metric(rows, "eval_sample_success_rate", output_dir / "sample_success_vs_steps.png", "Sample success rate vs env steps", args.individual)
    plot_metric(rows, "eval_sample_final_dirty_fraction", output_dir / "sample_dirty_fraction_vs_steps.png", "Sample final dirty fraction vs env steps", args.individual)
    plot_metric(rows, "eval_mode_completion_step_mean", output_dir / "mode_completion_step_vs_steps.png", "Mode completion step vs env steps", args.individual)
    plot_metric(rows, "eval_sample_completion_step_mean", output_dir / "sample_completion_step_vs_steps.png", "Sample completion step vs env steps", args.individual)
    plot_metric(rows, "eval_sample_completion_step_success_mean", output_dir / "sample_success_completion_step_vs_steps.png", "Sample solved-episode completion step vs env steps", args.individual)
    plot_metric(rows, "clip_fraction", output_dir / "clip_fraction_vs_steps.png", "PPO clip fraction vs env steps", args.individual)
    plot_metric(rows, "diversity_cross_entropy", output_dir / "diversity_vs_steps.png", "Clipped cross-entropy diversity vs env steps", args.individual)
    plot_metric(rows, "diversity_cross_entropy_raw", output_dir / "diversity_raw_vs_steps.png", "Raw cross-entropy diversity vs env steps", args.individual)
    plot_metric(rows, "loss_forward_sec", output_dir / "loss_forward_time_vs_steps.png", "Loss forward time vs env steps", args.individual)
    plot_metric(rows, "diversity_sec", output_dir / "diversity_time_vs_steps.png", "Diversity bonus time vs env steps", args.individual)
    print(f"plots written to: {output_dir}")


if __name__ == "__main__":
    main()
