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

GROUP_KEYS = [
    "height",
    "width",
    "num_agents",
    "diversity_coeff",
    "diversity_pairs",
    "encoder",
    "policy_centralized",
    "critic_centralized",
    "share_policy_params",
    "share_critic_params",
]
ENV_KEYS = ["height", "width", "num_agents"]

PLOTS = [
    ("eval_mode_return_mean", "mode_return_vs_steps", "Mode eval return vs env steps"),
    ("eval_sample_return_mean", "sample_return_vs_steps", "Sample eval return vs env steps"),
    ("eval_mode_success_rate", "mode_success_vs_steps", "Mode success rate vs env steps"),
    ("eval_sample_success_rate", "sample_success_vs_steps", "Sample success rate vs env steps"),
    ("eval_sample_final_dirty_fraction", "sample_dirty_fraction_vs_steps", "Sample final dirty fraction vs env steps"),
    ("eval_mode_completion_step_mean", "mode_completion_step_vs_steps", "Mode completion step vs env steps"),
    ("eval_sample_completion_step_mean", "sample_completion_step_vs_steps", "Sample completion step vs env steps"),
    ("eval_sample_completion_step_success_mean", "sample_success_completion_step_vs_steps", "Sample solved-episode completion step vs env steps"),
    ("clip_fraction", "clip_fraction_vs_steps", "PPO clip fraction vs env steps"),
    ("diversity_cross_entropy", "diversity_vs_steps", "Clipped cross-entropy diversity vs env steps"),
    ("diversity_cross_entropy_raw", "diversity_raw_vs_steps", "Raw cross-entropy diversity vs env steps"),
    ("loss_forward_sec", "loss_forward_time_vs_steps", "Loss forward time vs env steps"),
    ("diversity_sec", "diversity_time_vs_steps", "Diversity bonus time vs env steps"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--individual", action="store_true", help="Draw individual seed curves behind the aggregate.")
    parser.add_argument("--carry-forward", action="store_true", default=True, help="Carry early-stopped runs forward at their final value.")
    return parser.parse_args()


def finite(value: float) -> bool:
    return not math.isnan(value) and not math.isinf(value)


def config_value(config: dict[str, Any], key: str) -> Any:
    if key == "diversity_coeff":
        return float(config.get("diversity_coeff", config.get("madpo_lite_coeff", 0.0)))
    if key == "diversity_pairs":
        return "none" if float(config.get("diversity_coeff", 0.0)) == 0.0 else config.get("diversity_pairs", "legacy")
    if key in {"height", "width", "num_agents"}:
        return int(config[key])
    if key in {"policy_centralized", "critic_centralized", "share_policy_params", "share_critic_params"}:
        return bool(config.get(key, False))
    return config.get(key, "")


def group_key_for(config: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(config_value(config, key) for key in GROUP_KEYS)


def env_key_for(config: dict[str, Any]) -> tuple[int, int, int]:
    return tuple(config_value(config, key) for key in ENV_KEYS)  # type: ignore[return-value]


def env_label_from_key(env_key: tuple[int, int, int]) -> str:
    height, width, agents = env_key
    return f"{height}x{width}, {agents} agents"


def env_slug_from_key(env_key: tuple[int, int, int]) -> str:
    height, width, agents = env_key
    return f"h{height}_w{width}_a{agents}"


def method_label_for(config: dict[str, Any]) -> str:
    coeff = float(config.get("diversity_coeff", config.get("madpo_lite_coeff", 0.0)))
    pairs = "none" if coeff == 0.0 else config.get("diversity_pairs", "legacy")
    if coeff == 0.0:
        return "baseline"
    return f"CE={coeff:g} {pairs}"


def full_label_for(config: dict[str, Any]) -> str:
    env = env_label_from_key(env_key_for(config))
    actor = "dec" if not bool(config.get("policy_centralized", False)) else "cent"
    share = "unshared" if not bool(config.get("share_policy_params", False)) else "shared"
    return f"{env} {method_label_for(config)} {actor}-{share}"


def load_rows(run_dir: Path) -> list[dict[str, Any]]:
    metrics_path = run_dir / "metrics.csv"
    config_path = run_dir / "config.json"
    if not metrics_path.exists() or not config_path.exists():
        return []
    config = json.loads(config_path.read_text())
    group_key = group_key_for(config)
    env_key = env_key_for(config)
    rows: list[dict[str, Any]] = []
    with metrics_path.open(newline="") as f:
        for row in csv.DictReader(f):
            parsed = {key: float(value) for key, value in row.items()}
            parsed["run"] = run_dir.name
            parsed["seed"] = int(config.get("seed", -1))
            parsed["height"] = env_key[0]
            parsed["width"] = env_key[1]
            parsed["num_agents"] = env_key[2]
            parsed["diversity_coeff"] = float(config.get("diversity_coeff", config.get("madpo_lite_coeff", 0.0)))
            parsed["diversity_pairs"] = config_value(config, "diversity_pairs")
            parsed["encoder"] = config.get("encoder", "mlp")
            parsed["policy_centralized"] = bool(config.get("policy_centralized", False))
            parsed["critic_centralized"] = bool(config.get("critic_centralized", False))
            parsed["share_policy_params"] = bool(config.get("share_policy_params", False))
            parsed["share_critic_params"] = bool(config.get("share_critic_params", False))
            parsed["group_key"] = group_key
            parsed["env_key"] = env_key
            parsed["label"] = full_label_for(config)
            parsed["method_label"] = method_label_for(config)
            rows.append(parsed)
    return rows


def load_all_rows(root: Path) -> list[dict[str, Any]]:
    rows = load_rows(root)
    if rows:
        return rows
    all_rows: list[dict[str, Any]] = []
    for run_dir in sorted(root.iterdir()):
        if run_dir.is_dir():
            all_rows.extend(load_rows(run_dir))
    return all_rows


def run_series(rows: list[dict[str, Any]], y_key: str) -> dict[str, list[dict[str, Any]]]:
    by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if y_key in row and finite(row[y_key]):
            by_run[row["run"]].append(row)
    for selected in by_run.values():
        selected.sort(key=lambda row: row["frames"])
    return by_run


def aggregate_by_frame(rows: list[dict[str, Any]], y_key: str, carry_forward: bool) -> dict[tuple[Any, ...], list[dict[str, float]]]:
    grouped_rows: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_rows[row["group_key"]].append(row)

    aggregates: dict[tuple[Any, ...], list[dict[str, float]]] = {}
    for group_key, selected_rows in grouped_rows.items():
        series_by_run = run_series(selected_rows, y_key)
        if not series_by_run:
            continue
        frames = sorted({row["frames"] for row in selected_rows if y_key in row and finite(row[y_key])})
        points: list[dict[str, float]] = []
        for frame in frames:
            values = []
            for series in series_by_run.values():
                exact = [row for row in series if row["frames"] == frame]
                if exact:
                    values.append(exact[-1][y_key])
                elif carry_forward and frame > series[-1]["frames"]:
                    values.append(series[-1][y_key])
            if not values:
                continue
            mean = statistics.fmean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            points.append({"frames": frame, "mean": mean, "std": std, "n": float(len(values))})
        if points:
            aggregates[group_key] = points
    return aggregates


def metadata_by_group(rows: list[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    metadata = {}
    for row in rows:
        metadata.setdefault(
            row["group_key"],
            {
                "env_key": row["env_key"],
                "label": row["label"],
                "method_label": row["method_label"],
                "height": row["height"],
                "width": row["width"],
                "num_agents": row["num_agents"],
                "diversity_coeff": row["diversity_coeff"],
                "diversity_pairs": row["diversity_pairs"],
            },
        )
    return metadata


def metric_limits(aggregates: dict[tuple[Any, ...], list[dict[str, float]]]) -> tuple[tuple[float, float], tuple[float, float]]:
    xs: list[float] = []
    ys: list[float] = []
    for points in aggregates.values():
        for point in points:
            xs.append(point["frames"])
            ys.append(point["mean"])
            ys.append(point["mean"] - point["std"])
            ys.append(point["mean"] + point["std"])
    if not xs or not ys:
        return (0.0, 1.0), (0.0, 1.0)
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_min == x_max:
        x_pad = max(abs(x_min) * 0.05, 1.0)
        x_min -= x_pad
        x_max += x_pad
    if y_min == y_max:
        y_pad = max(abs(y_min) * 0.05, 1.0)
        y_min -= y_pad
        y_max += y_pad
    else:
        y_pad = (y_max - y_min) * 0.05
        y_min -= y_pad
        y_max += y_pad
    return (x_min, x_max), (y_min, y_max)


def plot_metric_by_env(
    rows: list[dict[str, Any]],
    y_key: str,
    slug: str,
    title: str,
    output_dir: Path,
    individual: bool = False,
    carry_forward: bool = True,
) -> None:
    aggregates = aggregate_by_frame(rows, y_key, carry_forward=carry_forward)
    if not aggregates:
        return
    metadata = metadata_by_group(rows)
    x_limits, y_limits = metric_limits(aggregates)
    env_keys = sorted({meta["env_key"] for meta in metadata.values()})
    metric_dir = output_dir / "by_env" / slug
    metric_dir.mkdir(parents=True, exist_ok=True)
    for env_key in env_keys:
        env_rows = [row for row in rows if row["env_key"] == env_key]
        env_aggregates = {group_key: points for group_key, points in aggregates.items() if metadata[group_key]["env_key"] == env_key}
        if not env_aggregates:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        if individual:
            for selected in run_series(env_rows, y_key).values():
                if selected:
                    ax.plot(
                        [row["frames"] for row in selected],
                        [row[y_key] for row in selected],
                        color="0.75",
                        linewidth=0.8,
                        alpha=0.3,
                    )
        for group_key in sorted(env_aggregates, key=lambda key: (metadata[key]["diversity_coeff"], metadata[key]["diversity_pairs"])):
            points = env_aggregates[group_key]
            x = [point["frames"] for point in points]
            mean = [point["mean"] for point in points]
            std = [point["std"] for point in points]
            lower = [m - s for m, s in zip(mean, std)]
            upper = [m + s for m, s in zip(mean, std)]
            ax.plot(x, mean, label=metadata[group_key]["method_label"])
            if max(point["n"] for point in points) > 1:
                ax.fill_between(x, lower, upper, alpha=0.15)
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        ax.set_xlabel("environment steps")
        ax.set_ylabel(y_key.replace("_", " "))
        ax.set_title(f"{title}: {env_label_from_key(env_key)}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(metric_dir / f"{env_slug_from_key(env_key)}.png", dpi=180)
        plt.close(fig)


def write_summary(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        return
    fields = [
        "run",
        "label",
        "method_label",
        "seed",
        "height",
        "width",
        "num_agents",
        "diversity_coeff",
        "diversity_pairs",
        "encoder",
        "policy_centralized",
        "critic_centralized",
        "share_policy_params",
        "share_critic_params",
        "iter",
        "frames",
        "elapsed_sec",
        "episode_return_mean",
        "completed_episodes",
        "eval_mode_return_mean",
        "eval_mode_success_rate",
        "eval_mode_completion_step_mean",
        "eval_sample_return_mean",
        "eval_sample_success_rate",
        "eval_sample_completion_step_mean",
        "eval_sample_completion_step_success_mean",
        "eval_sample_final_dirty_fraction",
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


def first_threshold(rows: list[dict[str, Any]], key: str, threshold: float) -> tuple[float | str, float | str]:
    selected = sorted([row for row in rows if key in row and finite(row[key])], key=lambda row: row["frames"])
    for row in selected:
        if row[key] >= threshold:
            return row["frames"], row["elapsed_sec"]
    return "", ""


def write_final_summary(rows: list[dict[str, Any]], output_path: Path) -> None:
    by_group_run: dict[tuple[tuple[Any, ...], str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group_run[(row["group_key"], row["run"])].append(row)

    by_group: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    first_success_by_group: dict[tuple[Any, ...], list[tuple[float | str, float | str]]] = defaultdict(list)
    for (group_key, _run), selected in by_group_run.items():
        selected.sort(key=lambda row: row["frames"])
        by_group[group_key].append(selected[-1])
        first_success_by_group[group_key].append(first_threshold(selected, "eval_sample_success_rate", 0.85))

    fields = [
        "label",
        "method_label",
        "num_seeds",
        "height",
        "width",
        "num_agents",
        "diversity_coeff",
        "diversity_pairs",
        "final_sample_success_mean",
        "final_sample_success_std",
        "first_sample_success_frames_mean",
        "first_sample_success_time_mean",
        "final_mode_success_mean",
        "final_sample_completion_step_mean",
        "final_mode_completion_step_mean",
        "final_return_mean",
        "final_elapsed_sec_mean",
        "diversity_time_mean",
        "loss_forward_time_mean",
    ]
    metadata = metadata_by_group(rows)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for group_key, selected in sorted(by_group.items(), key=lambda item: metadata[item[0]]["label"]):
            sample_sr = [row["eval_sample_success_rate"] for row in selected if finite(row["eval_sample_success_rate"])]
            mode_sr = [row["eval_mode_success_rate"] for row in selected if finite(row["eval_mode_success_rate"])]
            sample_steps = [row["eval_sample_completion_step_mean"] for row in selected if finite(row["eval_sample_completion_step_mean"])]
            mode_steps = [row["eval_mode_completion_step_mean"] for row in selected if finite(row["eval_mode_completion_step_mean"])]
            returns = [row["eval_sample_return_mean"] for row in selected if finite(row["eval_sample_return_mean"])]
            elapsed = [row["elapsed_sec"] for row in selected if finite(row["elapsed_sec"])]
            diversity_time = [row["diversity_sec"] for row in selected if finite(row["diversity_sec"])]
            loss_time = [row["loss_forward_sec"] for row in selected if finite(row["loss_forward_sec"])]
            first_frames = [value[0] for value in first_success_by_group[group_key] if value[0] != ""]
            first_times = [value[1] for value in first_success_by_group[group_key] if value[1] != ""]
            meta = metadata[group_key]
            writer.writerow(
                {
                    "label": meta["label"],
                    "method_label": meta["method_label"],
                    "num_seeds": len(selected),
                    "height": meta["height"],
                    "width": meta["width"],
                    "num_agents": meta["num_agents"],
                    "diversity_coeff": meta["diversity_coeff"],
                    "diversity_pairs": meta["diversity_pairs"],
                    "final_sample_success_mean": statistics.fmean(sample_sr) if sample_sr else "",
                    "final_sample_success_std": statistics.stdev(sample_sr) if len(sample_sr) > 1 else 0.0,
                    "first_sample_success_frames_mean": statistics.fmean(first_frames) if first_frames else "",
                    "first_sample_success_time_mean": statistics.fmean(first_times) if first_times else "",
                    "final_mode_success_mean": statistics.fmean(mode_sr) if mode_sr else "",
                    "final_sample_completion_step_mean": statistics.fmean(sample_steps) if sample_steps else "",
                    "final_mode_completion_step_mean": statistics.fmean(mode_steps) if mode_steps else "",
                    "final_return_mean": statistics.fmean(returns) if returns else "",
                    "final_elapsed_sec_mean": statistics.fmean(elapsed) if elapsed else "",
                    "diversity_time_mean": statistics.fmean(diversity_time) if diversity_time else "",
                    "loss_forward_time_mean": statistics.fmean(loss_time) if loss_time else "",
                }
            )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.experiment_root / "plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_all_rows(args.experiment_root)
    if not rows:
        raise SystemExit(f"no metrics found under {args.experiment_root}")
    write_summary(rows, output_dir / "summary.csv")
    write_final_summary(rows, output_dir / "final_summary.csv")
    for y_key, slug, title in PLOTS:
        plot_metric_by_env(rows, y_key, slug, title, output_dir, args.individual, args.carry_forward)
    print(f"plots written to: {output_dir}")


if __name__ == "__main__":
    main()
