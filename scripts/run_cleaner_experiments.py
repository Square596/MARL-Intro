#!/usr/bin/env python3
"""Launch Cleaner PPO sweeps for finding seminar-ready results."""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunSpec:
    name: str
    num_agents: int
    diversity_coeff: float
    diversity_pairs: str
    seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path(".artifacts/experiments/cleaner_sweep"))
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0, help="Single seed, kept for backward compatibility.")
    parser.add_argument("--seeds", type=int, nargs="*", default=None, help="Seeds to run, for example: --seeds 0 1 2")
    parser.add_argument("--small-agents", type=int, nargs="*", default=[2, 3, 4])
    parser.add_argument("--large-agents", type=int, nargs="*", default=[8, 12])
    parser.add_argument("--coeffs", type=float, nargs="*", default=[0.0, 0.0001, 0.0003, 0.001, 0.003])
    parser.add_argument("--large-coeff", type=float, default=0.001)
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--n-iters", type=int, default=300)
    parser.add_argument("--frames-per-batch", type=int, default=4096)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--encoder", choices=["cnn", "mlp"], default="cnn")
    parser.add_argument("--penalty-per-timestep", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=5.0e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--entropy-coeff", type=float, default=1.0e-4)
    parser.add_argument("--diversity-bonus-max", type=float, default=3.0)
    parser.add_argument("--eval-rollout-length", type=int, default=64)
    parser.add_argument("--early-stop-mode-success", type=float, default=0.95)
    parser.add_argument("--early-stop-sample-success", type=float, default=0.85)
    parser.add_argument("--render-frequency", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.experiment_name:
        args.output_root = args.output_root.parent / args.experiment_name
    if args.seeds is None or len(args.seeds) == 0:
        args.seeds = [args.seed]
    return args


def build_specs(args: argparse.Namespace) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for seed, num_agents, coeff in itertools.product(args.seeds, args.small_agents, args.coeffs):
        label = "baseline" if coeff == 0.0 else f"ce{coeff:g}"
        specs.append(
            RunSpec(
                name=f"small_a{num_agents}_{label}_seed{seed}",
                num_agents=num_agents,
                diversity_coeff=coeff,
                diversity_pairs="all",
                seed=seed,
            )
        )
    for seed, num_agents in itertools.product(args.seeds, args.large_agents):
        for pairs in ["all", "sampled"]:
            specs.append(
                RunSpec(
                    name=f"large_a{num_agents}_ce{args.large_coeff:g}_{pairs}_seed{seed}",
                    num_agents=num_agents,
                    diversity_coeff=args.large_coeff,
                    diversity_pairs=pairs,
                    seed=seed,
                )
            )
        specs.append(
            RunSpec(
                name=f"large_a{num_agents}_baseline_seed{seed}",
                num_agents=num_agents,
                diversity_coeff=0.0,
                diversity_pairs="sampled",
                seed=seed,
            )
        )
    return specs


def command_for(args: argparse.Namespace, spec: RunSpec) -> list[str]:
    output_dir = args.output_root / spec.name
    return [
        sys.executable,
        "scripts/train_cleaner_ppo.py",
        "--seed",
        str(spec.seed),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--num-agents",
        str(spec.num_agents),
        "--num-envs",
        str(args.num_envs),
        "--n-iters",
        str(args.n_iters),
        "--frames-per-batch",
        str(args.frames_per_batch),
        "--ppo-epochs",
        str(args.ppo_epochs),
        "--minibatch-size",
        str(args.minibatch_size),
        "--encoder",
        args.encoder,
        "--eval-rollout-length",
        str(args.eval_rollout_length),
        "--penalty-per-timestep",
        str(args.penalty_per_timestep),
        "--early-stop-mode-success",
        str(args.early_stop_mode_success),
        "--early-stop-sample-success",
        str(args.early_stop_sample_success),
        "--lr",
        str(args.lr),
        "--gamma",
        str(args.gamma),
        "--entropy-coeff",
        str(args.entropy_coeff),
        "--render-frequency",
        str(args.render_frequency),
        "--diversity-coeff",
        str(spec.diversity_coeff),
        "--diversity-bonus-max",
        str(args.diversity_bonus_max),
        "--diversity-pairs",
        spec.diversity_pairs,
        "--output-dir",
        str(output_dir),
    ]


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    specs = build_specs(args)
    summary_path = args.output_root / "sweep_config.json"
    summary = {
        "launcher_args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "runs": [asdict(spec) for spec in specs],
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    for spec in specs:
        output_dir = args.output_root / spec.name
        metrics_path = output_dir / "metrics.csv"
        if metrics_path.exists() and not args.overwrite:
            print(f"skip completed: {spec.name}")
            continue
        cmd = command_for(args, spec)
        print("run:", " ".join(cmd))
        if args.dry_run:
            continue
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
