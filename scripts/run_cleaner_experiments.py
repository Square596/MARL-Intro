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
    height: int
    width: int
    num_agents: int
    diversity_coeff: float
    diversity_pairs: str
    seed: int


def parse_size(value: str) -> tuple[int, int]:
    normalized = value.lower().replace("*", "x")
    if "x" not in normalized:
        raise argparse.ArgumentTypeError(f"expected HxW size, got {value!r}")
    height_text, width_text = normalized.split("x", 1)
    try:
        height = int(height_text)
        width = int(width_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected integer HxW size, got {value!r}") from exc
    if height <= 0 or width <= 0:
        raise argparse.ArgumentTypeError(f"size must be positive, got {value!r}")
    return height, width


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path(".artifacts/experiments/cleaner_sweep"))
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0, help="Single seed, kept for backward compatibility.")
    parser.add_argument("--seeds", type=int, nargs="*", default=None, help="Seeds to run, for example: --seeds 0 1 2")
    parser.add_argument("--sizes", type=parse_size, nargs="*", default=None, help="Grid sizes as HxW, for example: --sizes 8x8 10x10")
    parser.add_argument("--height", type=int, default=8, help="Single size height, kept for backward compatibility.")
    parser.add_argument("--width", type=int, default=8, help="Single size width, kept for backward compatibility.")
    parser.add_argument("--agents", type=int, nargs="*", default=None, help="Unified agent counts. Overrides --small-agents/--large-agents.")
    parser.add_argument("--small-agents", type=int, nargs="*", default=[2, 3, 4])
    parser.add_argument("--large-agents", type=int, nargs="*", default=[8, 12])
    parser.add_argument("--coeffs", type=float, nargs="*", default=[0.0, 0.0001, 0.0003, 0.001, 0.003])
    parser.add_argument("--large-coeff", type=float, default=0.001, help="Legacy large-agent CE coefficient when --agents is not used.")
    parser.add_argument("--diversity-pairs", choices=["all", "adjacent", "sampled", "auto"], default="auto")
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
    parser.add_argument("--policy-centralized", choices=["true", "false"], default="false")
    parser.add_argument("--critic-centralized", choices=["true", "false"], default="true")
    parser.add_argument("--share-policy-params", choices=["true", "false"], default="false")
    parser.add_argument("--share-critic-params", choices=["true", "false"], default="true")
    parser.add_argument("--eval-rollout-length", type=int, default=64)
    parser.add_argument("--early-stop-mode-success", type=float, default=0.95)
    parser.add_argument("--early-stop-sample-success", type=float, default=0.85)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--early-stop-min-mode-success", type=float, default=0.95)
    parser.add_argument("--early-stop-min-sample-success", type=float, default=0.75)
    parser.add_argument("--render-frequency", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.experiment_name:
        args.output_root = args.output_root.parent / args.experiment_name
    if args.seeds is None or len(args.seeds) == 0:
        args.seeds = [args.seed]
    if args.sizes is None or len(args.sizes) == 0:
        args.sizes = [(args.height, args.width)]
    if args.agents is None:
        args.agents = []
    return args


def pair_mode_for(num_agents: int, coeff: float, requested: str) -> str:
    if coeff == 0.0:
        return "none"
    if requested == "auto":
        return "all" if num_agents <= 4 else "sampled"
    return requested


def coeffs_for_agent(args: argparse.Namespace, num_agents: int) -> list[float]:
    if args.agents:
        return args.coeffs
    if num_agents in args.small_agents:
        return args.coeffs
    return [0.0, args.large_coeff]


def build_specs(args: argparse.Namespace) -> list[RunSpec]:
    specs: list[RunSpec] = []
    agent_counts = args.agents or sorted(set(args.small_agents + args.large_agents))
    for (height, width), seed, num_agents in itertools.product(args.sizes, args.seeds, agent_counts):
        for coeff in coeffs_for_agent(args, num_agents):
            pairs = pair_mode_for(num_agents, coeff, args.diversity_pairs)
            label = "baseline" if coeff == 0.0 else f"ce{coeff:g}_{pairs}"
            specs.append(
                RunSpec(
                    name=f"h{height}_w{width}_a{num_agents}_{label}_seed{seed}",
                    height=height,
                    width=width,
                    num_agents=num_agents,
                    diversity_coeff=coeff,
                    diversity_pairs="sampled" if pairs == "none" else pairs,
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
        str(spec.height),
        "--width",
        str(spec.width),
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
        "--early-stop-patience",
        str(args.early_stop_patience),
        "--early-stop-min-mode-success",
        str(args.early_stop_min_mode_success),
        "--early-stop-min-sample-success",
        str(args.early_stop_min_sample_success),
        "--policy-centralized",
        args.policy_centralized,
        "--critic-centralized",
        args.critic_centralized,
        "--share-policy-params",
        args.share_policy_params,
        "--share-critic-params",
        args.share_critic_params,
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
        "num_runs": len(specs),
        "runs": [asdict(spec) for spec in specs],
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"planned runs: {len(specs)}")
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
