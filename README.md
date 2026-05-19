# MARL Intro Seminar

Small scripts for checking Jumanji Cleaner before building the TorchRL seminar. The current demo avoids the registered `Cleaner-v0` maze generator and uses a deterministic, dimension-aware obstacle layout that is easier to explain in class.

## Setup

```bash
pixi install
```

The project uses pixi because the next seminar steps need conda packages such as PyTorch, TensorDict, and TorchRL. Pixi installs the local `custom_cleaner` package in editable mode, keeps CUDA JAX available through the PyPI `jax[cuda12]` package, and installs CUDA-enabled PyTorch from the `pytorch`/`nvidia` conda channels.

## Cleaner Smoke Test

```bash
pixi run python scripts/cleaner_smoke.py --height 8 --width 8 --num-agents 3 --num-envs 4 --rollout-length 32
```

The script demonstrates:

- constructing `CustomCleaner` with `FixedObstacleGenerator` instead of `jumanji.make("Cleaner-v0")`;
- generating a fixed educational layout from `height`, `width`, and `num_agents`;
- keeping custom agent starts and rectangular-grid action masks consistent with the generated state;
- resetting and stepping one JAX-jitted environment;
- batching reset/step with plain `jax.vmap`;
- using Jumanji's `VmapWrapper` for the same batched API;
- saving a short GIF rollout to `.artifacts/cleaner_rollout.gif`.

A rectangular-grid check is also useful when changing the generator:

```bash
pixi run python scripts/cleaner_smoke.py --height 10 --width 12 --num-agents 2 --num-envs 3 --rollout-length 8 --gif-path .artifacts/cleaner_rollout_10x12.gif
```

## TorchRL GPU Smoke Test

```bash
pixi run python scripts/torchrl_jumanji_smoke.py --height 8 --width 8 --num-agents 3 --num-envs 4
```

This wraps the same `CustomCleaner` instance with TorchRL's `JumanjiWrapper`, checks that both JAX and PyTorch see CUDA, and runs `reset`, `rand_step`, and a short rollout through TensorDicts. TorchRL is imported as `torchrl`; for manual Jumanji vectorization, prefer a wrapper such as `VmapWrapper` or `VmapAutoResetWrapper`.

## PPO Training Prototype

```bash
pixi run ppo-smoke
```

For a longer single-seed sanity run:

```bash
pixi run python scripts/train_cleaner_ppo.py --n-iters 80 --frames-per-batch 4096 --num-envs 64
```

The training script follows the TorchRL MAPPO tutorial structure but adapts it to Cleaner: tutorial-style per-agent observation keys, a masked categorical policy, configurable centralized/decentralized actor and critic networks, configurable parameter sharing, `Collector`, GAE, and `ClipPPOLoss`. The default encoder is now a Jumanji-style spatial CNN: per-agent actor inputs use dirty, wall, own-agent, and all-agents channels; the centralized critic sees dirty, wall, and all-agents channels. Use `--obs-clean-channel true` or `--obs-other-channel true` for richer ablations, and `--encoder mlp` to compare against the older flattened grid representation. The default timestep penalty is relaxed to `--penalty-per-timestep 0.1` for seminar experiments. The default PPO update is intentionally conservative (`--lr 5e-4`, `--ppo-epochs 4`) to reduce sharp policy collapse/recovery spikes. Training stops early once eval mode success is at least `0.95` and eval sample success is at least `0.85`; override this with `--early-stop-mode-success` and `--early-stop-sample-success`. Add `--diversity-coeff 0.001` to include the auxiliary masked inter-agent cross-entropy diversity bonus after the CNN baseline is stable. This is logged separately from PPO's own entropy bonus so we can compare individual stochastic exploration against inter-agent heterogeneity. Each run also saves the best evaluation checkpoint as `best_policy.pt` plus `best_policy.json`.

Useful smoke checks:

```bash
pixi run inspect-observation
pixi run heuristic-rollout
pixi run ppo-smoke
pixi run ppo-mlp-smoke
pixi run ppo-diversity-smoke
pixi run ppo-mode-smoke
```

For a first seminar-style 3-seed baseline:

```bash
pixi run python scripts/run_cleaner_experiments.py \
  --experiment-name stable_cnn_a2 \
  --seeds 0 1 2 \
  --small-agents 2 \
  --large-agents \
  --coeffs 0 \
  --height 8 \
  --width 8 \
  --n-iters 300
pixi run python scripts/plot_cleaner_experiments.py .artifacts/experiments/stable_cnn_a2 --individual
```

For the first baseline-vs-diversity comparison, keep the same command but use `--coeffs 0 0.0001 0.0003 0.001`. The plotter writes mean-plus-std curves across seeds and per-run summaries under `plots/`. Each run writes `config.json`, `metrics.csv`, `learning_curve.png`, `best_policy.pt`, `best_policy.json`, and optional policy GIF checkpoints. Metrics include RewardSum-based completed episode return, collector-window return for debugging, success rate, dirty-cell fraction, completion step, PPO entropy, clipped and raw cross-entropy diversity, PPO KL/clip fraction, gradient norm, and timing breakdowns for collection, value/GAE, loss, diversity bonus, backward, optimizer, and evaluation. The CE diversity bonus used in the loss is clipped by `--diversity-bonus-max 3.0` by default because raw cross-entropy is unbounded and can destabilize PPO at large coefficients.
