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

For a longer run:

```bash
pixi run python scripts/train_cleaner_ppo.py --n-iters 20 --frames-per-batch 512 --num-envs 16 --ppo-epochs 4 --minibatch-size 256
```

The training script follows the TorchRL MAPPO tutorial structure but adapts it to Cleaner: tutorial-style per-agent observation keys, a masked categorical joint action, configurable centralized/decentralized actor and critic networks, configurable parameter sharing, `Collector`, GAE, and `ClipPPOLoss`. Add `--diversity-coeff 0.01` to include the auxiliary masked inter-agent cross-entropy diversity bonus. This is logged separately from PPO's own entropy bonus so we can compare individual stochastic exploration against inter-agent heterogeneity.

Useful smoke checks:

```bash
pixi run ppo-smoke
pixi run ppo-diversity-smoke
pixi run ppo-mode-smoke
```

For a first single-seed sweep:

```bash
pixi run python scripts/run_cleaner_experiments.py --n-iters 40 --frames-per-batch 1024 --render-frequency 20
pixi run python scripts/plot_cleaner_experiments.py .artifacts/experiments/cleaner_sweep
```

Each run writes `config.json`, `metrics.csv`, `learning_curve.png`, and optional policy GIF checkpoints. Metrics include reward, success rate, dirty-cell fraction, PPO entropy, cross-entropy diversity, PPO KL, gradient norm, and timing breakdowns for collection, value/GAE, loss, diversity bonus, backward, optimizer, and evaluation.
