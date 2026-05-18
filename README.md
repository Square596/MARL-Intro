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

The training script follows the TorchRL MAPPO tutorial structure but adapts it to Cleaner: a shared masked categorical actor for the agents, a centralized critic for the shared team reward, `Collector`, GAE, and `ClipPPOLoss`. Add `--madpo-lite-coeff 0.01` to include the auxiliary adjacent-agent masked action-distribution divergence penalty. Metrics are written to `metrics.csv` and `learning_curve.png` in the selected output directory.
