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

## Cleaner Seminar Notebook

The runnable notebook version of the seminar is `notebooks/cleaner_torchrl_ppo_seminar.ipynb`. It is self-contained for teaching: it introduces Jumanji Cleaner, defines the fixed generator and TorchRL-compatible Cleaner subclass, inspects TensorDict rollouts, builds the current CNN actor/critic setup, runs a small PPO loop, and reads sweep summaries when available.

Run it from the pixi environment on `rl2` so JAX, PyTorch, TorchRL, and CUDA match the scripts:

```bash
pixi run notebook
```

If the GPU is shared with a sweep, keep `XLA_PYTHON_CLIENT_PREALLOCATE=false`; the notebook sets this before importing JAX. Jumanji selects the `ipympl` backend in notebooks, so the pixi env includes `ipympl`; the notebook switches back to `%matplotlib inline` after imports for normal inline learning-curve plots.

## Cleaner PPO Training Setup

```bash
pixi run ppo-smoke
```

The current seminar training path is TorchRL PPO on `CustomCleaner`, following the MAPPO tutorial structure but adapted to the JAX/Jumanji Cleaner wrapper. The default setup is intentionally the one used for the active sweep:

- decentralized actors: `--policy-centralized false`;
- no actor parameter sharing: `--share-policy-params false`;
- centralized critic: `--critic-centralized true`;
- shared critic parameters: `--share-critic-params true`;
- CNN encoder over spatial Cleaner channels;
- `num_envs=64`, `frames_per_batch=4096`, `ppo_epochs=4`, `minibatch_size=512`;
- `lr=5e-4`, `gamma=0.995`, `entropy_coeff=1e-4`;
- timestep penalty `--penalty-per-timestep 0.1`.

Actor observations use dirty, wall, own-agent, and all-agents spatial channels. The centralized critic sees dirty, wall, and all-agents channels. Use `--encoder mlp` only as an ablation against the older flattened grid representation.

The MADPO-lite term is an auxiliary masked inter-agent action cross-entropy bonus. PPO entropy remains separate. The CE bonus is unbounded, so the value used in the loss is clipped by default with `--diversity-bonus-max 3.0`; both clipped and raw CE are logged. Use small coefficients first: `0`, `0.0003`, `0.001`, `0.003`. Larger values such as `0.01` are useful mainly as failure examples.

Early stopping uses a soft rolling window. Over the last `--early-stop-patience 5` evals, the mean mode success must be at least `0.95`, mean sample success at least `0.85`, min mode success at least `0.95`, and min sample success at least `0.75`. This avoids stopping on one lucky sampled-policy spike.

Each run writes `config.json`, `metrics.csv`, `learning_curve.png`, `best_policy.pt`, `best_policy.json`, and optional policy GIF checkpoints. Metrics include RewardSum-based completed episode return, collector-window return for debugging, success rate, dirty-cell fraction, completion step, PPO entropy, clipped/raw cross-entropy diversity, PPO KL/clip fraction, gradient norm, and timing breakdowns for collection, value/GAE, loss, diversity bonus, backward, optimizer, and evaluation.

Useful checks:

```bash
pixi run inspect-observation
pixi run heuristic-rollout
pixi run ppo-smoke
pixi run ppo-mlp-smoke
pixi run ppo-diversity-smoke
pixi run ppo-mode-smoke
```

For a single-condition 3-seed baseline:

```bash
pixi run python scripts/run_cleaner_experiments.py \
  --experiment-name stable_cnn_a2 \
  --sizes 8x8 \
  --agents 2 \
  --coeffs 0 \
  --seeds 0 1 2 \
  --n-iters 300
pixi run python scripts/plot_cleaner_experiments.py .artifacts/experiments/stable_cnn_a2 --individual
```

For the active multi-size, multi-agent sweep:

```bash
pixi run python scripts/run_cleaner_experiments.py \
  --experiment-name cleaner_sweep_stage1 \
  --sizes 8x8 10x10 \
  --agents 2 3 4 8 \
  --coeffs 0 0.0003 0.001 0.003 \
  --seeds 0 1 2 \
  --diversity-pairs auto \
  --n-iters 300 \
  --render-frequency 0
pixi run python scripts/plot_cleaner_experiments.py .artifacts/experiments/cleaner_sweep_stage1 --individual
```

This creates one run per size, agent count, coefficient, and seed. The plotter groups by full config identity and aggregates only across seeds, carrying early-stopped runs forward at their final value. Plots are written per environment setting under `plots/by_env/<metric>/h{height}_w{width}_a{agents}.png`, so each chart compares only the coefficient lines for one size/agent count while sharing metric-level x/y limits across environment charts.
