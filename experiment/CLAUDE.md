# CLAUDE.md — experiment/

Guidance for Claude Code when working inside `experiment/`. Read fully before
touching code here. The parent folder holds the coursework guide
(`../CwkGuide-ARS-26-27-v1.2.pdf`), the reference table
(`../reference_table_O2O_RL.xlsx`, `.bib`) and the papers (`../reference/`).

## 1. What this project is

ARS coursework (MSc, 2026/27): offline-to-online (O2O) reinforcement learning on
Gymnasium environments. One student, one repo (`github.com/sarahqxg/o2ocw`).

Research framing (fixed by Task 1, do not change):

- **Core paper A — OPT** (Shin et al., ICML 2025): *replace* the offline critic.
  Discard the pretrained Q-function and build a new one from online data before
  fine-tuning starts.
- **Core paper B — WSRL** (Zhou et al., ICLR 2025): *keep* the offline critic.
  Recalibrate it with a short warm-up of online rollouts, then fine-tune without
  retaining offline data.
- **Framing — Three Regimes** (Li et al., 2025): which remedy wins depends on
  dataset quality relative to the pretrained policy. We label each dataset by
  J(pretrained policy) vs J(behaviour policy) and predict the winner a priori.
- **Known confound — data mixing** (ROAD, ARB): OPT and WSRL differ in replay
  composition. Both arms must run with the same replay composition; the mixing
  ratio is an ablation axis, not a hidden difference.
- **Possible extension**: ARB-style on-policyness replay prioritisation.

The experiment is a *controlled comparison* of "new critic" vs "old critic"
fine-tuning for discrete-action DQN agents, across dataset-quality regimes, on
two small environments. Internal comparisons only; leaderboard scores are
irrelevant.

## 2. Decisions already made (confirmed by the owner — ask before changing)

| Topic | Decision |
|---|---|
| Environments | `CartPole-v1` (classic control) and `LunarLander-v3` (Box2D, **discrete**, 4 actions) |
| Code layout | Installable Python package + thin Colab notebooks. Logic lives in the package, never only in a notebook. |
| Baseline algorithms | **d3rlpy** for both online (`DoubleDQN`) and offline (`DiscreteCQL`) and for online fine-tuning from a pretrained model. |
| Novel methods (OPT, WSRL, extensions) | Adapted from the **authors' GitHub repos**, which the owner will share. Do not re-invent them from the papers alone. Cite the repo in the module docstring. |
| Offline datasets | **Self-generated**: train an online DQN per environment, checkpoint at *random / medium / expert* return levels, roll each out to a fixed-size dataset. Generated once, stored on Drive, reused everywhere. |
| Persistence | **Google Drive** for datasets, checkpoints, result CSVs, under `MyDrive/roboticSystem cw/` (notebooks live there too). GitHub for code only; repo is **public**, so PDFs/xlsx stay out (see `../.gitignore`). |
| Logging | Per-run CSV (per-episode/eval rows) + config JSON on Drive. Plots built afterwards with matplotlib in an analysis notebook. No W&B, no TensorBoard. |
| Compute | **Free Colab CPU**. Small MLPs only. |
| Seeds | **5 seeds per condition** minimum. |
| Language | English for code, comments, docstrings, file names. |

Assumptions I made that the owner has not yet confirmed (flag, then follow
until told otherwise):

- Package name: `o2o` (importable as `import o2o`).

## 3. Hard rules for Claude

1. **Ask before deciding.** Any choice not listed in §2 (hyperparameters that
   affect the comparison, dataset sizes, metric definitions, new dependencies,
   folder renames, deleting artifacts) goes to the owner first via a question.
   Do not pick a "sensible default" silently; state it and get a yes.
2. **Everything must run in Google Colab, every time.** Before claiming a
   notebook or module works, run the smoke test (§7) locally if possible and
   keep the Colab bootstrap cell (§5) identical across notebooks. If a change
   needs a system package, add it to the bootstrap cell, not to a README.
3. **No notebook-only logic.** A notebook may: bootstrap, set config, call
   package functions, show plots. It may not define agents, datasets, training
   loops or metrics.
4. **Library first, author code second, own code last.** Use d3rlpy for
   anything it already provides. Use the authors' repo for OPT/WSRL mechanics.
   Write custom code only for the glue and the experiment harness.
5. **Reproducibility.** Every run takes an explicit `seed`; seed `numpy`,
   `torch`, `random`, the env (`env.reset(seed=...)`) and d3rlpy. Every output
   file carries the config that produced it.
6. **Checkpoint and resume.** Colab disconnects. Sweeps must be resumable: skip
   a (condition, seed) whose result file already exists on Drive; save model
   checkpoints at least every N steps.
7. **Academic honesty.** Never fabricate, smooth, or cherry-pick results. Cite
   libraries and author repos in docstrings and in the paper's bib. Keep
   `../reference_table_O2O_RL.bib` as the single source of citation keys.
8. **Git.** Do not commit or push unless asked. Large artifacts (datasets,
   checkpoints, logs) never enter the repo; `.gitignore` them.
9. **Keep it small.** Free CPU budget. Prefer fewer, cleaner experiments over
   many noisy ones. Default network: 2-layer MLP, 256 units.

## 4. Planned layout

```
experiment/
  CLAUDE.md                 ← this file
  pyproject.toml            ← package metadata; deps pinned (see §6)
  o2o/
    __init__.py
    envs.py                 ← make_env(name, seed, render=False); env registry; obs/action shapes
    seeding.py              ← seed_everything(seed)
    paths.py                ← Drive/local path resolution; mkdir helpers
    datasets.py             ← collect_dataset(policy, env, n_transitions) → d3rlpy MDPDataset; save/load (.h5) + metadata.json
    behaviour.py            ← train online DQN and dump checkpoints at target return levels (random/medium/expert)
    agents/
      __init__.py
      baselines.py          ← d3rlpy DoubleDQN (online), DiscreteCQL (offline) factory functions
      opt.py                ← OPT adaptation for DQN (from authors' repo; cite)
      wsrl.py               ← WSRL adaptation for DQN (from authors' repo; cite)
      replay.py             ← shared replay composition control; (later) ARB prioritisation
    configs/
      __init__.py
      cartpole.py           ← env-specific hyperparameters
      lunarlander.py
    experiments/
      __init__.py
      runner.py             ← run_condition(cfg, seed) → writes CSV + JSON; resumable sweep loop
      conditions.py         ← the 3 configurations (online / offline / offline-online) × method × dataset regime
      metrics.py            ← convergence speed (episodes to threshold), average return, Q-vs-MC-return calibration
    analysis/
      __init__.py
      load.py               ← read all result CSVs into one DataFrame
      plots.py              ← learning curves (mean ± CI over seeds), regime tables
  notebooks/
    00_setup_check.ipynb    ← bootstrap + smoke test both envs
    01_collect_datasets.ipynb
    02_task2_prototypes.ipynb   ← the 3 PoCs per env (Demo 1)
    03_task3_experiments.ipynb  ← OPT vs WSRL sweep
    04_analysis.ipynb
  tests/
    test_envs.py            ← both envs build, step, seed deterministically
    test_datasets.py        ← round-trip save/load, shapes
    test_runner.py          ← a 200-step run writes a valid CSV + JSON
```

Create files only as they are needed; do not scaffold empty modules ahead of use.

## 5. Colab contract

Every notebook starts with the same bootstrap cell, kept byte-identical:

```python
# --- Colab bootstrap (identical in every notebook) ---
import os, sys, subprocess
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    subprocess.run(["apt-get", "install", "-y", "-qq", "swig"], check=True)   # needed by box2d
    if not os.path.exists("/content/o2ocw"):
        subprocess.run(["git", "clone", "-q", "https://github.com/sarahqxg/o2ocw.git", "/content/o2ocw"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "/content/o2ocw/experiment"], check=True)
    DRIVE_ROOT = "/content/drive/MyDrive/roboticSystem cw"
else:
    DRIVE_ROOT = os.path.expanduser("~/o2ocw_artifacts")   # local fallback
os.makedirs(DRIVE_ROOT, exist_ok=True)
import o2o
o2o.paths.set_root(DRIVE_ROOT)
```

Rules:

- `o2o.paths` is the only place that knows about `DRIVE_ROOT`. Modules take
  paths from it; nothing hard-codes `/content/drive`.
- Rendering is off by default. `render_mode="rgb_array"` only in
  `00_setup_check` for a single frame; never during training.
- Box2D on Colab needs `swig` before `gymnasium[box2d]`. If the install breaks
  on a Colab image update, fix the bootstrap cell and re-run `00_setup_check`.
- Do not rely on Colab secrets, API keys, or GPU.

Drive layout:

```
MyDrive/roboticSystem cw/
  datasets/<env>/<regime>/data.h5 + metadata.json     # regime ∈ {random, medium, expert}
  behaviour/<env>/<regime>/model.d3                   # checkpoint that generated the dataset
  runs/<env>/<condition>/<method>/<seed>/{log.csv, config.json, model.d3}
  figures/
```

## 6. Dependencies

Pinned in `pyproject.toml`; bump only with the owner's agreement and after
`00_setup_check` passes on Colab. Target set:

- `gymnasium[classic-control,box2d]`
- `d3rlpy` (2.x line)
- `torch` (whatever Colab ships; do not pin a CUDA build)
- `numpy`, `pandas`, `matplotlib`, `h5py`
- `pytest` (dev)

Check the installed d3rlpy API (`DoubleDQNConfig`, `DiscreteCQLConfig`,
`MDPDataset`, `fit`, `fit_online`, `save`/`load_learnable`) against its version
before writing against it; the 1.x and 2.x APIs differ.

## 7. Smoke test (must pass before any "it works")

```python
from o2o.envs import make_env
for name in ["CartPole-v1", "LunarLander-v3"]:
    env = make_env(name, seed=0)
    obs, info = env.reset(seed=0)
    for _ in range(50):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        if term or trunc: obs, info = env.reset()
    env.close()
print("envs ok")
```

Plus `pytest tests/` locally when a local Python is available. Report test
output verbatim; if something fails, say so.

## 8. Experimental design (summary; details in `conditions.py` once agreed)

Three configurations required by the coursework, per environment:

1. **Online**: DoubleDQN from scratch. Baseline curve and source of behaviour checkpoints.
2. **Offline**: DiscreteCQL on each dataset regime. Evaluate the frozen policy.
3. **Offline-online**: fine-tune from the offline model with
   - *OPT arm* (new critic), and
   - *WSRL arm* (old critic + warm-up),
   with matched total online sample budget (count OPT's online pre-training
   samples and WSRL's warm-up rollouts inside the same budget) and matched
   replay composition.

Metrics (coursework-required plus project-specific):

- Convergence speed: episodes (and env steps) to reach a threshold return
  (threshold per env, to be agreed), averaged across seeds and across envs.
- Average return over a fixed evaluation window; report mean and 95% CI over
  5 seeds (bootstrap, per Agarwal et al. 2021).
- Q-value calibration: estimated Q vs Monte-Carlo return during the first
  fine-tuning episodes, to diagnose the critic at the switch.
- Regime label per dataset: J(offline policy) vs J(behaviour policy).

## 9. Open questions (ask the owner when they become relevant)

- Dataset size per regime (transitions) and the return thresholds that define
  random / medium / expert for each env.
- Online sample budget for fine-tuning and warm-up length.
- Convergence threshold per env (e.g. 475 for CartPole, 200 for LunarLander?).
- Which author repos to adapt (URLs pending from the owner).
- Whether ARB goes in as an extension or stays future work.
- Package name `o2o` (assumed, not confirmed).
