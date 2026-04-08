# Skill Checkpoints

This directory holds trained PPO walking policies. Each subdirectory is one
training run, named `{milestone}_{config_version}_run{n}`.

## Layout

```
checkpoints/
└── m1_v2_run1/                        ← M1 with v2 config (parity reward), first run
    ├── model_0.pt                     ← initial weights (sanity)
    ├── model_100.pt                   ← intermediate
    ├── model_200.pt                   ← near-converged
    ├── model_249.pt                   ← FINAL — load this
    ├── events.out.tfevents.<...>      ← tensorboard log
    ├── git/                           ← rsl-rl git snapshot
    └── training_configs_and_results.json
```

## Loading a checkpoint

```python
import torch
from rsl_rl.runners import OnPolicyRunner

# (env must be built first — see notebooks/m1_train_skill.ipynb Step 6)

train_cfg = go2_train.get_train_cfg("go2_walking_meta_phase1")
runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
runner.load("pbt_meta/skill/checkpoints/m1_v2_run1/model_249.pt")
policy = runner.get_inference_policy(device="cuda:0")

# CRITICAL: env tensors are inference tensors after load — wrap operations
with torch.inference_mode():
    obs_td = env.reset()
    actions = policy(obs_td)
    obs_td = env.step(actions)
```

## Adding a new run

Convention: bump `run` number for hyperparameter tweaks of the same config
version, bump `config_version` for structural changes.

Examples:
- `m1_v2_run2` — same patches, different random seed
- `m1_v3_run1` — new reward shape (e.g., add foot clearance term)
- `m2_v1_run1` — first M2 retraining (if needed)

## Backup

All runs are mirrored to Google Drive at:
`PBT_Robotics_Hierarchical/`

Drive is the source of truth for the tensorboard event files and git snapshot.
The `.pt` files are tracked in this repo because they fit comfortably under
GitHub's 100 MB limit (each is ~4.5 MB).
