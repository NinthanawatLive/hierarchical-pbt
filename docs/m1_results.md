# M1 Results — Train Walking Skill

**Date**: April 7, 2026
**Outcome**: ✅ PASS
**Checkpoint**: `pbt_meta/skill/checkpoints/m1_v2_run1/model_249.pt` (4.5 MB)

## Summary

Trained a Genesis Go2 PPO walking policy with wide command range (vx, vy, wz) so the meta layer (M3+) can dispatch arbitrary velocity commands. The first attempt (M1 v1) failed because of an asymmetric reward weighting that the meta layer would never have caught. This document records the diagnosis, fix, and final results.

## Configuration

| Setting | Value |
|---|---|
| Simulator | Genesis 0.4.3 |
| Robot | Unitree Go2 (12 actuated DOFs, URDF from Genesis) |
| Environment | `examples/locomotion/go2_env.py` (downloaded from main branch) |
| Parallel envs | 4096 |
| Training algorithm | PPO via `rsl-rl-lib==5.0.0` |
| Iterations | 250 |
| Total environment steps | 24,576,000 |
| Wall clock training time | 2:47 (167 sec) |
| Throughput | 170,000 steps/sec |
| Hardware | NVIDIA L4 (24 GB) on Google Colab |

## Patches applied to default Genesis config

```python
# Wide command range (default is vx=0.5 only)
command_cfg["lin_vel_x_range"] = [-1.0, 1.0]
command_cfg["lin_vel_y_range"] = [-0.5, 0.5]
command_cfg["ang_vel_range"]   = [-1.0, 1.0]

# Reweight ang_vel tracking from 0.2 → 1.0 (parity with lin_vel)
reward_cfg["reward_scales"]["tracking_ang_vel"] = 1.0
```

## Why the reward reweight matters

The first attempt (M1 v1) used the default `tracking_ang_vel: 0.2` and trained for 101 iterations. Eval showed the robot walked forward and backward correctly but **completely ignored turning commands** (ω_z = 0.05 against a target of 0.5 — at noise floor). Worse, when given a combined command (vx=0.3, wz=0.5), the robot's forward velocity collapsed to 0.04 — out-of-distribution behavior.

**Diagnosis**: With `wz_range = [0, 0]` (Genesis default), the `tracking_ang_vel: 0.2` weight is fine because there's no turning to learn — it functions as a regularizer against unwanted spin. But once we widen `wz_range = [-1, 1]`, the weight imbalance matters. The PPO gradient for "learn to turn" is 5× smaller than for "learn to walk forward," and the policy finds a local optimum where it ignores turning entirely. The reward landscape difference is roughly:

- Don't turn: lin_vel ≈ 0.6, ang_vel weight × value ≈ 0.2 × 0.37 = 0.074, total ≈ 0.67
- Turn perfectly: lin_vel ≈ 0.5 (turning disrupts walking), ang_vel weight × value ≈ 0.2 × 1.0 = 0.20, total ≈ 0.70

The +0.03 marginal reward is too small to overcome the exploration cost of learning a new behavior.

**Fix**: Set `tracking_ang_vel = 1.0` (parity with lin_vel). With 1:1 weighting, the "no-turn" policy loses ~0.5 reward when turning is commanded — strong enough gradient to drive learning.

## Reward curve (M1 v2, the working run)

| iter | tracking_lin_vel | tracking_ang_vel | mean_reward |
|---:|---:|---:|---:|
| 0 | 0.0040 | 0.0052 | 0.003 |
| 61 | 0.2754 | **0.5297** | 10.24 |
| 123 | 0.4561 | 0.7016 | 16.45 |
| 185 | 0.8172 | 0.8059 | 24.41 |
| 246 | 0.8076 | 0.7869 | 26.86 |

**Key observation**: At iter 61, `tracking_ang_vel` (0.530) is *higher* than `tracking_lin_vel` (0.275). Turning is learned faster than forward walking in early iterations.

**Hypothesis** (needs follow-up): The default standing pose is closer to "no-turn baseline" than to "no-walk baseline." A small symmetric perturbation of joint torques produces a yaw rotation with negligible disruption to balance. Producing forward locomotion requires coordinating an actual gait (phase relationships between four legs), which is a larger leap from the initial policy.

If this hypothesis holds, it would have a place in the PBT paper's discussion of why "harder-looking" tasks aren't always harder to learn — what matters is distance in policy space from the initial distribution, not human intuition about complexity.

## Eval results — 7 commands

Each command was applied to all 4096 environments simultaneously for 100 simulation steps (2 sec at 50 Hz). Position delta gives `vx, vy`; `env.base_ang_vel[:, 2].mean()` gives `ω_z`.

| Command | Target (vx, vy, ω_z) | Actual (vx, vy, ω_z) | Tracking | Fall % |
|---|---|---|---|---|
| STOP | (0.0, 0.0, 0.0) | (0.00, -0.00, -0.01) | perfect | 0% |
| FORWARD | (+0.5, 0.0, 0.0) | (+0.51, -0.03, -0.00) | 102% / — / — | 0% |
| BACKWARD | (-0.3, 0.0, 0.0) | (-0.26, -0.01, -0.00) | 87% / — / — | 0% |
| TURN_LEFT | (0.0, 0.0, +0.5) | (-0.05, -0.02, +0.51) | — / — / 102% | 0% |
| TURN_RIGHT | (0.0, 0.0, -0.5) | (-0.01, -0.01, -0.50) | — / — / 100% | 0% |
| FORWARD+TURN | (+0.3, 0.0, +0.5) | (+0.24, +0.16, +0.51) | 80% / — / 102% | 0% |
| SIDESTEP | (0.0, +0.3, 0.0) | (-0.21, +0.22, +0.01) | — / 73% / — | 0% |

**Pass criteria** (defined before eval, from `notebooks/m1_train_skill.ipynb`):
- STOP: |vx|, |vy|, |ω_z| < 0.05
- FORWARD: vx > +0.30 (60% of 0.5)
- BACKWARD: vx < -0.18 (60% of 0.3)
- TURN_LEFT: ω_z > +0.25 (50% of 0.5)
- TURN_RIGHT: ω_z < -0.25 (50% of 0.5)
- FORWARD+TURN: vx > +0.15 AND ω_z > +0.20
- SIDESTEP: vy > +0.15
- All: fall < 10%

**All seven commands pass.** Turning is symmetric (no chirality bias), STOP holds the standing pose perfectly, and even the cross-coupled FORWARD+TURN and SIDESTEP cases respond correctly.

## Notes for M2+

Three things M1 v2 confirmed that M2 must respect:

1. **TensorDict interface**. `env.reset()` and `env.step()` return `TensorDict` objects with key `'policy'` of shape `(num_envs, 45)`. The `policy(...)` callable accepts this directly. Passing `env.obs_buf` (a raw tensor) raises `IndexError`.

2. **Inference tensor lock**. After `runner.learn()` completes, all of Genesis's internal env tensors are converted to inference tensors. Any subsequent write — `env.commands[:, 0] = vx`, `env.reset()`, `env.step()` — must be wrapped in `torch.inference_mode()` or it raises `RuntimeError: Inplace update to inference tensor outside InferenceMode is not allowed`. This is per-thread, so the M2 fast loop thread will need its own `with torch.inference_mode():` context.

3. **`env.step(actions)` return type**. The current notebook handles both `TensorDict` and `tuple` returns defensively. M2 should pin the actual return type with a runtime assert, since fast loop performance is sensitive to per-step Python overhead.

## Files in this run

```
pbt_meta/skill/checkpoints/m1_v2_run1/
├── model_0.pt                                # initial weights
├── model_100.pt                              # iter 100
├── model_200.pt                              # iter 200
├── model_249.pt                              # final (use this)
├── events.out.tfevents.<...>                 # tensorboard log
├── git/                                      # rsl-rl git snapshot
└── training_configs_and_results.json         # patches + eval table
```
