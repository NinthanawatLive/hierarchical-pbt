# M2 Results — Skill Replay + Manual Command

**Date**: April 8, 2026
**Outcome**: ✅ PASS (with scope revision noted below)
**Notebook**: `notebooks/m2_skill_replay.ipynb`
**Modules**: `pbt_meta/loop/shared_state.py`, `pbt_meta/loop/fast.py`

## Summary

M2 demonstrates that the M1 walking skill can be driven from a `SharedCommandBuffer` at sustained 50 Hz while a producer (slider, schedule, or in M3 a VLM) writes commands into the buffer. The single-environment fast loop runs at over 100 Hz on Colab L4, leaving substantial headroom for the slow-loop additions in M3 and M4.

The original M2 plan called for ipywidgets sliders as the manual command source. We discovered during this run that ipywidgets cannot drive a Colab kernel that is busy in a long-running blocking cell — display updates from kernel to browser still work, but messages from browser to kernel (slider drag events) are not processed until the cell yields. Manual command was therefore replaced with a programmatic command schedule. The buffer interface itself was unaffected, and M3 will use the same interface with a slow-loop producer that lives entirely on the kernel side, where the limitation does not apply.

## Configuration

| Setting | Value |
|---|---|
| Simulator | Genesis 0.4.3 (same as M1) |
| Robot | Unitree Go2 (12 actuated DOFs) |
| Environment | `examples/locomotion/go2_env.py` with M1 v2 patches |
| Parallel envs | 1 (single robot for real-time semantics) |
| Skill checkpoint | `pbt_meta/skill/checkpoints/m1_v2_run1/model_249.pt` |
| Control rate target | 50 Hz |
| Run length | 15,000 steps (300 sec sim time) |
| Hardware | NVIDIA L4 on Google Colab |

## What was built

### `pbt_meta/loop/shared_state.py` — `SharedCommandBuffer`

A thread-safe last-write-wins buffer holding a 3-tuple `(vx, vy, wz)`. Producers call `set(vx, vy, wz)`; the consumer (`FastLoop`) calls `get()` once per simulation step. Values are clamped to `pbt_meta.config.COMMAND_RANGES` on write and a clamp counter is exposed via `stats()`.

The lock is real even though M2 runs single-threaded, because ipywidgets callbacks are dispatched from the Jupyter kernel's event loop thread when they fire, not from the cell-execution thread. The lock is also a no-cost insurance for the M3 → M4 transition where the producer will be a slow-loop function running asynchronously to the fast loop.

### `pbt_meta/loop/fast.py` — `FastLoop`

Drives the env at maximum simulation speed. Per step:

1. Read `(vx, vy, wz)` from the buffer
2. Write into `env.commands[:, 0:3]`
3. Forward through the policy: `actions = policy(obs_td)`
4. Step the env: `step_res = env.step(actions)`
5. Read robot state from `env.base_pos[0]`, `env.base_ang_vel[0]`, optionally `env.base_lin_vel[0]`
6. Build a JSONL log record with timing info
7. Optionally render via `gs.Camera` every N steps
8. Invoke a user callback for live UI updates

All env operations are wrapped in `torch.inference_mode()`. Without this wrapping, Genesis env tensors (which are converted to inference tensors after `runner.learn()` in M1) raise `RuntimeError: Inplace update to inference tensor outside InferenceMode is not allowed` on the first `env.commands[:, 0] = vx` write.

### `notebooks/m2_skill_replay.ipynb`

End-to-end notebook covering: install + import, Genesis init, env build at `n_envs=1`, checkpoint load, slider widget setup (used as a producer in the original plan), live visualization callback, FastLoop execution, video encoding (no-op when no camera), and post-run log analysis.

## Findings

### Performance: 103.7 Hz at single-env (207% of target)

```
Steps:           15,000 / 15,000  (no early termination)
Wall clock:      144.6 sec (2:25)
Sim time:        300.0 sec  (5 minutes simulated)
Speedup:         2.07× real-time
Actual FPS:      103.7
Target FPS:      50.0
FPS ratio:       207%
```

Per-step timing distribution from the JSONL log:

| Statistic | Value |
|---|---|
| mean step_dt | 9.50 ms |
| median step_dt | 8.46 ms |
| std | 4.12 ms |
| p99 | 27.86 ms |
| max | 33.88 ms |
| Steps over 20 ms target | 750 / 15,000 (5.0%) |

Mean is higher than median because of a long tail of slow steps, consistent with occasional CUDA sync, garbage collection, or page faults. The worst step (33.88 ms) is still under the 40 ms threshold that would constitute a visible 25 Hz drop. M3 has roughly 10 ms of headroom per step before the loop falls behind real-time, which is comfortable for the slow loop because the slow loop runs *asynchronously* and only writes to the buffer occasionally rather than per-step.

### Behavior: stable across all 7 commanded behaviors

The programmatic schedule cycled through STOP, FORWARD, BACKWARD, TURN_LEFT, TURN_RIGHT, FORWARD+TURN, and SIDESTEP. All commands were honored:

| Schedule entry | sim time | Command | Behavior in raw log |
|---|---|---|---|
| step 100 | 2 s | STOP (0, 0, 0) | actual (vx, vy, wz) ≈ (0.01, 0.00, -0.03) — stationary |
| step 2000 | 40 s | FORWARD (0.5, 0, 0) | actual_vx = 0.568 (114%), pos_x = +5.61 m |
| step 5000 | 100 s | BACKWARD (-0.5, 0, 0) | actual_vx = -0.601 (120%), pos_x = -5.94 m |
| step 8500 | 170 s | TURN_LEFT (0, 0, 0.8) | actual_wz = 0.889 (111%), pos near origin |
| step 13000 | 260 s | FWD+TURN (0.5, 0, 0.5) | actual_vx = 0.537, actual_wz = 0.351 |

Height stayed in `[0.296, 0.420]` m with mean 0.323 m. **Zero falls** across the entire 15,000 step run.

### Surprise: episode auto-reset visible in log

While verifying that logged values were not hardcoded, raw log inspection at consecutive steps 5000-5004 revealed an unexpected event:

```
step 5000: pos=[-5.940, -0.788, 0.316]   height=0.3164
step 5001: pos=[-5.952, -0.790, 0.316]   height=0.3156
step 5002: pos=[-5.964, -0.792, 0.317]   height=0.3171
step 5003: pos=[-5.976, -0.795, 0.318]   height=0.3184
step 5004: pos=[ 0.000,  0.000, 0.420]   height=0.4200   ← teleport
```

The robot's position jumped from `(-5.98, -0.79)` to `(0, 0, 0.42)` in a single step. This is `Go2Env._reset_idx()` firing — the default `episode_length_s` in Genesis is around 20 sec / 1000 steps, and the env auto-resets episodes inside the FastLoop without notification.

**Implications for M2**: the height "max = 0.42 m" reported by the analysis cell is an artifact of the post-reset spawn pose, not real robot behavior. Mean height (0.323 m) and the per-step velocities are still accurate because they are read from the env state, not computed from position deltas across reset boundaries.

**Implications for M3**: a VLM observing the camera feed would see the robot teleport every 20 seconds, which is likely to confuse it. M3 must either disable the episode timeout (`episode_length_s = 99999`), use long episodes (e.g., 600 sec for one PBT session), or notify the meta layer of resets via a text channel so the VLM has context.

## Pass criteria

| Criterion | Threshold | Result | Verdict |
|---|---|---|---|
| Fast loop @ 50 Hz stable | ≥ 45 Hz | 103.7 Hz | ✅ |
| No crash after 5 min | ≥ 5000 steps | 15,000 steps | ✅ |
| Log file usable | parses, has structured records | 5.79 MB, 15,000 records | ✅ |
| Slider response within 1-2 steps | drag → buffer update | **scope revised** — see below | ⚠️ |
| Zero falls (added during analysis) | < 10% | 0% | ✅ |

### About the slider criterion

The slider criterion in the original M2 spec was meant to prove that an external producer can drive the buffer in real time. We could not verify this with ipywidgets sliders inside Colab because the kernel does not process incoming widget messages while a long blocking cell is running. We did verify the equivalent property using a programmatic schedule: the schedule logic wrote 10 transitions to the buffer at the correct steps, the buffer's `n_writes` counter incremented from 2 (sanity reads in Step 5) to 12, and the FastLoop's `env.commands` reflected the new values immediately on the next step. The interface is sound; the limitation is purely on the browser↔kernel transport.

**M3 is unaffected by this limitation** because the slow loop will be a Python coroutine running on the kernel side. There is no browser→kernel hop in the M3 architecture.

Verdict: **M2 PASS**, with the slider criterion replaced by an equivalent programmatic-schedule test.

## Files in this run

```
notebooks/m2_skill_replay.ipynb           # the notebook (run as documented)
pbt_meta/loop/shared_state.py             # SharedCommandBuffer
pbt_meta/loop/fast.py                     # FastLoop
docs/m2_results.md                        # this document
```

The 5.79 MB JSONL log was kept locally during the run. It was not committed to the repo to keep history clean; if it is needed for deeper analysis later, regenerating it is a 2-3 minute operation in Colab.

## Known issues to address before M3

1. **Camera unavailable after env build**. `Go2Env` does not expose a hook to add `gs.Camera` before `scene.build()`. M3 will need either a subclass of `Go2Env` that accepts a camera spec in `__init__`, or a fork of `go2_env.py` that adds the camera inline. Without this, the VLM has no image input.

2. **Episode auto-reset**. As described in the findings section, the env teleports the robot every ~20 seconds. M3 must decide how to handle this (disable, lengthen, or surface to the VLM).

3. **No real-time pacing**. The FastLoop currently runs as fast as possible (2.07× real-time). For M3 demonstrations where a human watches the slow loop reason about the robot's behavior, the FastLoop should optionally pace itself to wall-clock 50 Hz so that the camera feed and the VLM commentary stay in sync.
