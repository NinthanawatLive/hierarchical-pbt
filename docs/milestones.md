# Milestones — Hierarchical PBT Phase 1

**Goal of Phase 1**: Demonstrate Stage 1-2 of PBT (detect + sustain) with `R_sensory` only, using a meta layer running on top of a borrowed walking skill.

**Total estimated time**: 8-14 days
**Status**: M0 ✓, M1 ✓, M2-M4 pending

---

## M0 — Smoke Test ✅ (Apr 7, 2026)

**Goal**: Verify all components install and connect on Colab L4

**Deliverable**: `notebooks/m0_smoke_test.ipynb`

**Tests**:
1. L4 GPU available with ≥20 GB
2. Genesis 0.4.3 installs cleanly
3. Genesis renders Go2 headlessly via `gs.Camera`
4. Ollama installs + runs as background service
5. Qwen2.5-VL-3B pulls + responds to image+text input

**Result**: All 5 tests PASS
- Genesis init: 0.9s, build: 134s (first), render: 35s (first frame, JIT)
- Ollama: install OK with `zstd + pciutils + lshw` apt deps + `!` shell magic
- Qwen2.5-VL-3B: cold latency 67s (model load), warm latency 0.3s
- Frame: 640×480×3 PNG ~60 KB, no resize needed

---

## M1 — Train Walking Skill ✅ (Apr 7, 2026)

**Goal**: Train Go2 walking PPO checkpoint with **wide command range** so the meta layer can drive it

**Deliverable**: `notebooks/m1_train_skill.ipynb` + `pbt_meta/skill/checkpoints/m1_v2_run1/model_249.pt`

**Configuration**:
- Genesis `examples/locomotion/go2_env.py` (downloaded fresh from main branch)
- 4096 parallel environments
- 250 PPO iterations
- Two patches to default config:
  1. **Wide command range**: `vx ∈ [-1, 1]`, `vy ∈ [-0.5, 0.5]`, `wz ∈ [-1, 1]`
  2. **Reward reweight**: `tracking_ang_vel: 0.2 → 1.0` (parity with `tracking_lin_vel`)

**Critical lessons learned**:
1. `rsl-rl-lib` MUST be `5.0.0`, not 2.2.4 — Genesis main branch is TensorDict-native
2. Default `tracking_ang_vel: 0.2` is too low for wide command range; policy ignores turning entirely until reweighted to parity
3. Genesis env tensors become inference tensors after `runner.learn()` — ALL subsequent `env.reset()`, `env.step()`, and `env.commands[:]= ...` operations must be wrapped in `torch.inference_mode()`
4. `env.reset()` returns a `TensorDict` with key `'policy'` of shape `(num_envs, 45)`, not a raw tensor
5. `policy(tensordict)` works directly; `policy(env.obs_buf)` raises IndexError

**Training performance**: 250 iter × 4096 envs = 24.5M steps in **2:47** at 170k steps/sec on L4. Far faster than Phase 3 PBT-augmented runs (which use 64 envs and ~2050 steps/sec) — confirms PBT module overhead dominates compute cost in our other robotics work.

**Reward curve**:
| iter | tracking_lin_vel | tracking_ang_vel | mean_reward |
|---:|---:|---:|---:|
| 0 | 0.004 | 0.005 | 0.003 |
| 61 | 0.275 | **0.530** | 10.24 |
| 123 | 0.456 | 0.702 | 16.45 |
| 185 | 0.817 | 0.806 | 24.41 |
| 246 | 0.808 | 0.787 | 26.86 |

**Surprising finding**: Angular velocity tracking learns *faster* than linear velocity tracking in early iterations (0.530 vs 0.275 at iter 61). Hypothesis: the "no-turn" baseline pose is closer to "default standing" than the "no-walk" baseline is to gait coordination, so turning is a smaller perturbation from the initial policy.

**Eval results** (7 commands, 100 steps each, 4096 envs, fall = `base_z < 0.2`):

| Command | Target | Actual (vx, vy, ω_z) | Tracking | Fall % |
|---|---|---|---|---|
| STOP | (0, 0, 0) | (0.00, -0.00, -0.01) | perfect | 0% |
| FORWARD | vx=0.5 | +0.51 | **102%** | 0% |
| BACKWARD | vx=-0.3 | -0.26 | 87% | 0% |
| TURN_LEFT | ω=+0.5 | +0.51 | **102%** | 0% |
| TURN_RIGHT | ω=-0.5 | -0.50 | 100% | 0% |
| FORWARD+TURN | (0.3, 0, 0.5) | (+0.24, +0.16, +0.51) | 80% / 102% | 0% |
| SIDESTEP | vy=+0.3 | +0.22 | 73% | 0% |

**Verdict**: M1 PASS. Ready for M2.

**Checkpoint location**:
- In repo: `pbt_meta/skill/checkpoints/m1_v2_run1/model_249.pt` (4.5 MB)
- Backup: Google Drive `PBT_Robotics_Hierarchical/m1_v2_run1/`

---

## M2 — Skill Replay + Manual Command (planned, 1-2 days)

**Goal**: Fast loop runs in separate thread, controlled by shared buffer

**Deliverable**: `notebooks/m2_skill_replay.ipynb`, modules `pbt_meta/loop/{fast,shared_state}.py`

**Steps**:
1. `SharedCommandBuffer` (thread-safe, last-write-wins)
2. `FastLoop` reading buffer every step (in `torch.inference_mode()`)
3. Notebook UI: 3 sliders (vx, vy, wz) + start/stop
4. Log every step → JSONL

**Success criteria**:
- Fast loop @ 50 Hz stable (measured FPS)
- Slider response within 1-2 steps
- Log file usable in analysis notebook
- No crash after 5 min

**Open questions**:
- Does Genesis maintain 50 Hz at small batch (1 or 16 envs)?
- How does TensorDict policy interface behave inside a thread?

---

## M3 — VLM Dispatch (planned, 2-3 days)

**Goal**: VLM reads camera + user text → outputs command → fast loop uses

**Deliverable**: `notebooks/m3_vlm_dispatch.ipynb`, modules `pbt_meta/meta/{world,state_summary}.py` + `pbt_meta/loop/slow.py`

**Success criteria**:
- "เดินไปข้างหน้า" → vx ≈ 0.3-0.7
- "หยุด" → (0, 0, 0)
- "เลี้ยวขวา" → wz < 0
- VLM call rate ≥ 1 Hz
- JSON parse failure < 10%

---

## M4 — PBT Meta Loop (planned, design TBD, 3-7 days)

**Goal**: A-E-V-M-R loop on top of VLM dispatch

**Status**: ⚠ DESIGN DEFERRED — refine after M1-M3 working

**Tentative components**:
1. `E^meta`: compare VLM prediction vs reality after 1 cycle
2. `V^meta`: gap → 3-axis valence
3. `M^meta`: V_acc + R_sensory per paper Section 2.1
4. `A^meta`: Triadic Gating, pulse_A binary
5. Instrumentation: log JSONL, plot R trace + pulse density + C_ext/C_int ratio

**Phase 1 final deliverable**:
- **Visible**: Robot responds to changing situations
- **Measurable**:
  - R_sensory dynamics per equation
  - Pulse density correlates with R
  - C_ext/C_int ratio shifts dynamically (test paper Section 3.3.3)
