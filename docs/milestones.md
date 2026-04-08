# Milestones — Hierarchical PBT Phase 1

**Goal of Phase 1**: Demonstrate Stage 1-2 of PBT (detect + sustain) with `R_sensory` only, using a meta layer running on top of a borrowed walking skill.

**Total estimated time**: 8-14 days
**Status**: M0 ✓, M1 ✓, M2 ✓, M3-M4 pending

---

## M0 — Smoke Test ✅ (Apr 7, 2026)

**Goal**: Verify all components install and connect on Colab L4

**Deliverable**: `notebooks/m0_smoke_test.ipynb`

All 5 tests PASS. Genesis renders Go2 headlessly via `gs.Camera`. Ollama installs cleanly with the right apt deps. Qwen2.5-VL-3B warm latency 0.3s — 16× faster than the 5s budget.

---

## M1 — Train Walking Skill ✅ (Apr 7, 2026)

**Goal**: Train Go2 walking PPO checkpoint with **wide command range** so the meta layer can drive it

**Deliverable**: `notebooks/m1_train_skill.ipynb` + `pbt_meta/skill/checkpoints/m1_v2_run1/model_249.pt`

**Configuration**:
- Genesis `examples/locomotion/go2_env.py` with two patches:
  1. Wide command range: `vx ∈ [-1, 1]`, `vy ∈ [-0.5, 0.5]`, `wz ∈ [-1, 1]`
  2. Reward reweight: `tracking_ang_vel: 0.2 → 1.0`
- 4096 envs, 250 PPO iter
- `rsl-rl-lib==5.0.0` (TensorDict-native, required)

**Performance**: 24.5M steps in 2:47 at 170k steps/sec on L4.

**Eval**: All 7 commands pass criteria, 0% fall. See `docs/m1_results.md`.

---

## M2 — Skill Replay + Manual Command ✅ (Apr 8, 2026)

**Goal**: Run the M1 walking skill at 50 Hz controlled by a `SharedCommandBuffer`, with the buffer interface in place for M3.

**Deliverable**: `notebooks/m2_skill_replay.ipynb`, `pbt_meta/loop/{shared_state,fast}.py`

**Performance**: 15,000 steps in 144.6 sec wall clock = **103.7 Hz** (207% of 50 Hz target). 0 falls. step_dt mean 9.5 ms, p99 27.86 ms.

**Critical lessons learned**:
1. **ipywidgets sliders cannot drive a blocking Colab cell**. Display kernel→browser still works; input browser→kernel does not. Replaced with programmatic command schedule. M3 unaffected because slow loop lives kernel-side.
2. **`add_camera()` after `scene.build()` not supported in `Go2Env`**. M2 ran without rendering. M3 must subclass to inject camera before build.
3. **Genesis env auto-resets episodes inside the FastLoop** (~20 sec / 1000 steps default). Visible as position teleport in the raw log. M3 must handle this.

**Verdict**: M2 PASS, slider criterion replaced by equivalent programmatic-schedule test. See `docs/m2_results.md`.

---

## M3 — VLM Dispatch (planned, 2-3 days)

**Goal**: VLM reads camera + user text → outputs command → fast loop uses

**Deliverable**: `notebooks/m3_vlm_dispatch.ipynb`, modules `pbt_meta/meta/{world,state_summary}.py` + `pbt_meta/loop/slow.py`

**Steps**:
1. **Subclass / fork `Go2Env`** to inject `gs.Camera` before `scene.build()` — fixes M2 camera issue
2. **Disable / lengthen episode timeout** to avoid teleport in camera feed
3. `World` class wrapping Ollama (image + prompt → JSON command)
4. `state_summary.summarize(robot_state) → str`
5. `SlowLoop` extracting frame, composing prompt, parsing JSON, writing buffer
6. Prompt template enforcing JSON output
7. Notebook UI: text input for user instructions

**Success criteria**:
- "เดินไปข้างหน้า" → `vx ≈ 0.3-0.7`
- "หยุด" → `(0, 0, 0)`
- "เลี้ยวขวา" → `wz < 0`
- VLM call rate ≥ 1 Hz
- JSON parse failure < 10%

**Open design decisions**:
- Slow loop = coroutine in same thread, true thread, or asyncio task?
- Real-time pacing for the fast loop (so demo wall clock matches sim time)?
- How to surface episode resets to the meta layer?

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
