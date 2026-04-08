# Architecture — Hierarchical PBT

## Two-loop design

```
═══════════════════════════════════════════════════════════════════════════════
                  SLOW LOOP (~2-3 Hz, async)
                  "PBT meta layer"
═══════════════════════════════════════════════════════════════════════════════

  ┌─────────────────────────────────────────────────────────┐
  │                                                         │
  │  Camera frame ──────► W^meta (Qwen2.5-VL-3B)            │
  │  User text   ──────►   ↓                                │
  │                      JSON command                       │
  │                        │                                │
  │                        ▼                                │
  │  E^meta ──── compare VLM prediction vs. reality         │
  │      │                                                  │
  │      ▼                                                  │
  │  V^meta ──── 3-axis valence (pain/pleasure/epistemic)   │
  │      │                                                  │
  │      ▼                                                  │
  │  M^meta ──── V_acc + R_sensory (paper Section 2.1)     │
  │      │                                                  │
  │      ▼                                                  │
  │  A^meta ──── Triadic Gating, pulse_A binary             │
  │      │                                                  │
  │      ▼                                                  │
  └──────┼──────────────────────────────────────────────────┘
         │
         ▼
  ┌──────────────────────────┐
  │  SharedCommandBuffer     │
  │  (vx, vy, wz)            │
  │  thread-safe             │
  └────────────┬─────────────┘
               │
═══════════════│═══════════════════════════════════════════════════════════════
               │  FAST LOOP (50 Hz, sync with Genesis)
               │  "borrowed reflex (skill)"
═══════════════│═══════════════════════════════════════════════════════════════
               │
               ▼
  ┌─ proprio observation (45-dim) ──┐
  │  base_ang_vel        × 3         │
  │  projected_gravity   × 3         │
  │  commands            × 3 ◄──────┤
  │  dof_pos − default   × 12        │
  │  dof_vel             × 12        │
  │  last_action         × 12        │
  └──────────────────┬───────────────┘
                     │
                     ▼
    Go2 walking policy (PPO, trained 1× in Genesis)
                     │
                     ▼
            action (12-dim joint targets, scale 0.25)
                     │
                     ▼
           env.step(action)
```

## Theory mapping

The hierarchical structure tests a key claim of the PBT paper: that the same five-module pattern (W, E, V, M, A) can operate at multiple cognitive timescales without architectural redesign.

| Module | Lower level (skill) | Upper level (meta) |
|---|---|---|
| **W** (World) | Implicit (PPO actor weights) | Qwen2.5-VL-3B via Ollama |
| **E** (Encoder) | None — direct proprioception | Compare VLM prediction vs. reality after 1 cycle |
| **V** (Valence) | None — reward only at training | 3-axis (pain, pleasure, epistemic) over gap |
| **M** (Memory) | LSTM hidden state inside actor | V_acc + R_sensory per paper |
| **A** (Awareness) | Always-on (no gating) | Triadic Gating, pulse_A binary |

The lower level is a **borrowed reflex** — it has no PBT modules. We treat it as a fixed motor primitive, like the mammalian brainstem. The upper level is where PBT lives.

## Why this is interesting (and risky)

The PBT paper has proven the modules work at the sensory level (Vision v5.9, Robotics Step 0/v18b, language with tau_ep). What we don't know is:

1. **Does the same structure work when its inputs are abstract?** The meta layer's "world" is JSON-serialized natural language, not raw pixels.
2. **Does R_sensory scale up?** At the sensory level, R lives in embedding space. At the meta level, it has to live in something more like "concept space."
3. **Does the C_ext/C_int dynamic shift work at the meta level?** Paper Section 3.3.3 predicts the ratio should change with task difficulty. Phase 1 will measure this.
4. **Does the 3-axis non-collapse problem move with the level, or does it disappear?** Open question — see PBT main paper.

If the answer to all four is "yes", then the architecture has a strong claim to generality. If any one is "no", we learn something specific about what the lower level was secretly providing for free.

## Hardware constraints

- Colab L4 GPU (24 GB VRAM)
- Genesis 4096 envs uses ~1 GB → plenty of headroom
- Qwen2.5-VL-3B uses ~3 GB when loaded
- PPO actor + critic + replay = ~1 GB
- Free for browser display + other processes: ~18 GB

## File mapping

| Layer | Files |
|---|---|
| Skill (lower) | `pbt_meta/skill/policy.py`, `checkpoints/` |
| Sim bridge | `pbt_meta/sim/go2_env.py`, `camera.py` |
| Meta (upper) | `pbt_meta/meta/{world,encoder,valence,memory,awareness,state_summary}.py` |
| Loops | `pbt_meta/loop/{fast,slow,shared_state}.py` |
| Logging | `pbt_meta/instrumentation/{logger,metrics}.py` |
| Notebooks | `notebooks/m{0,1,2,3,4}_*.ipynb` |
