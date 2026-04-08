# Hierarchical PBT (Meta Controller)

**Phase 1 of Hierarchical Predictive Boundary Theory (PBT)** — a research project exploring whether the PBT modules (Awareness, Encoder, Valence, Memory, World) can be lifted from a single-agent reactive level to a meta-cognitive layer that supervises a borrowed motor skill.

**Lead researcher**: Ninthanawat N. (Boundary Research Initiative, Bangkok)
**Status**: Phase 1 in progress — see [docs/milestones.md](docs/milestones.md)

---

## What is this?

In the original PBT formulation, all five modules (A, E, V, M, W) operate at a single timescale on raw sensory input. This works for vision and locomotion but raises a question: can the same structural architecture work *recursively*, with a slow PBT loop sitting on top of a fast reactive policy and supervising it via summary descriptions of the world?

Hierarchical PBT tests this by combining:

- **Lower level (fast loop, 50 Hz)**: A walking policy trained with PPO in [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) — purely reactive, no PBT modules. Treated as a "borrowed reflex."
- **Upper level (slow loop, ~2-3 Hz)**: A meta PBT loop using a Vision Language Model (Qwen2.5-VL-3B via Ollama) as the World module, plus Python implementations of E^meta, V^meta, M^meta, A^meta — operating on natural language descriptions of robot state and camera frames.

The two loops communicate through a thread-safe `SharedCommandBuffer`. The slow loop writes velocity commands; the fast loop reads them at every step.

For full theoretical background, see the main PBT paper (separate repository).

---

## Quickstart (Colab)

1. Open [`notebooks/m0_smoke_test.ipynb`](notebooks/m0_smoke_test.ipynb) in Google Colab with **L4 GPU** runtime
2. Run all cells — verifies Genesis + Ollama + Qwen2.5-VL install correctly
3. Open [`notebooks/m1_train_skill.ipynb`](notebooks/m1_train_skill.ipynb) in the same session
4. Run all cells — trains the walking skill (~3 minutes on L4)
5. Ready for M2+ (coming soon)

**Hardware**: L4 GPU (24 GB VRAM) — A100 unnecessary for this project

---

## Repository structure

```
hierarchical-pbt/
├── README.md
├── requirements.txt
├── docs/
│   ├── milestones.md           ← M0-M4 plan + status
│   ├── architecture.md         ← Two-loop diagram + theory mapping
│   └── m1_results.md           ← M1 PASS evidence
├── notebooks/
│   ├── m0_smoke_test.ipynb     ← M0 — environment validation
│   └── m1_train_skill.ipynb    ← M1 — train walking PPO policy
└── pbt_meta/                   ← Python package (filled in M2+)
    ├── config.py
    ├── sim/                    ← Genesis bridge
    ├── skill/                  ← Borrowed reflex (PPO policy)
    │   └── checkpoints/        ← Trained model weights
    ├── meta/                   ← W^meta, E^meta, V^meta, M^meta, A^meta
    ├── loop/                   ← Fast (50 Hz) + Slow (~2 Hz) loops
    └── instrumentation/        ← Logging, metrics
```

---

## Milestone status

- [x] **M0** — Smoke test (Genesis + Ollama + Qwen install)
- [x] **M1** — Train walking skill (Genesis go2, 250 iter PPO)
- [ ] **M2** — Fast loop in separate thread + manual command sliders
- [ ] **M3** — VLM dispatch (Qwen reads camera + user text → command)
- [ ] **M4** — PBT meta loop (A-E-V-M-R on top of M3)

---

## License

TBD (research code, not yet released)
