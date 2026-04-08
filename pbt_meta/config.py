"""Global configuration constants for Hierarchical PBT.

These values are pinned to match what M1 trained against. Changing them after
training will silently break the loaded skill checkpoint.
"""

# ============================================================
# Skill (lower level) — Genesis Go2 walking
# ============================================================

# Observation dimensions (matches Genesis examples/locomotion/go2_env.py)
OBS_DIM = 45  # 3 ang_vel + 3 gravity + 3 commands + 12 dof_pos + 12 dof_vel + 12 last_action
ACTION_DIM = 12  # 12 joint position targets

# Control rate
CONTROL_DT = 0.02  # seconds — 50 Hz
CONTROL_HZ = 50

# Action scaling (from Genesis default)
ACTION_SCALE = 0.25

# Command space — matches m1_v2_run1 patches
COMMAND_RANGES = {
    "lin_vel_x": (-1.0, 1.0),
    "lin_vel_y": (-0.5, 0.5),
    "ang_vel_z": (-1.0, 1.0),
}

# Default skill checkpoint
DEFAULT_SKILL_CHECKPOINT = "pbt_meta/skill/checkpoints/m1_v2_run1/model_249.pt"


# ============================================================
# Meta layer (upper level) — VLM + PBT modules
# ============================================================

# VLM
VLM_MODEL_TAG = "qwen2.5vl:3b"
VLM_HOST = "http://localhost:11434"

# Slow loop target rate (revised after M0 measured 0.3s warm latency)
SLOW_LOOP_HZ = 3  # ~333 ms per cycle, with headroom for VLM latency variability

# Camera (Genesis gs.Camera)
CAMERA_RES = (640, 480)
CAMERA_POS = (2.5, 0.0, 1.5)
CAMERA_LOOKAT = (0.0, 0.0, 0.5)
CAMERA_FOV = 40
