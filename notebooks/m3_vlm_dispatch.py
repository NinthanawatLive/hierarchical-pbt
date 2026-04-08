"""
M3 VLM Dispatch -- Qwen2.5-VL -> SharedCommandBuffer -> FastLoop
Hierarchical PBT Phase 1

Convert to Colab notebook or copy cells. Each # %% is a code cell.

PASS criteria:
  1. VLM receives camera frame + state text
  2. VLM outputs JSON command (any parse tier)
  3. FastLoop executes VLM command (robot moves)
  4. >=100 VLM dispatch cycles without crash
  5. Log shows VLM -> buffer -> velocity chain
"""

# %% Cell 1: Start Ollama + Qwen
import subprocess, time, os, sys
ollama_proc = subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(3)
os.system("ollama pull qwen2.5-vl:3b")
time.sleep(2)
import requests
resp = requests.get("http://localhost:11434/api/tags")
models = [m["name"] for m in resp.json().get("models", [])]
print(f"Models: {models}")
assert any("qwen" in m for m in models), "Qwen not found!"
print("OK: Ollama + Qwen ready")

# %% Cell 2: Genesis init
import genesis as gs
gs.init(backend=gs.cuda)
import torch
import numpy as np
sys.path.insert(0, "/content/hierarchical-pbt")
print(f"Genesis {gs.__version__}, PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}")

# %% Cell 3: Env config (copy from M1, override for M3)
from go2_env import Go2Env

env_cfg = {
    "num_actions": 12,
    "episode_length_s": 99999,    # M3: disable timeout
    "resampling_time_s": 99999,
    "action_scale": 0.25,
    "kp": 20.0, "kd": 0.5,
}
obs_cfg = {"num_obs": 48}
reward_cfg = {"tracking_lin_vel": 1.0, "tracking_ang_vel": 1.0}
command_cfg = {
    "num_commands": 3,
    "lin_vel_x_range": [-1.0, 1.0],
    "lin_vel_y_range": [0.0, 0.0],
    "ang_vel_range": [-1.0, 1.0],
}

# %% Cell 4: Create env with camera
# Option A: wrapper
try:
    from pbt_meta.sim.go2_env_camera import Go2EnvWithCamera, DEFAULT_CAMERA_CFG
    env = Go2EnvWithCamera(num_envs=1, env_cfg=env_cfg, obs_cfg=obs_cfg,
                           reward_cfg=reward_cfg, command_cfg=command_cfg,
                           camera_cfg=DEFAULT_CAMERA_CFG, show_viewer=False)
    print("OK: Go2EnvWithCamera (Option A)")
    USE_WRAPPER = True
except Exception as e:
    print(f"Option A failed: {e}\n-> Use Option B below")
    USE_WRAPPER = False

# %% Cell 4b: Option B -- manual env + camera
if not USE_WRAPPER:
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.02), show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(gs.morphs.URDF(file="urdf/go2/urdf/go2.urdf", pos=(0,0,0.42)))
    camera = scene.add_camera(res=(320,240), pos=(0.3,0.0,0.57), lookat=(2.0,0.0,0.22), fov=60, GUI=False)
    scene.build(n_envs=1)
    env_cfg["episode_length_s"] = 99999
    env = Go2Env(num_envs=1, env_cfg=env_cfg, obs_cfg=obs_cfg,
                 reward_cfg=reward_cfg, command_cfg=command_cfg, show_viewer=False)
    env.camera = camera
    env.camera_cfg = {"width": 320, "height": 240, "pos_offset": (0.3,0.0,0.15), "lookat_offset": (2.0,0.0,-0.2)}
    def grab_frame_no_step(env_idx=0):
        try:
            camera.render()
            return np.array(camera.get_rgb())
        except:
            camera.start_recording()
            scene.step()
            f = camera.stop_recording()
            return f[-1] if len(f.shape)==4 else f
    env.grab_frame_no_step = grab_frame_no_step
    print("OK: Manual env with camera (Option B)")

# %% Cell 5: Load M1 policy
checkpoint_path = "pbt_meta/skill/checkpoints/m1_v2_run1/model_249.pt"
checkpoint = torch.load(checkpoint_path, map_location="cuda")
print(f"Checkpoint: {checkpoint_path}")
print(f"Keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'tensor'}")
# TODO: reconstruct your ActorCritic and load weights here
policy = None  # Replace with loaded actor

# %% Cell 6: Init M3 components
from pbt_meta.loop.shared_state import SharedCommandBuffer
from pbt_meta.meta.vlm_client import OllamaVLMClient
from pbt_meta.meta.slow_loop import SlowLoop
from pbt_meta.loop.fast import FastLoop

buffer = SharedCommandBuffer()
vlm_client = OllamaVLMClient(model="qwen2.5-vl:3b", timeout_s=30.0, temperature=0.1)
assert vlm_client.health_check(), "Ollama not reachable!"
slow_loop = SlowLoop(vlm_client=vlm_client, command_buffer=buffer, env=env,
                     tick_interval=10, user_instruction="explore the area", lang="en")
fast_loop = FastLoop(env=env, policy=policy, buffer=buffer, slow_loop=slow_loop,
                     real_time=False, dt=0.02, log_every=1)
print("OK: All M3 components ready")

# %% Cell 7: VLM smoke test
from pbt_meta.meta.vlm_prompt import build_vlm_prompt, build_state_summary, parse_vlm_response
test_frame = env.grab_frame_no_step(0)
print(f"Frame: {test_frame.shape} {test_frame.dtype}")
state_text = build_state_summary(0,0,0, 0,0, 0,0, "STOP", 0.0)
sys_p, usr_p = build_vlm_prompt(state_text, "explore the area")
vlm_resp = vlm_client.query(test_frame, sys_p, usr_p)
print(f"VLM success={vlm_resp.success}, latency={vlm_resp.latency_ms:.0f}ms")
print(f"Raw: {vlm_resp.text}")
cmd = parse_vlm_response(vlm_resp.text)
print(f"Parsed: {cmd.command} vx={cmd.vx} wz={cmd.wz} tier={cmd.parse_tier}")

# %% Cell 8: Run M3 full loop
import logging
logging.basicConfig(level=logging.INFO, format='%(name)s | %(message)s')
NUM_STEPS = 1500
os.makedirs("logs", exist_ok=True)
print(f"Running M3: {NUM_STEPS} steps, VLM every {slow_loop.tick_interval} steps")
summary = fast_loop.run(num_steps=NUM_STEPS, log_path="logs/m3_fast.jsonl")
slow_loop.export_log_jsonl("logs/m3_slow.jsonl")
print(f"\nSteps: {summary['total_steps']}, Hz: {summary['hz']}, Wall: {summary['wall_time_s']}s")
if summary['slow_loop_stats']:
    sl = summary['slow_loop_stats']
    print(f"VLM calls: {sl['total_vlm_calls']}, success: {sl['success_rate']:.0%}, tiers: {sl['tier_counts']}")

# %% Cell 9: PASS verification
sl_stats = slow_loop.get_stats()
checks = {
    "VLM received frames": sl_stats["total_vlm_calls"] > 0,
    "VLM produced commands": sl_stats["successful_vlm_calls"] > 0,
    ">=100 dispatch cycles": sl_stats["tick_count"] >= 100,
    "No crash": summary["total_steps"] >= NUM_STEPS * 0.95,
    "Parse success (tier 1 or 2)": sl_stats["tier_counts"].get(1,0) + sl_stats["tier_counts"].get(2,0) > 0,
}
all_pass = True
for name, passed in checks.items():
    s = "PASS" if passed else "FAIL"
    print(f"  {s}  {name}")
    if not passed: all_pass = False
print(f"\n{'M3 PASS' if all_pass else 'M3 NOT YET PASSED'}")
