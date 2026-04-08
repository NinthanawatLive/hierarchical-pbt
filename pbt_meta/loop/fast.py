"""Fast loop — runs the borrowed walking skill at 50 Hz.

Reads commands from `SharedCommandBuffer`, writes them into `env.commands`,
calls `policy(obs_td)`, and steps Genesis. Logs every step to JSONL.
Optionally renders frames at a configurable interval.

This is the M2 single-thread version. M3 will move this into a true thread,
but the buffer-based interface means the FastLoop class itself does not need
to change.

CRITICAL: All env operations MUST happen inside `torch.inference_mode()`.
Genesis converts env tensors to inference tensors after PPO training, and
writing to them outside `inference_mode` raises RuntimeError. The
`run()` method wraps the loop in this context.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch

from pbt_meta.config import CONTROL_DT
from pbt_meta.loop.shared_state import SharedCommandBuffer


@dataclass
class FastLoopStats:
    """Per-run statistics for one fast-loop session."""
    n_steps: int = 0
    wall_clock_sec: float = 0.0
    actual_fps: float = 0.0
    n_renders: int = 0
    n_log_writes: int = 0
    target_fps: float = 50.0

    def fps_ratio(self) -> float:
        """How close to target. 1.0 = perfect, < 1.0 = slower than target."""
        if self.target_fps <= 0:
            return 0.0
        return self.actual_fps / self.target_fps


class FastLoop:
    """Single-environment fast loop for M2 + onward.

    Args:
        env: A Genesis Go2Env (or compatible) instance, already built with
             num_envs=1.
        policy: A callable that takes a TensorDict and returns an action tensor.
                Typically `runner.get_inference_policy(device='cuda:0')`.
        buffer: A SharedCommandBuffer instance. The loop reads from it every
                step.
        log_path: Where to write the JSONL log. Parent directory must exist.
        camera: Optional `gs.Camera` instance for rendering. If None, no
                rendering happens.
        render_every_n_steps: How often to call `camera.render()`. Default 25
                (= 0.5s real-time at 50 Hz, ~2 Hz visual update). Set to 0 to
                disable rendering even if a camera is provided.
        device: Torch device for the policy.
    """

    def __init__(
        self,
        env,
        policy: Callable,
        buffer: SharedCommandBuffer,
        log_path: Path,
        camera=None,
        render_every_n_steps: int = 25,
        device: str = "cuda:0",
    ) -> None:
        self.env = env
        self.policy = policy
        self.buffer = buffer
        self.log_path = Path(log_path)
        self.camera = camera
        self.render_every_n_steps = render_every_n_steps
        self.device = device

        # Frame buffer for video encoding (kept in RAM — at 50 Hz × 5 min × 25-step
        # interval × 640×480×3 bytes ≈ 550 MB which is too much. We downsample
        # storage to ONLY the rendered frames, so 600 frames × 1 MB ≈ 600 MB.
        # Acceptable on Colab L4 with 50 GB RAM.)
        self.rendered_frames: List[np.ndarray] = []
        self.rendered_frame_steps: List[int] = []

        # Sentinel to allow external stop
        self._should_stop = False

        # Verify log path is writable
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def stop(self) -> None:
        """Request the loop to stop at the next iteration boundary."""
        self._should_stop = True

    def run(
        self,
        max_steps: int,
        on_step: Optional[Callable[[int, dict], None]] = None,
    ) -> FastLoopStats:
        """Run the loop for at most `max_steps` steps.

        The loop terminates early if `stop()` is called from another callback
        (e.g., a UI button) or if max_steps is reached. Walltime is measured
        but the loop does NOT sleep — Genesis steps as fast as it can. The
        actual FPS is computed at the end and reported via stats.

        Args:
            max_steps: Maximum number of steps to run.
            on_step: Optional callback called as `on_step(step_index, log_record)`
                     after each step. Useful for live UI updates. Should be cheap
                     (< 1ms) or it will impact the loop FPS.

        Returns:
            FastLoopStats with FPS, step count, and metadata.
        """
        self._should_stop = False
        self.rendered_frames.clear()
        self.rendered_frame_steps.clear()

        stats = FastLoopStats()
        log_file = open(self.log_path, "w", buffering=8192)

        # Sanity: env must be built with n_envs=1 for this loop semantics
        if hasattr(self.env, "num_envs") and self.env.num_envs != 1:
            print(
                f"[FastLoop WARNING] env.num_envs = {self.env.num_envs}, "
                f"expected 1. Loop will read env 0 only."
            )

        t_start = time.perf_counter()

        # CRITICAL: env tensors are inference tensors after training. ALL
        # operations on env must be inside inference_mode or RuntimeError.
        with torch.inference_mode():
            # Reset env to get initial observation
            reset_res = self.env.reset()
            obs_td = reset_res[0] if isinstance(reset_res, tuple) else reset_res

            for step in range(max_steps):
                if self._should_stop:
                    break

                t_step_start = time.perf_counter()

                # 1. Read command from shared buffer
                vx, vy, wz = self.buffer.get()

                # 2. Write into env.commands
                #    All N envs get the same command (N=1 here, but supports N>1)
                self.env.commands[:, 0] = vx
                self.env.commands[:, 1] = vy
                self.env.commands[:, 2] = wz

                # 3. Policy forward
                actions = self.policy(obs_td)

                # 4. Step env
                step_res = self.env.step(actions)
                obs_td = step_res[0] if isinstance(step_res, tuple) else step_res

                # 5. Snapshot env state for log (env 0)
                base_pos = self.env.base_pos[0].cpu().numpy()
                base_ang_vel = self.env.base_ang_vel[0].cpu().numpy()
                # base_lin_vel might not exist on all env versions; guard
                base_lin_vel = None
                if hasattr(self.env, "base_lin_vel"):
                    base_lin_vel = self.env.base_lin_vel[0].cpu().numpy()

                # 6. Build log record
                t_step_end = time.perf_counter()
                step_dt_ms = (t_step_end - t_step_start) * 1000.0

                record = {
                    "step": step,
                    "t_wall": t_step_end - t_start,
                    "step_dt_ms": round(step_dt_ms, 3),
                    "cmd": [vx, vy, wz],
                    "base_pos": base_pos.tolist(),
                    "base_ang_vel": base_ang_vel.tolist(),
                    "actual_vx": float(base_lin_vel[0]) if base_lin_vel is not None else None,
                    "actual_vy": float(base_lin_vel[1]) if base_lin_vel is not None else None,
                    "actual_wz": float(base_ang_vel[2]),
                    "height": float(base_pos[2]),
                }

                # 7. Render (sparingly)
                if (
                    self.camera is not None
                    and self.render_every_n_steps > 0
                    and step % self.render_every_n_steps == 0
                ):
                    rgb, _, _, _ = self.camera.render(rgb=True)
                    if rgb is not None:
                        self.rendered_frames.append(rgb)
                        self.rendered_frame_steps.append(step)
                        stats.n_renders += 1

                # 8. Write log
                log_file.write(json.dumps(record) + "\n")
                stats.n_log_writes += 1

                # 9. User callback (UI updates, etc.)
                if on_step is not None:
                    on_step(step, record)

                stats.n_steps = step + 1

        t_end = time.perf_counter()
        log_file.close()

        stats.wall_clock_sec = t_end - t_start
        if stats.wall_clock_sec > 0:
            stats.actual_fps = stats.n_steps / stats.wall_clock_sec

        return stats
