"""
SlowLoop -- VLM dispatch coroutine for M3.
Q1: (a) coroutine in same thread, called every 10 fast steps.
"""
import time
import logging
import json
from dataclasses import dataclass
from typing import Optional, List
import numpy as np

from pbt_meta.meta.vlm_prompt import (
    build_vlm_prompt, build_state_summary, parse_vlm_response, VLMCommand,
)
from pbt_meta.meta.vlm_client import OllamaVLMClient

logger = logging.getLogger(__name__)


@dataclass
class SlowTickLog:
    step: int
    wall_time: float
    vlm_latency_ms: float
    vlm_success: bool
    command: str
    vx: float
    wz: float
    parse_tier: int
    reason: str
    raw_output: str = ""


class SlowLoop:
    """VLM dispatch coroutine -- called by FastLoop every N steps."""

    def __init__(self, vlm_client, command_buffer, env, tick_interval=10,
                 user_instruction="", lang="en", log_raw_output=True, env_idx=0):
        self.vlm_client = vlm_client
        self.buffer = command_buffer
        self.env = env
        self.tick_interval = tick_interval
        self.user_instruction = user_instruction
        self.lang = lang
        self.log_raw_output = log_raw_output
        self.env_idx = env_idx
        self.last_command = None
        self.last_command_step = 0
        self.tick_count = 0
        self.tick_log = []
        self.total_vlm_calls = 0
        self.successful_vlm_calls = 0
        self.tier_counts = {1: 0, 2: 0, 3: 0}

    def should_tick(self, step):
        return step % self.tick_interval == 0

    def tick(self, step):
        if not self.should_tick(step):
            return None
        t0 = time.perf_counter()
        self.tick_count += 1

        # 1. Grab frame
        try:
            frame = self.env.grab_frame_no_step(self.env_idx)
        except Exception as e:
            logger.warning(f"SlowLoop frame grab failed: {e}")
            frame = np.zeros((240, 320, 3), dtype=np.uint8)

        # 2. Build state summary
        state_summary = self._get_state_summary(step)

        # 3. Build prompt
        system_prompt, user_prompt = build_vlm_prompt(
            state_summary=state_summary,
            user_instruction=self.user_instruction, lang=self.lang)

        # 4. Query VLM
        self.total_vlm_calls += 1
        vlm_response = self.vlm_client.query(frame, system_prompt, user_prompt)

        # 5. Parse
        if vlm_response.success:
            self.successful_vlm_calls += 1
            cmd = parse_vlm_response(vlm_response.text, last_command=self.last_command)
        else:
            logger.warning(f"SlowLoop VLM failed: {vlm_response.error}")
            if self.last_command:
                cmd = VLMCommand(command=self.last_command.command,
                                 vx=self.last_command.vx, wz=self.last_command.wz,
                                 reason=f"[vlm-failed: {vlm_response.error[:80]}]", parse_tier=3)
            else:
                cmd = VLMCommand(command="STOP", vx=0.0, wz=0.0,
                                 reason="[vlm-failed, no history]", parse_tier=3)

        # 6. Write to buffer
        try:
            self.buffer.write(cmd.vx, cmd.wz, f"vlm:{cmd.command}")
        except Exception as e:
            logger.error(f"Buffer write failed: {e}")

        # 7. Update state
        self.last_command = cmd
        self.last_command_step = step
        self.tier_counts[cmd.parse_tier] = self.tier_counts.get(cmd.parse_tier, 0) + 1

        # 8. Log
        wall_time = time.perf_counter() - t0
        self.tick_log.append(SlowTickLog(
            step=step, wall_time=wall_time, vlm_latency_ms=vlm_response.latency_ms,
            vlm_success=vlm_response.success, command=cmd.command,
            vx=cmd.vx, wz=cmd.wz, parse_tier=cmd.parse_tier, reason=cmd.reason,
            raw_output=vlm_response.text[:500] if self.log_raw_output else ""))

        logger.info(f"SlowLoop [{step}] -> {cmd.command} vx={cmd.vx:.2f} wz={cmd.wz:.2f} "
                     f"(tier={cmd.parse_tier}, vlm={vlm_response.latency_ms:.0f}ms)")
        return cmd

    def _get_state_summary(self, step):
        try:
            def to_np(t):
                if hasattr(t, 'cpu'): t = t.cpu().numpy()
                if hasattr(t, 'shape') and len(t.shape) > 1: t = t[self.env_idx]
                return np.array(t, dtype=np.float32)
            pos = to_np(self.env.base_pos)
            lin_vel = to_np(self.env.base_lin_vel)
            ang_vel = to_np(self.env.base_ang_vel)
            duration_s = (step - self.last_command_step) * 0.02
            return build_state_summary(
                vx=float(lin_vel[0]), vy=float(lin_vel[1]),
                wz=float(ang_vel[2]) if len(ang_vel) > 2 else 0.0,
                roll_deg=0.0, pitch_deg=0.0,
                pos_x=float(pos[0]), pos_y=float(pos[1]),
                last_command=self.last_command.command if self.last_command else "STOP",
                last_command_duration_s=duration_s)
        except Exception as e:
            logger.warning(f"State summary failed: {e}")
            return "Robot state: unavailable"

    def get_stats(self):
        return {
            "tick_count": self.tick_count,
            "total_vlm_calls": self.total_vlm_calls,
            "successful_vlm_calls": self.successful_vlm_calls,
            "success_rate": self.successful_vlm_calls / max(1, self.total_vlm_calls),
            "tier_counts": dict(self.tier_counts),
            "avg_vlm_latency_ms": self.vlm_client.avg_latency_ms,
            "last_command": self.last_command.command if self.last_command else None,
        }

    def export_log_jsonl(self, path):
        with open(path, "w") as f:
            for e in self.tick_log:
                row = {"step": e.step, "wall_time": round(e.wall_time, 4),
                       "vlm_latency_ms": round(e.vlm_latency_ms, 1),
                       "vlm_success": e.vlm_success, "command": e.command,
                       "vx": round(e.vx, 3), "wz": round(e.wz, 3),
                       "parse_tier": e.parse_tier, "reason": e.reason}
                if e.raw_output:
                    row["raw_output"] = e.raw_output
                f.write(json.dumps(row) + "\n")
