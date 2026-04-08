"""
VLM Prompt Templates + Robust JSON Parser for M3 VLM Dispatch.
Design decisions (Apr 8, 2026): Q5 English prompt + JSON + 3-tier fallback parser.
"""
import json
import re
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class VLMCommand:
    command: str
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0
    reason: str = ""
    raw_output: str = ""
    parse_tier: int = 0


COMMAND_DEFAULTS = {
    "FORWARD":    {"vx":  0.5, "vy": 0.0, "wz": 0.0},
    "BACKWARD":   {"vx": -0.5, "vy": 0.0, "wz": 0.0},
    "TURN_LEFT":  {"vx":  0.0, "vy": 0.0, "wz":  0.8},
    "TURN_RIGHT": {"vx":  0.0, "vy": 0.0, "wz": -0.8},
    "STOP":       {"vx":  0.0, "vy": 0.0, "wz": 0.0},
}

THAI_COMMAND_MAP = {
    "\u0e40\u0e14\u0e34\u0e19\u0e44\u0e1b\u0e02\u0e49\u0e32\u0e07\u0e2b\u0e19\u0e49\u0e32": "FORWARD",
    "\u0e40\u0e14\u0e34\u0e19\u0e2b\u0e19\u0e49\u0e32": "FORWARD",
    "\u0e16\u0e2d\u0e22\u0e2b\u0e25\u0e31\u0e07": "BACKWARD",
    "\u0e40\u0e25\u0e35\u0e49\u0e22\u0e27\u0e0b\u0e49\u0e32\u0e22": "TURN_LEFT",
    "\u0e40\u0e25\u0e35\u0e49\u0e22\u0e27\u0e02\u0e27\u0e32": "TURN_RIGHT",
    "\u0e2b\u0e22\u0e38\u0e14": "STOP",
}

COMMAND_ALIASES = {
    "forward": "FORWARD", "go": "FORWARD", "walk": "FORWARD",
    "backward": "BACKWARD", "back": "BACKWARD", "reverse": "BACKWARD",
    "turn_left": "TURN_LEFT", "left": "TURN_LEFT", "turnleft": "TURN_LEFT",
    "turn_right": "TURN_RIGHT", "right": "TURN_RIGHT", "turnright": "TURN_RIGHT",
    "stop": "STOP", "halt": "STOP", "wait": "STOP", "stand": "STOP",
}


SYSTEM_PROMPT = """You are a quadruped robot controller. You receive a camera image from the robot's front-facing camera and a text summary of the robot's current state.

Your task: decide the next movement command for the robot.

Output ONLY a valid JSON object with this exact format -- no other text before or after:
{"command": "<CMD>", "vx": <float>, "wz": <float>, "reason": "<brief explanation>"}

Valid commands: FORWARD, BACKWARD, TURN_LEFT, TURN_RIGHT, STOP
Valid ranges: vx in [-1.0, 1.0], wz in [-1.0, 1.0]

Examples:
{"command": "FORWARD", "vx": 0.5, "wz": 0.0, "reason": "clear path ahead"}
{"command": "TURN_LEFT", "vx": 0.0, "wz": 0.8, "reason": "obstacle on the right"}
{"command": "STOP", "vx": 0.0, "wz": 0.0, "reason": "reached target area"}
"""


def build_state_summary(vx, vy, wz, roll_deg, pitch_deg, pos_x, pos_y,
                        last_command="STOP", last_command_duration_s=0.0, extra=""):
    lines = [
        f"Robot state: vx={vx:.2f} m/s, vy={vy:.2f} m/s, wz={wz:.2f} rad/s, "
        f"roll={roll_deg:.1f} deg, pitch={pitch_deg:.1f} deg",
        f"Position: x={pos_x:.1f}m, y={pos_y:.1f}m",
        f"Last command: {last_command} for {last_command_duration_s:.1f}s",
    ]
    if extra:
        lines.append(extra)
    return "\n".join(lines)


def build_vlm_prompt(state_summary, user_instruction="", lang="en"):
    if lang == "th" and user_instruction:
        mapped = THAI_COMMAND_MAP.get(user_instruction.strip())
        if mapped:
            user_instruction = f"Execute: {mapped}"
        else:
            user_instruction = f"User instruction (Thai): {user_instruction}"
    parts = [state_summary]
    if user_instruction:
        parts.append(f"\nHigh-level goal: {user_instruction}")
    parts.append("\nDecide the next command. Output JSON only.")
    return SYSTEM_PROMPT, "\n".join(parts)


# ========== 3-tier JSON parser ==========

def parse_vlm_response(raw_output, last_command=None):
    raw_output = raw_output.strip()
    cmd = _try_json_parse(raw_output)
    if cmd is not None:
        cmd.raw_output = raw_output
        cmd.parse_tier = 1
        return cmd
    cmd = _try_regex_extract(raw_output)
    if cmd is not None:
        cmd.raw_output = raw_output
        cmd.parse_tier = 2
        return cmd
    cmd = _try_keyword_scan(raw_output, last_command)
    cmd.raw_output = raw_output
    cmd.parse_tier = 3
    return cmd


def _try_json_parse(text):
    try:
        data = json.loads(text)
        return _dict_to_command(data)
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def _try_regex_extract(text):
    matches = re.findall(r'\{[^{}]+\}', text)
    for match in matches:
        try:
            data = json.loads(match)
            if "command" in data:
                return _dict_to_command(data)
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return None


def _try_keyword_scan(text, last_command):
    text_lower = text.lower()
    for keyword, cmd_name in [
        ("stop", "STOP"), ("backward", "BACKWARD"), ("back", "BACKWARD"),
        ("turn_left", "TURN_LEFT"), ("turn left", "TURN_LEFT"), ("left", "TURN_LEFT"),
        ("turn_right", "TURN_RIGHT"), ("turn right", "TURN_RIGHT"), ("right", "TURN_RIGHT"),
        ("forward", "FORWARD"), ("walk", "FORWARD"), ("go", "FORWARD"),
    ]:
        if keyword in text_lower:
            defaults = COMMAND_DEFAULTS[cmd_name]
            return VLMCommand(command=cmd_name, vx=defaults["vx"], wz=defaults["wz"],
                              reason=f"[tier3-keyword: '{keyword}']")
    if last_command is not None:
        return VLMCommand(command=last_command.command, vx=last_command.vx, wz=last_command.wz,
                          reason="[tier3-fallback: keeping last command]")
    return VLMCommand(command="STOP", vx=0.0, wz=0.0,
                      reason="[tier3-fallback: no parse, defaulting STOP]")


def _dict_to_command(data):
    cmd_raw = str(data.get("command", "")).upper().strip()
    cmd_name = COMMAND_ALIASES.get(cmd_raw.lower(), cmd_raw)
    if cmd_name not in COMMAND_DEFAULTS:
        raise KeyError(f"Unknown command: {cmd_raw}")
    defaults = COMMAND_DEFAULTS[cmd_name]
    vx = max(-1.0, min(1.0, float(data.get("vx", defaults["vx"]))))
    wz = max(-1.0, min(1.0, float(data.get("wz", defaults["wz"]))))
    return VLMCommand(command=cmd_name, vx=vx, wz=wz, reason=str(data.get("reason", "")))
