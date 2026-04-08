# M3 VLM Dispatch -- Hierarchical PBT Phase 1

**Milestone**: Qwen2.5-VL-3B reads camera + state -> dispatches command -> robot executes

## Architecture

\`\`\`
  FastLoop (~30 Hz)          SlowLoop (every 10 steps)
  policy -> env -> log  <->  grab frame -> Qwen VLM -> parse JSON -> buffer
\`\`\`

## Design Decisions (Apr 8, 2026)

| Q | Decision | Rationale |
|---|----------|-----------|
| Q1 Concurrency | (a) coroutine, tick every 10 steps | Simple, no GIL risk |
| Q2 Camera | (a) subclass Go2Env | No upstream fork |
| Q3 Episode reset | (a) disable timeout (99999s) | VLM confused by teleport |
| Q4 Real-time | Optional flag, default OFF | 2x speed good for dev |
| Q5 Prompt | English + JSON + 3-tier parser | 3B model needs robust fallback |

## Files

| File | Description |
|------|-------------|
| pbt_meta/sim/go2_env_camera.py | Go2Env + camera + no timeout |
| pbt_meta/meta/vlm_prompt.py | Prompt template + 3-tier JSON parser |
| pbt_meta/meta/vlm_client.py | Ollama/Qwen HTTP client |
| pbt_meta/meta/slow_loop.py | VLM dispatch coroutine |
| pbt_meta/loop/shared_state.py | Updated buffer (M2->M3) |
| pbt_meta/loop/fast.py | Updated FastLoop (M2->M3) |
| notebooks/m3_vlm_dispatch.py | End-to-end notebook |

## PASS Criteria

- VLM receives camera frame + state text
- VLM outputs JSON command (any parse tier)
- FastLoop executes VLM command
- >=100 VLM dispatch cycles without crash
- Log shows VLM -> buffer -> velocity chain

## 3-Tier JSON Parser

1. **Tier 1**: Direct json.loads() on raw output
2. **Tier 2**: Regex extract {..} from mixed text, then json.loads()
3. **Tier 3**: Keyword scan (forward/stop/left/right) -> default velocities, or keep last command

## Known Adaptation Points

- go2_env_camera.py: Genesis API may differ -- Option B in notebook as fallback
- grab_frame_no_step(): render-only path needs testing on Genesis 0.4.3
- Policy loading (Cell 5): must match M1 ActorCritic architecture
- env_cfg (Cell 3): must copy full fields from M1 training config
