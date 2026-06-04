# Rollout Debug

Run a dogfight episode and produce per-step JSON telemetry + PNG plots so you can analyze agent behaviour directly.

## Usage

```bash
# Random policy (no model needed — good for sanity-checking the env)
python -m tasks.rollout_debug --random

# Standard dogfight model
python -m tasks.rollout_debug --model tasks/dogfight/ppo_dogfight_final.zip

# Curriculum model (with VecNormalize stats)
python -m tasks.rollout_debug \
    --task curriculum \
    --model tasks/curriculum_dogfight/Phase_2_Approaching_and_Offensive_final.zip \
    --vecnorm tasks/curriculum_dogfight/Phase_2_Approaching_and_Offensive_final_vecnormalize.pkl

# Multiple episodes, custom output dir
python -m tasks.rollout_debug --model PATH --episodes 3 --out tasks/debug_runs/my_run
```

## Output (per episode)

| File | Contents |
|---|---|
| `ep01_telemetry.json` | Per-step dict for agent_1 and agent_2: lat/lon/alt, roll/pitch/yaw, airspeed, distance, energy, tracking_time, reward, all reward_components |
| `ep01_overview.png` | 6-panel plot: altitude, inter-agent distance, airspeed, WEZ tracking time, specific energy, top-down flight path |
| `ep01_rewards.png` | One subplot per reward component (agent_1), with episode total in title |
| `ep01_summary.txt` | Episode outcome, step count, min/max/mean for key metrics, reward component totals |

## After running

Load the telemetry to answer specific questions:

```python
import json
with open("tasks/debug_runs/<timestamp>/ep01_telemetry.json") as f:
    log = json.load(f)

a1 = log["agent_1"]   # list of per-step dicts
a2 = log["agent_2"]

# Example: find the step where agent_1 first entered WEZ range
wez_steps = [f["step"] for f in a1 if f["dist_ft"] < 3000]

# Example: find minimum altitude reached
min_alt = min(f["alt_m"] for f in a1)
```

Read the PNG files with the Read tool to visually inspect the flight.

## Key episode outcomes

- `agent_1_wins` / `agent_2_wins` — tracking_time exceeded 20 while in WEZ
- `crash` — altitude dropped below 1000 ft
- `out_of_bounds` — distance exceeded 150,000 ft
- `truncation` — episode hit the 180 s / ~10800 step limit
