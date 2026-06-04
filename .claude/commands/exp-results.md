# Load Experiment Results

Parse all TensorBoard logs in this project into memory so you can answer questions about training metrics directly.

## Steps

1. Run the parser to load all runs:

```python
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import json, os

def parse_dir(base_dir, label):
    results = {}
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if not f.startswith('events.out.tfevents'): continue
            ea = EventAccumulator(root)
            ea.Reload()
            tags = ea.Tags().get('scalars', [])
            rel = os.path.relpath(root, base_dir)
            key = f'{label}/{rel}'
            results[key] = {}
            for tag in tags:
                events = ea.Scalars(tag)
                results[key][tag] = [[e.step, round(e.value, 6)] for e in events]
    return results

all_results = {}
all_results.update(parse_dir('tasks/curriculum_dogfight/tensorboard', 'curriculum'))
all_results.update(parse_dir('tasks/dogfight/dogfight_tensorboard', 'dogfight_selfplay'))
print(json.dumps({k: list(v.keys()) for k, v in all_results.items()}, indent=2))
```

2. A pre-dumped snapshot is also available at `tasks/exp_results.json` — load it with:

```python
import json
with open('tasks/exp_results.json') as f:
    all_results = json.load(f)
```

## Run structure

**Curriculum training** (`curriculum/Phase_*/PPO_1`):
- `Phase_1_Basics` — 40M steps (basic flight survival)
- `Phase_2_Approaching_and_Offensive` — 200M steps (approach + geometry)
- `Phase_3_WEZ_and_Victory` — ~5M steps (cut short)

**Self-play runs** (`dogfight_selfplay/PPO_*`):
- PPO_5: 200M steps (longest self-play run)
- PPO_9: 36M steps, PPO_10: 22M steps, PPO_12: 14M steps

## Available metrics (per run)

| Group | Keys |
|---|---|
| `rollout/` | `ep_rew_mean`, `ep_len_mean` |
| `metrics/` | `avg_distance_ft`, `avg_energy`, `avg_tracking_time` |
| `rewards/` | `survival_reward`, `crash_penalty`, `out_of_bounds_penalty`, `delta_energy_reward`, `action_penalty`, `distance_reward`, `offensive_reward`, `g_limit_penalty`, `wez_reward`, `victory_reward`, `defeat_penalty` |
| `train/` | `approx_kl`, `clip_fraction`, `entropy_loss`, `explained_variance`, `loss`, `policy_gradient_loss`, `std`, `value_loss` |

## Usage

After loading, answer any metric question directly. Example:

```python
# Get wez_reward progression for Phase 2
data = all_results['curriculum/Phase_2_Approaching_and_Offensive/PPO_1']['rewards/wez_reward']
# data is a list of [step, value] pairs
```

Note: the JSON snapshot at `tasks/exp_results.json` may be stale if new training runs have been added. Re-run the parser in step 1 to refresh.
