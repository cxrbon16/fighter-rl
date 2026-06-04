# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Experiment Results

When the user asks anything about training metrics, reward curves, learning progress, or experimental results, **immediately load the experiment data** before answering — do not ask the user to run `/exp-results` first:

```python
import json
with open('tasks/exp_results.json') as f:
    all_results = json.load(f)
```

If new training runs may have been added since the snapshot, re-parse from source instead (see `.claude/commands/exp-results.md`).

## Project Overview

This is a reinforcement learning project for training F-16 fighter jet agents to perform air-to-air dogfighting. Agents are trained using PPO (Proximal Policy Optimization) via Stable-Baselines3 in a multi-agent self-play setup. The flight simulation backend is JSBSim.

## Debugging Rollouts

To inspect agent behaviour in a specific episode, run the rollout debugger (see `/rollout-debug` for the full skill):

```bash
# Random policy — env sanity check
python -m tasks.rollout_debug --random

# Trained model
python -m tasks.rollout_debug --model tasks/dogfight/ppo_dogfight_final.zip

# Curriculum model
python -m tasks.rollout_debug --task curriculum \
    --model tasks/curriculum_dogfight/Phase_2_Approaching_and_Offensive_final.zip \
    --vecnorm tasks/curriculum_dogfight/Phase_2_Approaching_and_Offensive_final_vecnormalize.pkl
```

Output lands in `tasks/debug_runs/<timestamp>/`: JSON telemetry + PNG plots per episode. Read the PNGs with the Read tool and load the JSON to answer step-level questions about the flight.

## Running Training and Tests

All scripts must be run from the repository root (not from within subdirectories), since imports are relative to the project root:

```bash
# Standard dogfight training (self-play)
python -m tasks.dogfight.ppo_training

# Curriculum dogfight training (3-phase curriculum)
python -m tasks.curriculum_dogfight.curriculum_training

# Test/visualize standard dogfight (requires Rerun viewer)
python -m tasks.dogfight.test

# Test/visualize curriculum dogfight (requires Rerun viewer)
python -m tasks.curriculum_dogfight.test
```

Monitor training with TensorBoard:
```bash
tensorboard --logdir tasks/dogfight/dogfight_tensorboard
tensorboard --logdir tasks/curriculum_dogfight/tensorboard
```

Weights & Biases is also used for experiment tracking; `wandb` must be logged in before training.

## Architecture

### `lib/base_env.py` — `BaseEnv`
Abstract base class extending PettingZoo's `ParallelEnv`. Manages:
- JSBSim FDM (Flight Dynamics Model) instances, one per agent
- Simulation stepping: 8 JSBSim substeps per RL step
- Action space: 3-dimensional continuous `[-1, 1]` — aileron, elevator, throttle
- Render modes: `"none"` (training), `"debug"` (telemetry in infos), `"human"` (FlightGear via XML protocol)

Subclasses must implement: `_get_initial_conditions()`, `_get_obs(agent_id)`, `_calculate_rewards_and_dones(actions)`. Optional override: `_task_reset()`.

### `tasks/dogfight/dogfight.py` — `SelfPlayDogfightEnv`
Concrete 2-agent self-play environment. Observation space: 28-dimensional normalized float32 vector covering own kinematics, aerodynamics, control surface states, relative geometry (ATA/AA angles, distance), opponent state, and specific energy (Energy Maneuverability Theory).

Victory condition: maintain WEZ (Weapon Engagement Zone — within 3000 ft and ATA < 10°) for 20 consecutive steps. Termination conditions: altitude < 1000 ft (crash), distance > 150,000 ft (out of bounds), sim time > 180 s (truncation).

### `tasks/curriculum_dogfight/dogfight.py` — `SelfPlayDogfightEnv` (curriculum variant)
Same environment with an additional `reward_weights` dict parameter. Each reward component is multiplied by its weight, enabling curriculum-phase-specific reward shaping without changing reward logic.

### Training Pipeline (both tasks)
The environment is wrapped with SuperSuit then SB3:
1. `ss.black_death_v3` — removes terminated agents' observations gracefully
2. `ss.frame_stack_v1` — stacks N frames (4 for standard, 8 for curriculum)
3. `ss.pettingzoo_env_to_vec_env_v1` — converts to SB3-compatible VecEnv
4. `ss.concat_vec_envs_v1` — runs N parallel environments (64 for standard, 32 for curriculum)
5. `VecMonitor` + `VecNormalize` — reward normalization (obs normalization disabled)

PPO policy: MLP with `pi=[256,128,64]`, `vf=[256,256,128]`, Tanh activation, `lr=1e-5`, `gamma=0.99`.

### Curriculum Phases (`curriculum_training.py`)
Three sequential phases defined in the `PHASES` list. Each phase specifies timesteps and per-component reward weights:
- **Phase 1** (40M steps): Basic flight survival — no offensive rewards
- **Phase 2** (200M steps): Approaching and offensive geometry
- **Phase 3** (1B steps): WEZ entry and victory emphasis

Between phases, VecNormalize stats (`.pkl`) and model weights (`.zip`) are saved and reloaded to maintain training continuity.

### Visualization
Tests use [Rerun](https://rerun.io/) for 3D visualization. The F-16 model (`static/f16.glb`) is rendered with trajectory trails. Coordinate conversion: JSBSim NED Euler angles → Rerun ENU quaternions.

## Key Paths
- Saved models: `tasks/dogfight/models_checkpoints/`, `tasks/curriculum_dogfight/models_checkpoints/`
- Final models: `tasks/dogfight/ppo_dogfight_final.zip`, `tasks/curriculum_dogfight/<Phase>_final.zip`
- VecNormalize stats: `tasks/curriculum_dogfight/<Phase>_final_vecnormalize.pkl`
- JSBSim config for FlightGear output: `config/agent_1_protocol.xml`
