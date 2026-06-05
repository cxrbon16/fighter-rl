# Self-Play Reinforcement Learning for Superhuman Air Combat Policies

Reinforcement learning project for training F-16 agents to perform air-to-air dogfighting using [JSBSim](https://github.com/JSBSim-Team/jsbsim) for flight dynamics and [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) PPO for training.

## Overview

Two agents fly against each other in a self-play loop. A win is achieved by staying within the Weapon Engagement Zone (WEZ) — within 3000 ft and antenna train angle < 10° — for 20 consecutive steps. Training uses a 3-phase curriculum that progressively introduces offensive and lethality rewards on top of basic flight survival.

**Observation space (28-dim):** own kinematics, aerodynamics, control surface state, relative geometry (ATA/AA angles, distance), opponent state, specific energy (Energy Maneuverability Theory).

**Action space (3-dim, continuous [-1, 1]):** aileron, elevator, throttle.

## Structure

```
lib/                        Base JSBSim environment (BaseEnv)
tasks/
  dogfight/                 Self-play training and test
  curriculum_dogfight/      3-phase curriculum training and test
  navigation/               Single-agent navigation task (F-15)
  climb/                    Single-agent climb task
  rollout_debug.py          Episode debugger — JSON telemetry + PNG plots
```

## Training

```bash
# Self-play
python -m tasks.dogfight.ppo_training

# Curriculum (3 phases: Basics → Approach & Offensive → WEZ & Victory)
python -m tasks.curriculum_dogfight.curriculum_training
```

TensorBoard logs land in `tasks/<task>/tensorboard/<run_timestamp>/`. Each training run gets its own timestamped directory.

```bash
tensorboard --logdir tasks/curriculum_dogfight/tensorboard
```

## Visualization & Debugging

```bash
# Interactive 3D viewer (Rerun)
python -m tasks.curriculum_dogfight.test

# Rollout debugger — saves JSON + plots to tasks/debug_runs/
python -m tasks.rollout_debug --model tasks/curriculum_dogfight/Phase_2_Approaching_and_Offensive_final.zip --task curriculum
```

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies: `jsbsim`, `stable-baselines3`, `supersuit`, `pettingzoo`, `wandb`, `rerun-sdk`.
