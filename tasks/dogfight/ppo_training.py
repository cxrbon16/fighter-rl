import os
from datetime import datetime
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
from tasks.dogfight.dogfight import SelfPlayDogfightEnv
from tasks.callbacks import DogfightMetricsCallback
import torch
import numpy as np
import wandb
from wandb.integration.sb3 import WandbCallback

custom_policy_kwargs = dict(
    activation_fn=torch.nn.Tanh,
    net_arch=dict(
        pi=[256, 128, 64],
        vf=[256, 256, 128]
    )
)

def train():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_log = f"./tasks/dogfight/tensorboard/{run_id}/"

    env = SelfPlayDogfightEnv(render_mode="none")

    env = ss.black_death_v3(env)
    env = ss.frame_stack_v1(env, stack_size=4)
    env = ss.pettingzoo_env_to_vec_env_v1(env)

    env = ss.concat_vec_envs_v1(env, num_vec_envs=64, num_cpus=10, base_class='stable_baselines3')
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    run = wandb.init(
        project="f16-dogfight-selfplay",
        sync_tensorboard=True,
        monitor_gym=True,
    )

    global_save_freq = 500_000
    real_save_freq = max(1, global_save_freq // env.num_envs)

    checkpoint_dir = f'./tasks/dogfight/models_checkpoints/{run_id}'
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=real_save_freq,
        save_path=checkpoint_dir,
        name_prefix='ppo_dogfight'
    )

    metrics_callback = DogfightMetricsCallback()

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=custom_policy_kwargs,
        verbose=1,
        learning_rate=3e-4,
        n_steps=4096,
        n_epochs=5,
        ent_coef=0.01,
        batch_size=16384,
        gamma=0.99,
        tensorboard_log=tb_log,
    )

    # Experiment 3 budget (restore /24000 offensive closeness); see EXPERIMENTS.md.
    model.learn(total_timesteps=50_000_000, callback=[checkpoint_callback, WandbCallback(), metrics_callback])

    model.save("tasks/dogfight/ppo_dogfight_final")
    run.finish()

if __name__ == "__main__":
    train()
