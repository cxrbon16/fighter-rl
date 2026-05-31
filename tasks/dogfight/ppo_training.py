import os
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
from tasks.dogfight.dogfight import SelfPlayDogfightEnv
import torch
import numpy as np
import wandb
from wandb.integration.sb3 import WandbCallback

custom_policy_kwargs = dict(
    activation_fn=torch.nn.Tanh,
    net_arch=dict(
        pi=[256, 256],
        vf=[256, 256]
    )
)

def train():
    env = SelfPlayDogfightEnv(render_mode="none")
    
    env = ss.black_death_v3(env) 
    env = ss.frame_stack_v1(env, stack_size=4)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    
    env = ss.concat_vec_envs_v1(env, num_vec_envs=8, num_cpus=8, base_class='stable_baselines3')
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    run = wandb.init(
        project="f16-dogfight-selfplay",
        sync_tensorboard=True,
        monitor_gym=True,
    )

    global_save_freq = 500_000
    real_save_freq = max(1, global_save_freq // env.num_envs)

    os.makedirs('./tasks/dogfight/models_checkpoints', exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=real_save_freq, 
        save_path='./tasks/dogfight/models_checkpoints',
        name_prefix='ppo_dogfight'
    )

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=custom_policy_kwargs,
        verbose=1,
        learning_rate=3e-5,
        n_steps=2048,
        n_epochs=10,
        ent_coef=0.01,
        batch_size=8192,
        gamma=0.99,
        tensorboard_log="./tasks/dogfight/dogfight_tensorboard/"
    )

    model.learn(total_timesteps=100_000_000, callback=[checkpoint_callback, WandbCallback()])

    model.save("tasks/dogfight/ppo_dogfight_final")
    run.finish()

if __name__ == "__main__":
    train()
