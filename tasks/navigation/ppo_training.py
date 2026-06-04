from datetime import datetime
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
from tasks.navigation.navigation import NavigationTaskEnv
import torch
import numpy as np
import wandb
from wandb.integration.sb3 import WandbCallback

custom_policy_kwargs = dict(
    activation_fn=torch.nn.Tanh,  # Aktivasyon fonksiyonu (Tanh veya ReLU kullanılabilir)
    net_arch=dict(
        pi=[128, 128],    # Aktör (Policy) Ağı: Hangi hamleyi yapacağına karar veren ağ
        vf=[128, 128]     # Kritik (Value) Ağı: Bulunduğu durumun ne kadar iyi olduğunu tahmin eden ağ
    )
)



def train():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_log = f"./tasks/navigation/tensorboard/{run_id}/"

    env = NavigationTaskEnv(render_mode="none")
    env = ss.black_death_v3(env)
    env = ss.frame_stack_v1(env, stack_size=4)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=128, num_cpus=8, base_class='stable_baselines3')
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    run = wandb.init(
        project="f15-navigation",
        sync_tensorboard=True,
        monitor_gym=True,
    )

    global_save_freq = 250_000
    real_save_freq = max(1, global_save_freq // env.num_envs)

    checkpoint_callback = CheckpointCallback(
        save_freq=real_save_freq,
        save_path='./tasks/navigation/models_checkpoints',
        name_prefix='ppo_navigation'
    )

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=custom_policy_kwargs,
        verbose=0,
        learning_rate=1e-5,
        n_steps=2048,
        n_epochs=8,
        ent_coef=0.08,
        batch_size=16384,
        gamma=0.99,
        tensorboard_log=tb_log,
    )

    model.learn(total_timesteps=100_000_000, callback=[checkpoint_callback, WandbCallback()])

    model.save("tasks/navigation/ppo_navigation_final")
    run.finish()
    print("Eğitim tamamlandı.")


if __name__ == "__main__":
    train()