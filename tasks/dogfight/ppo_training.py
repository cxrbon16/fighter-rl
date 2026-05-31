import os
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
from tasks.dogfight.dogfight import SelfPlayDogfightEnv
import torch
import numpy as np
import wandb
from wandb.integration.sb3 import WandbCallback

class DogfightMetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(DogfightMetricsCallback, self).__init__(verbose)
        self.episode_dists = []
        self.episode_energies = []
        self.episode_tracking_times = []
        self.agent_wins = {"agent_1": 0, "agent_2": 0}

    def _on_step(self) -> bool:
        for info in self.locals['infos']:
            if 'dist_ft' in info:
                self.episode_dists.append(info['dist_ft'])
            if 'energy' in info:
                self.episode_energies.append(info['energy'])
            if 'tracking_time' in info:
                self.episode_tracking_times.append(info['tracking_time'])

        for done, info in zip(self.locals['dones'], self.locals['infos']):
            if done:
                if 'dist_ft' in info:
                    self.logger.record("metrics/avg_distance_ft", np.mean(self.episode_dists))
                    self.logger.record("metrics/avg_energy", np.mean(self.episode_energies))
                    self.logger.record("metrics/avg_tracking_time", np.mean(self.episode_tracking_times))
                    
                    # Kazanan tespiti: tracking_time > 50 olan kazanmıştır
                    if info.get('tracking_time', 0) >= 50:
                        # Bu info hangi ajana aitse o kazanmıştır, ama VecEnv'de bunu anlamak 
                        # info içindeki yapıya göre değişir. PettingZoo VecEnv'de ajanlar ardışıktır.
                        pass 

                    self.episode_dists = []
                    self.episode_energies = []
                    self.episode_tracking_times = []
        return True

custom_policy_kwargs = dict(
    activation_fn=torch.nn.Tanh,
    net_arch=dict(
        pi=[512, 512, 512],
        vf=[512, 512, 512]
    )
)

def train():
    env = SelfPlayDogfightEnv(render_mode="none")
    
    env = ss.black_death_v3(env) 
    env = ss.frame_stack_v1(env, stack_size=4)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    
    env = ss.concat_vec_envs_v1(env, num_vec_envs=64, num_cpus=8, base_class='stable_baselines3')
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
    
    metrics_callback = DogfightMetricsCallback()

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

    model.learn(total_timesteps=100_000_000, callback=[checkpoint_callback, WandbCallback(), metrics_callback])

    model.save("tasks/dogfight/ppo_dogfight_final")
    run.finish()

if __name__ == "__main__":
    train()
