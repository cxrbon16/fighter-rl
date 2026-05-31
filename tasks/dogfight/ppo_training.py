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
        self.env_data = {} # Her env_idx için ayrı veri tutacağız

    def _on_step(self) -> bool:
        n_envs = len(self.locals['dones'])
        
        # İlk adımda env_data sözlüğünü başlat
        if not self.env_data:
            for i in range(n_envs):
                self.env_data[i] = {"dists": [], "energies": [], "tracking": []}

        for i, (info, done) in enumerate(zip(self.locals['infos'], self.locals['dones'])):
            # Verileri topla ve NaN kontrolü yap
            dist = info.get('dist_ft', 0)
            energy = info.get('energy', 0)
            tracking = info.get('tracking_time', 0)

            # JSBSim NaN dönerse 0 kabul et
            self.env_data[i]["dists"].append(np.nan_to_num(dist))
            self.env_data[i]["energies"].append(np.nan_to_num(energy))
            self.env_data[i]["tracking"].append(np.nan_to_num(tracking))

            if done:
                # Sadece veri varsa logla (NaN önlemek için)
                if len(self.env_data[i]["dists"]) > 0:
                    self.logger.record("metrics/avg_distance_ft", np.mean(self.env_data[i]["dists"]))
                    self.logger.record("metrics/avg_energy", np.mean(self.env_data[i]["energies"]))
                    self.logger.record("metrics/avg_tracking_time", np.mean(self.env_data[i]["tracking"]))
                    
                    # Verileri sıfırla
                    self.env_data[i] = {"dists": [], "energies": [], "tracking": []}
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
