import os
from datetime import datetime
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
from tasks.curriculum_dogfight.dogfight import SelfPlayDogfightEnv
import torch
import numpy as np
import wandb
from wandb.integration.sb3 import WandbCallback

class DogfightMetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(DogfightMetricsCallback, self).__init__(verbose)
        self.env_data = {} 

    def _on_step(self) -> bool:
        n_envs = len(self.locals['dones'])
        
        if not self.env_data:
            for i in range(n_envs):
                self.env_data[i] = {
                    "dists": [], 
                    "energies": [], 
                    "tracking": [],
                    "rewards": {
                        "survival_reward": [],
                        "g_limit_penalty": [],
                        "action_penalty": [],
                        "offensive_reward": [],
                        "delta_energy_reward": [],
                        "distance_reward": [],
                        "wez_reward": [],
                        "crash_penalty": [],
                        "out_of_bounds_penalty": [],
                        "victory_reward": [],
                        "defeat_penalty": []
                    }
                }

        for i, (info, done) in enumerate(zip(self.locals['infos'], self.locals['dones'])):
            dist = info.get('dist_ft', 0)
            energy = info.get('energy', 0)
            tracking = info.get('tracking_time', 0)

            self.env_data[i]["dists"].append(np.nan_to_num(dist))
            self.env_data[i]["energies"].append(np.nan_to_num(energy))
            self.env_data[i]["tracking"].append(np.nan_to_num(tracking))
            
            # Reward detaylarını topla
            comps = info.get('reward_components', {})
            for key in self.env_data[i]["rewards"].keys():
                val = comps.get(key, 0.0)
                self.env_data[i]["rewards"][key].append(np.nan_to_num(val))

            if done:
                if len(self.env_data[i]["dists"]) > 0:
                    self.logger.record("metrics/avg_distance_ft", np.mean(self.env_data[i]["dists"]))
                    self.logger.record("metrics/avg_energy", np.mean(self.env_data[i]["energies"]))
                    self.logger.record("metrics/avg_tracking_time", np.mean(self.env_data[i]["tracking"]))
                    
                    # Reward section logları
                    for key, values in self.env_data[i]["rewards"].items():
                        if len(values) > 0:
                            self.logger.record(f"rewards/{key}", np.sum(values))
                    
                    # Verileri sıfırla
                    self.env_data[i] = {
                        "dists": [], 
                        "energies": [], 
                        "tracking": [],
                        "rewards": {k: [] for k in self.env_data[i]["rewards"].keys()}
                    }
        return True

custom_policy_kwargs = dict(
    activation_fn=torch.nn.Tanh,
    net_arch=dict(
        pi=[256, 128, 64],
        vf=[256, 256, 128]
    )
)

PHASES = [
    {
        "name": "Phase_1_Basics",
        "timesteps": 40_000_000,
        "reward_weights": {
            "survival_reward": 1.0,
            "crash_penalty": 1.0,
            "out_of_bounds_penalty": 1.0,
            "delta_energy_reward": 0.5,

            "action_penalty": 0.4,
            "g_limit_penalty": 0.0,

            "distance_reward": 0.0,
            "offensive_reward": 0.0,
            "wez_reward": 0.0,
            "victory_reward": 0.0,
            "defeat_penalty": 0.0
        }
    },

    {
        "name": "Phase_2_Approaching_and_Offensive",
        "timesteps": 200_000_000,
        "reward_weights": {
            "survival_reward": 0.3,
            "crash_penalty": 0.3,
            "out_of_bounds_penalty": 0.3,
            "delta_energy_reward": 0.3,
            "action_penalty": 0.3,
            
            "distance_reward": 1.0,
            "offensive_reward": 1.0,
            "g_limit_penalty": 0.0,
            
            "wez_reward": 1.0,
            "victory_reward": 1.0,
            "defeat_penalty": 1.0
        }
    },
    {
        "name": "Phase_3_WEZ_and_Victory",
        "timesteps": 1_000_000_000,
        "reward_weights": {
            "survival_reward": 0.2,
            "crash_penalty": 0.2,
            "out_of_bounds_penalty": 0.2,
            "delta_energy_reward": 0.2,
            "action_penalty": 0.2,
            
            "distance_reward": 0.5,
            "offensive_reward": 0.5,
            "g_limit_penalty": 0.0,
            
            "wez_reward": 5.0,
            "victory_reward": 5.0,
            "defeat_penalty": 0.8,
        }
    }
]

def make_env(reward_weights):
    env = SelfPlayDogfightEnv(render_mode="none", reward_weights=reward_weights)
    env = ss.black_death_v3(env) 
    env = ss.frame_stack_v1(env, stack_size=8)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=32, num_cpus=8, base_class='stable_baselines3')
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)
    return env

def train():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_base = f"./tasks/curriculum_dogfight/tensorboard/{run_id}"

    os.makedirs('./tasks/curriculum_dogfight/models_checkpoints', exist_ok=True)

    run = wandb.init(
        project="curriculum-f16-dogfight",
        sync_tensorboard=True,
        monitor_gym=True,
    )

    model = None
    previous_model_path = None

    for i, phase in enumerate(PHASES):
        print(f"==================================================")
        print(f"STARTING {phase['name']}  (run: {run_id})")
        print(f"Weights: {phase['reward_weights']}")
        print(f"==================================================")

        tb_log = f"{tb_base}/{phase['name']}/"
        env = make_env(phase["reward_weights"])

        # Load previous vec normalize stats if available
        if previous_model_path and os.path.exists(previous_model_path + "_vecnormalize.pkl"):
            print(f"Loading VecNormalize stats from {previous_model_path}_vecnormalize.pkl")
            env = VecNormalize.load(previous_model_path + "_vecnormalize.pkl", env)
            env.training = True

        if model is None:
            if previous_model_path and os.path.exists(previous_model_path + ".zip"):
                print(f"Loading existing model from {previous_model_path}.zip")
                model = PPO.load(previous_model_path, env=env)
                model.tensorboard_log = tb_log
            else:
                model = PPO(
                    "MlpPolicy",
                    env,
                    policy_kwargs=custom_policy_kwargs,
                    verbose=1,
                    learning_rate=1e-5,
                    n_steps=8192,
                    n_epochs=3,
                    ent_coef=0.1,
                    batch_size=16384,
                    gamma=0.99,
                    tensorboard_log=tb_log,
                )
        else:
            model.set_env(env)
            model.tensorboard_log = tb_log

        global_save_freq = 5_000_000
        real_save_freq = max(1, global_save_freq // env.num_envs)

        checkpoint_callback = CheckpointCallback(
            save_freq=real_save_freq,
            save_path=f'./tasks/curriculum_dogfight/models_checkpoints/{phase["name"]}',
            name_prefix='ppo_dogfight'
        )
        metrics_callback = DogfightMetricsCallback()

        model.learn(
            total_timesteps=phase["timesteps"],
            callback=[checkpoint_callback, WandbCallback(), metrics_callback],
            reset_num_timesteps=True,
        )

        previous_model_path = f"tasks/curriculum_dogfight/{phase['name']}_final"
        model.save(previous_model_path)
        env.save(previous_model_path + "_vecnormalize.pkl")

        env.close()

    run.finish()

if __name__ == "__main__":
    train()