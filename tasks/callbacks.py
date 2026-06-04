import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class DogfightMetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
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
                        "defeat_penalty": [],
                    }
                }

        for i, (info, done) in enumerate(zip(self.locals['infos'], self.locals['dones'])):
            self.env_data[i]["dists"].append(np.nan_to_num(info.get('dist_ft', 0)))
            self.env_data[i]["energies"].append(np.nan_to_num(info.get('energy', 0)))
            self.env_data[i]["tracking"].append(np.nan_to_num(info.get('tracking_time', 0)))

            comps = info.get('reward_components', {})
            for key in self.env_data[i]["rewards"]:
                self.env_data[i]["rewards"][key].append(np.nan_to_num(comps.get(key, 0.0)))

            if done and len(self.env_data[i]["dists"]) > 0:
                self.logger.record("metrics/avg_distance_ft", np.mean(self.env_data[i]["dists"]))
                self.logger.record("metrics/avg_energy", np.mean(self.env_data[i]["energies"]))
                self.logger.record("metrics/avg_tracking_time", np.mean(self.env_data[i]["tracking"]))
                for key, values in self.env_data[i]["rewards"].items():
                    if values:
                        self.logger.record(f"rewards/{key}", np.sum(values))
                self.env_data[i] = {
                    "dists": [],
                    "energies": [],
                    "tracking": [],
                    "rewards": {k: [] for k in self.env_data[i]["rewards"]},
                }

        return True
