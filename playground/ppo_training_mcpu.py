import supersuit as ss
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor

from dogfight_env import DogfightParallelEnv


custom_policy_kwargs = dict(
    activation_fn=torch.nn.Tanh,
    net_arch=dict(
        pi=[64, 64],
        vf=[64, 64]
    )
)


def linear_schedule(initial_value):
    def func(progress_remaining):
        return progress_remaining * initial_value
    return func


def train():

    env = DogfightParallelEnv(render_mode="none")

    env = ss.black_death_v3(env)

    env = ss.pettingzoo_env_to_vec_env_v1(env)

    # IMPORTANT:
    # num_cpus=1 avoids the buggy multiprocessing constructor
    env = ss.concat_vec_envs_v1(
        env,
        num_vec_envs=128,
        num_cpus=8,
        base_class="stable_baselines3"
    )

    env = VecMonitor(env)

    checkpoint_callback = CheckpointCallback(
        save_freq=250_000,
        save_path="./models/dogfight_ppo/",
        name_prefix="ppo_dogfight"
    )

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=custom_policy_kwargs,
        verbose=1,

        learning_rate=linear_schedule(3e-5),

        n_steps=2048,
        batch_size=8192,
        n_epochs=4,

        gamma=0.99,
        ent_coef=0.01,

        tensorboard_log="./dogfight_tensorboard/"
    )

    print("Training started...")
    print("tensorboard --logdir ./dogfight_tensorboard/")

    model.learn(
        total_timesteps=10_000_000,
        callback=checkpoint_callback
    )

    model.save("ppo_dogfight_final")

    print("Training completed.")


if __name__ == "__main__":
    train()