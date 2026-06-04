from pathlib import Path
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
from tasks.navigation.navigation import NavigationTaskEnv
import torch
from typing import Callable
import numpy as np
import wandb
from wandb.integration.sb3 import WandbCallback

REPO_ROOT = Path(__file__).resolve().parents[2]

custom_policy_kwargs = dict(
    activation_fn=torch.nn.Tanh,  
    net_arch=dict(
        pi=[128, 128],    # Aktör (Policy) Ağı: Hangi hamleyi yapacağına karar veren ağ
        vf=[128, 128]     # Kritik (Value) Ağı: Bulunduğu durumun ne kadar iyi olduğunu tahmin eden ağ
    )
)


def cosine_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Öğrenme hızını (LR) eğitimin başından sonuna kadar kosinüs eğrisiyle düşürür.
    progress_remaining: Eğitimin başında 1.0, sonunda 0.0 olur.
    """
    def func(progress_remaining: float) -> float:
        cos_out = 0.5 * (1.0 + np.cos(np.pi * (1.0 - progress_remaining)))
        return cos_out * initial_value
        
    return func


def train():
    env = NavigationTaskEnv(render_mode="none")
    
    env = ss.black_death_v3(env) 

    env = ss.frame_stack_v1(env, stack_size=4)
    
    env = ss.pettingzoo_env_to_vec_env_v1(env)

    env = ss.concat_vec_envs_v1(env, num_vec_envs=128, num_cpus=8, base_class='stable_baselines3')
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)


    run = wandb.init(
        project="f15-dogfight", # W&B sitesinde görünecek proje adı
        sync_tensorboard=True,  # İşte sihir bu! SB3'ün ürettiği tüm TB loglarını internete pushlar
        monitor_gym=True,       # İstersen ileride ortamın videosunu da pushlar
    )

    global_save_freq = 250_000
    real_save_freq = max(1, global_save_freq // env.num_envs)

    checkpoint_callback = CheckpointCallback(
        save_freq=real_save_freq, 
        save_path='./tasks/navigation/models_checkpoints',
        name_prefix='ppo_navigation'
    )

    checkpoint_path = REPO_ROOT / "tasks/navigation/models_checkpoints/ppo_navigation_99942400_steps.zip"
    
    if checkpoint_path is not None:
        print(f"Önceki eğitimden devam ediliyor: {checkpoint_path}")
        # Modeli yüklerken 'env' parametresini vermek ZORUNLUDUR!
        # custom_objects: Eğer özel bir learning rate (cosine_schedule) kullanıyorsan 
        # SB3'ün hata vermemesi için bunu da geçmen faydalı olabilir.
        model = PPO.load(
            checkpoint_path, 
            env=env, 
            tensorboard_log="./tasks/navigation/dogfight_tensorboard/",
            custom_objects={"lr_schedule": cosine_schedule(1e-5)} # Özel LR fonksiyonunu tekrar tanıtıyoruz
        )
    else:
        print("Sıfırdan yeni bir model eğitiliyor...")
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=custom_policy_kwargs,
            verbose=0,
            learning_rate=cosine_schedule(1e-5), # Öğrenme hızını burada kullanıyoruz
            n_steps=2048,           
            n_epochs=8,
            ent_coef=0.08,
            batch_size=16384,
            gamma=0.99,             
            tensorboard_log="./tasks/navigation/dogfight_tensorboard/"
        )

    # W&B ve Checkpoint Callback'leri aynı şekilde devam eder
    print("Eğitim başlıyor...")
    model.learn(total_timesteps=100_000_000, callback=[checkpoint_callback, WandbCallback()], reset_num_timesteps=False)

    # 5. Eğitilmiş Modeli Kaydet
    model.save("tasks/navigation/ppo_dogfight_final")
    run.finish() 
    print("Eğitim tamamlandı")


def test():
    print("Eğitilmiş model test ediliyor...")
    
    env = NavigationTaskEnv(render_mode="human")
    
    env = ss.black_death_v3(env) 
    env = ss.frame_stack_v1(env, stack_size=4)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    
    env = ss.concat_vec_envs_v1(
        env, 
        num_vec_envs=1, 
        num_cpus=1, 
        base_class='stable_baselines3'
    )
    
    model = PPO.load("tasks/navigation/ppo_dogfight_final")
    
    obs = env.reset()

    
    for step in range(10000):
        # Artık obs %100 homojen bir NumPy matrisi, predict anında çalışacak
        action, _states = model.predict(obs, deterministic=True)
        
        # SB3 VecEnv kullanınca step() tam 4 değer döndürür
        # (terminated ve truncated arka planda birleşip 'dones' olur)
        obs, rewards, dones, infos = env.step(action)
        
        # --- TELEMETRİ PRINT BÖLÜMÜ ---
        if step % 5 == 0:
            print(f"Adım: {step:04d} "
                  f"| A1 Ödül: {rewards[0]:.2f} "
                  f"| A2 Ödül: {rewards[1]:.2f} "
                  f"| A1 Aksiyon (Ail/Elev/Thro): [{action[0][0]:.2f}, {action[0][1]:.2f}, {action[0][2]:.2f}]")
        
        # dones dizisi, 1. veya 2. ajanın (index 0 veya 1) o adımda ölüp ölmediğini söyler
        if dones[0]:
            print("Agent 1 elendi veya süre bitti! Yeniden doğuyor...")
        if dones[1]:
            print("Agent 2 elendi veya süre bitti! Yeniden doğuyor...")

if __name__ == "__main__":
    # Önce eğitimi çalıştır
    train()
    
    # Eğitim bittikten sonra sonuçları görsel olarak gör
    test()