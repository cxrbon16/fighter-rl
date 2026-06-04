import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
# from dogfight_env import DogfightParallelEnv  # Ortamınızı içeren dosya
from tasks.navigation.navigation import NavigationTaskEnv
import torch
from typing import Callable
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


def cosine_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Öğrenme hızını (LR) eğitimin başından sonuna kadar kosinüs eğrisiyle düşürür.
    progress_remaining: Eğitimin başında 1.0, sonunda 0.0 olur.
    """
    def func(progress_remaining: float) -> float:
        # progress_remaining 1'den 0'a inerken, (1 - progress_remaining) 0'dan 1'e çıkar.
        # cosinus içerisi 0'dan pi'ye doğru gider.
        # cos(0)=1, cos(pi)=-1 olur. Başına eksi koyup 1 ekleyince [0, 2] arası salınır, 2'ye bölünce [0, 1] olur.
        cos_out = 0.5 * (1.0 + np.cos(np.pi * (1.0 - progress_remaining)))
        return cos_out * initial_value
        
    return func


def train():
    # 1. Ortamı yarat (Eğitim sırasında render kapalı olmalı)
    env = NavigationTaskEnv(render_mode="none")
    
    # 2. PettingZoo ortamını SB3'ün anlayacağı VecEnv formatına çevir
    # Bu wrapper, ParallelEnv içindeki ajanları düzleştirerek tek bir politikanın
    # (parameter sharing) her iki ajanı da kontrol etmesini sağlar.
    env = ss.black_death_v3(env) 

    env = ss.frame_stack_v1(env, stack_size=4)
    
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    
    # JSBSim'in RAM ve CPU kullanımı yoğun olduğu için num_vec_envs=1 tutuyoruz.
    # Eğer sisteminiz güçlüyse num_vec_envs değerini artırıp eğitimi hızlandırabilirsiniz.
    env = ss.concat_vec_envs_v1(env, num_vec_envs=128, num_cpus=8, base_class='stable_baselines3')
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)


    run = wandb.init(
        project="f15-dogfight", # W&B sitesinde görünecek proje adı
        sync_tensorboard=True,  # İşte sihir bu! SB3'ün ürettiği tüm TB loglarını internete pushlar
        monitor_gym=True,       # İstersen ileride ortamın videosunu da pushlar
    )

    # Global frekansı, toplam paralel ajan sayısına bölüyoruz
    global_save_freq = 250_000
    real_save_freq = max(1, global_save_freq // env.num_envs)

    # Modelin düzenli kaydedilmesi için callback
    checkpoint_callback = CheckpointCallback(
        save_freq=real_save_freq, 
        save_path='./tasks/navigation/models_checkpoints',
        name_prefix='ppo_navigation'
    )

    # 3. PPO Modelini Tanımla
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=custom_policy_kwargs,
        verbose=0,
        learning_rate=1e-5,
        n_steps=2048,           # JSBSim sürekli bir simülasyon olduğu için yüksek n_steps iyidir
        n_epochs=8,
        ent_coef=0.08,
        batch_size=16384,
        gamma=0.99,             # Gelecekteki ödüllerin önem katsayısı
        tensorboard_log="./tasks/navigation/dogfight_tensorboard/"
    )

    print("Eğitim başlıyor... Tensorboard ile izlemek için yeni terminalde şu komutu çalıştırın:")
    print("tensorboard --logdir ./dogfight_tensorboard/")
    
    # 4. Modeli Eğit
    model.learn(total_timesteps=100_000_000, callback=[checkpoint_callback, WandbCallback()])

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