import os
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor
from dogfight_env import DogfightParallelEnv  # Ortamınızı içeren dosya
import torch

custom_policy_kwargs = dict(
    activation_fn=torch.nn.Tanh,  # Aktivasyon fonksiyonu (Tanh veya ReLU kullanılabilir)
    net_arch=dict(
        pi=[64, 64],    # Aktör (Policy) Ağı: Hangi hamleyi yapacağına karar veren ağ
        vf=[64, 64]     # Kritik (Value) Ağı: Bulunduğu durumun ne kadar iyi olduğunu tahmin eden ağ
    )
)

def train():
    # 1. Ortamı yarat (Eğitim sırasında render kapalı olmalı)
    env = DogfightParallelEnv(render_mode="none")
    
    # 2. PettingZoo ortamını SB3'ün anlayacağı VecEnv formatına çevir
    # Bu wrapper, ParallelEnv içindeki ajanları düzleştirerek tek bir politikanın
    # (parameter sharing) her iki ajanı da kontrol etmesini sağlar.
    env = ss.black_death_v3(env) 
    
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    
    # JSBSim'in RAM ve CPU kullanımı yoğun olduğu için num_vec_envs=1 tutuyoruz.
    # Eğer sisteminiz güçlüyse num_vec_envs değerini artırıp eğitimi hızlandırabilirsiniz.
    env = ss.concat_vec_envs_v1(env, num_vec_envs=128, num_cpus=1, base_class='stable_baselines3')
    env = VecMonitor(env)

    # Modelin düzenli kaydedilmesi için callback (Her 10.000 adımda bir)
    checkpoint_callback = CheckpointCallback(
        save_freq=250_000, 
        save_path='./models/dogfight_ppo/',
        name_prefix='ppo_dogfight'
    )

    # 3. PPO Modelini Tanımla
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=custom_policy_kwargs,
        verbose=1,
        learning_rate=3e-5,
        n_steps=2048,           # JSBSim sürekli bir simülasyon olduğu için yüksek n_steps iyidir
        n_epochs=4,
        ent_coef=0.01,
        batch_size=8192,
        gamma=0.99,             # Gelecekteki ödüllerin önem katsayısı
        tensorboard_log="./dogfight_tensorboard/"
    )

    print("Eğitim başlıyor... Tensorboard ile izlemek için yeni terminalde şu komutu çalıştırın:")
    print("tensorboard --logdir ./dogfight_tensorboard/")
    
    # 4. Modeli Eğit
    # İt dalaşı zor bir problemdir, anlamlı sonuçlar için bu sayıyı 1_000_000'a çıkarmanız gerekebilir.
    model.learn(total_timesteps=10_000_000, callback=checkpoint_callback)

    # 5. Eğitilmiş Modeli Kaydet
    model.save("ppo_dogfight_final")
    print("Eğitim tamamlandı ve model kaydedildi: ppo_dogfight_final.zip")


def test():
    print("Eğitilmiş model test ediliyor...")
    
    # Test için render modunu açıyoruz
    env = DogfightParallelEnv(render_mode="human")
    
    # Kaydedilmiş modeli yüklüyoruz
    model = PPO.load("ppo_dogfight_final")
    
    observations, infos = env.reset()
    
    # Ortamda ajan kaldığı sürece döngüye devam et
    while env.agents:
        actions = {}
        
        # Her bir ajan için modelden ayrı ayrı aksiyon tahmini al
        for agent in env.agents:
            # Deterministic=True modeli rastgelelikten çıkarıp öğrendiği en iyi hamleyi yapmaya zorlar
            action, _states = model.predict(observations[agent], deterministic=True)
            actions[agent] = action
            
        # Ajanların aksiyonlarını ortama gönder ve yeni durumları al
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # Eğer ajanlardan biri elendiyse (terminations/truncations True olduysa)
        # PettingZoo kuralı gereği ajanları güncelliyoruz (bu sizin env.step içindeki self.agents = [] mantığınızla çalışır)
        if not env.agents:
            print("Simülasyon bitti!")
            break

if __name__ == "__main__":
    # Önce eğitimi çalıştır
    train()
    
    # Eğitim bittikten sonra sonuçları görsel olarak gör
    test()