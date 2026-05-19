import os
import supersuit as ss
from stable_baselines3 import PPO
import numpy as np
from dogfight_env import DogfightParallelEnv  # Ortamınızı içeren dosya

def test():
    print("Eğitilmiş model test ediliyor...")
    
    # Test için render modunu açıyoruz
    env = DogfightParallelEnv(render_mode="panda")
    
    # 2. 🔥 EĞİTİMDEKİ WRAPPER ZİNCİRİNİ BURADA DA AYNEN KURUYORUZ
    env = ss.black_death_v3(env) 
    
    # Boyut hatasını bitiren asıl sihir: Gözlemi 8'den 32'ye çıkaran hafıza katmanı
    env = ss.frame_stack_v1(env, stack_size=4)
    
    # Çoklu ajan yapısını SB3'ün anlayacağı tek bir vektör haline getiriyoruz (Boyut: 64 veya 128 olur)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    
    # Test esnasında izleme wrapper'ı (isteğe bağlı ama uyum için iyi olur)

    # ... (Yukarıdaki env tanımlamaları ve model.load kısımları AYNEN KALDI) ...
    
    # Modeli yükledik
    model = PPO.load("ppo_dogfight_final")
    
        
    # 1. Ortamı sıfırla
    raw_obs = env.reset()
    
    # 🔄 EMNİYET KEMERİ 1: SuperSuit'in döndürdüğü o tuple/dict karmaşasından 
    # saf NumPy matrisini ayıklıyoruz. Genelde ilk eleman ana matristir.
    if isinstance(raw_obs, tuple):
        obs = raw_obs[0]
    else:
        obs = raw_obs

    print("Simülasyon başladı! İzlemek için FlightGear'a bakın...")
    
    for step in range(10000):
        # Modelden aksiyonları al
        action, _states = model.predict(obs, deterministic=True)
        
        # 🔄 REÇETE: 4 yerine tam 5 değer açıyoruz! (terminated ve truncated geldi)
        raw_obs, rewards, terminations, truncations, infos = env.step(action)
        
        # Emniyet kemerimiz: Gelen ham gözlem tuple ise ilk elemanı (saf matrisi) al
        if isinstance(raw_obs, tuple):
            obs = raw_obs[0]
        else:
            obs = raw_obs
        
        # --- TELEMETRİ PRINT BÖLÜMÜ ---
        if step % 5 == 0:
            print(f"✈️ Adım: {step:04d} "
                  f"| A1 Ödül: {rewards[0]:.2f} "
                  f"| A2 Ödül: {rewards[1]:.2f} "
                  f"| A1 Aksiyon (Ail/Elev/Thro): [{action[0][0]:.2f}, {action[0][1]:.2f}, {action[0][2]:.2f}]")
        
        # 🔄 YENİ KONTROL: modern sistemde bir uçağın sıfırlanması için 
        # ya elenmesi (termination) ya da süresinin bitmesi (truncation) gerekir.
        if terminations[0] or truncations[0]:
            print("💥 Agent 1 elendi veya süre bitti! Yeniden doğuyor...")
        if terminations[1] or truncations[1]:
            print("💥 Agent 2 elendi veya süre bitti! Yeniden doğuyor...")

if __name__ == "__main__":
    test()