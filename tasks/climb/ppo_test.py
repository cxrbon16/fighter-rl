import os
import time
import supersuit as ss
from stable_baselines3 import PPO
import numpy as np
from dogfight_env import DogfightParallelEnv  # Dosya adın neyse ona göre değiştir

def test():
    print("Eğitilmiş model test ediliyor...")
    
    # Test için render modunu açıyoruz (FlightGear veya Panda3D)
    env = DogfightParallelEnv(render_mode="human")
    
    # 🔥 EĞİTİMDEKİ WRAPPER ZİNCİRİNİ BİREBİR KURUYORUZ
    env = ss.black_death_v3(env) 
    env = ss.frame_stack_v1(env, stack_size=4) # 12 obs -> 48'e çıkıyor
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    
    # 🔥 DİKKAT: Eğittiğin modelin doğru yolunu (path) buraya gir!
    model_path = "/home/ayganyavuz/Desktop/dogfighting_rl/playground/models/dogfight_ppo/ppo_dogfight_38247552_steps" 
    model = PPO.load(model_path)
        
    # 1. Ortamı sıfırla
    raw_obs = env.reset()
    
    # Emniyet kemerimiz: SuperSuit'in tuple'ından saf matrisi çıkar
    if isinstance(raw_obs, tuple):
        obs = raw_obs[0]
    else:
        obs = raw_obs

    print("Simülasyon başladı! İzlemek için motora geçin...")
    
    for step in range(10000):
        # Modelden aksiyonları al (deterministic=True ajanın saçmalamasını engeller)
        action, _states = model.predict(obs, deterministic=True)
        
        # 5 değer dönüyoruz (terminated ve truncated dahil)
        raw_obs, rewards, terminations, truncations, infos = env.step(action)
        
        if isinstance(raw_obs, tuple):
            obs = raw_obs[0]
        else:
            obs = raw_obs
        
        # --- HUD: TELEMETRİ PRINT BÖLÜMÜ ---
        if step % 5 == 0:
            # Frame Stack yapıldığı için obs[0]'ı düzleştirip son 12 verinin (en taze karenin) ilk elemanını alıyoruz
            irtifa_ft = obs[0].flatten()[-12] * 30000.0
            
            print(f"✈️ Adım: {step:04d} "
                  f"| İrtifa: {irtifa_ft:05.0f} ft "
                  f"| A1 Ödül: {rewards[0]:+06.2f} "
                  f"| A1 Aksiyon (Ail/Elev/Thro): [{action[0][0]:+0.2f}, {action[0][1]:+0.2f}, {action[0][2]:+0.2f}]")
        
        # Yeniden doğma kontrolü
        if terminations[0] or truncations[0]:
            print("💥 Raund bitti (Elendi veya Süre Doldu)! Yeniden doğuyor...")
            
        # ⏱️ GERÇEK ZAMAN FRENİ (1x HIZ İÇİN)
        # Fizik motorunda 8 frame (0.066s) atladığımız için, gerçek hayatta da
        # o kadar süre bekletiyoruz ki uçak Matrix gibi x20 hızda akmasın.
        time.sleep(8.0 / 120.0)

if __name__ == "__main__":
    test()