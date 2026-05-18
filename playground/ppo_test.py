import os
import supersuit as ss
from stable_baselines3 import PPO
import numpy as np
from dogfight_env import DogfightParallelEnv  # Ortamınızı içeren dosya

def test():
    print("Eğitilmiş model test ediliyor...")
    
    # Test için render modunu açıyoruz
    env = DogfightParallelEnv(render_mode="human")
    
    # Kaydedilmiş modeli yüklüyoruz
    model = PPO.load("ppo_dogfight_final")
    
    observations, infos = env.reset()
    
    step_count = 0
    
    # Ortamda ajan kaldığı sürece döngüye devam et
    while env.agents:
        actions = {}
        
        # Her bir ajan için modelden ayrı ayrı aksiyon tahmini al
        for agent in env.agents:
            obs = observations[agent]
            action, _states = model.predict(obs, deterministic=True)
            actions[agent] = action
            
            # EKRANI TEMİZ TUTMAK İÇİN SADECE AGENT_1'İ LOGLAYALIM
            if agent == "agent_1":
                # Gözlemleri (Observations) Ayrıştır
                alt = obs[0]
                vc = obs[1]
                roll_deg = np.degrees(obs[2])   # Radyanı dereceye çevir
                pitch_deg = np.degrees(obs[3])  # Radyanı dereceye çevir
                
                # Modelin Kararlarını (Actions) Ayrıştır
                aileron = action[0]   # -1.0 (Tam Sol) ile 1.0 (Tam Sağ)
                elevator = action[1]  # -1.0 (Aşağı bas) ile 1.0 (Kendine çek)
                
                # PPO -1 ile 1 arası değer üretir, ortam bunu 0-1 arası throttle'a çevirir
                throttle_gercek = (action[2] + 1.0) / 2.0 
                
                # Terminalde yan yana ve üst üste yazması için (Spam yapmaması için \r kullanıyoruz)
                log_text = (
                    f"✈️ ALT: {alt:5.0f}ft | HIZ: {vc:3.0f}kts | "
                    f"PITCH: {pitch_deg:5.1f}° | ROLL: {roll_deg:5.1f}°  ||  "
                    f"🕹️ LÖVYE -> Ail: {aileron:5.2f} | Elev: {elevator:5.2f} | Thr: %{throttle_gercek*100:3.0f}   "
                )
                print(log_text, end='\r')
            
        # Ajanların aksiyonlarını ortama gönder ve yeni durumları al
        observations, rewards, terminations, truncations, infos = env.step(actions)
        step_count += 1
        
        if not env.agents:
            print("\nSimülasyon bitti! Toplam Adım:", step_count)
            break

if __name__ == "__main__":
    test()