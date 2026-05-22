import os
import time
import supersuit as ss
from stable_baselines3 import PPO
import numpy as np
from tasks.navigation.navigation import NavigationTaskEnv

def test():
    print("Eğitilmiş model test ediliyor...")
    
    env = NavigationTaskEnv(render_mode="human")
        
    env = ss.black_death_v3(env) 
    env = ss.frame_stack_v1(env, stack_size=4) # 14 obs -> 56'ya çıkıyor
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    
    # 🔥 İŞTE EKSİK OLAN VE HATAYI ÇÖZECEK SATIR BURASI 🔥
    env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class='stable_baselines3')    
    model_path = "/home/ayganyavuz/Desktop/dogfighting_rl/tasks/navigation/ppo_dogfight_final.zip" 
    model = PPO.load(model_path)
        
    obs = env.reset()

    print("Simülasyon başladı! İzlemek için motora geçin...")
    
    for step in range(10000):
        action, _states = model.predict(obs, deterministic=True)
        
        obs, rewards, dones, infos = env.step(action)
        
        if step % 5 == 0:
            latest_frame = obs[0].flatten()[-14:]
            
            irtifa_ft = latest_frame[0] * 30000.0
            hiz_kts = latest_frame[1] * 1000.0
            mesafe_ft = latest_frame[12] * 50000.0
            sapma_radyan = latest_frame[13] * np.pi
            
            # 🔥 METRİK SİSTEME ÇEVİRİM 🔥
            irtifa_m = irtifa_ft * 0.3048
            hiz_ms = hiz_kts * 0.514444
            mesafe_m = mesafe_ft * 0.3048
            sapma_derece = sapma_radyan * (180.0 / np.pi)
            
            print(f"✈️ Adım: {step:04d} "
                  f"| Hız: {hiz_ms:03.0f} m/s "
                  f"| İrtifa: {irtifa_m:05.0f} m "
                  f"| Hedefe: {mesafe_m:05.0f} m "
                  f"| Açı Sapması: {sapma_derece:+04.0f}° "
                  f"| Ödül: {rewards[0]:+06.2f} "
                  f"| Aksiyon (A/E/T): [{action[0][0]:+0.2f}, {action[0][1]:+0.2f}, {action[0][2]:+0.2f}]")
        
        if dones[0]:
            print("💥 Raund bitti (Elendi, Hedefe Vardı veya Süre Doldu)! Yeniden doğuyor...")
            break  
            
        time.sleep(4.0 / 120.0)

if __name__ == "__main__":
    test()