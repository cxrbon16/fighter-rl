import os
import time
import supersuit as ss
from stable_baselines3 import PPO
import numpy as np
from tasks.navigation.navigation import NavigationTaskEnv
import rerun as rr


def test():
    print("Eğitilmiş model test ediliyor...")
    
    base_env = NavigationTaskEnv(render_mode="debug")
        
    env = ss.black_death_v3(base_env) 
    env = ss.frame_stack_v1(env, stack_size=4) 
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    
    env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class='stable_baselines3')    
    model_path = "/home/ayganyavuz/Desktop/dogfighting_rl/tasks/navigation/models_checkpoints/ppo_navigation_99942400_steps.zip" 
    model = PPO.load(model_path)
        
    obs = env.reset()
    trajectory = []
    rr.init("F15_Dogfight_Radar", spawn=True)    
    
    print("Simülasyon başladı! İzlemek için motora geçin...")
    
    # Uçağın ilk başladığı noktayı orijin (0,0) kabul etmek için değişkenler
    start_lat = None
    start_lon = None
    
    for step in range(10000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        
        env_info = infos[0]

        if not env_info:
            if dones[0]:
                print("💥 Raund bitti! Ortam sıfırlanıyor...")
                trajectory.clear()
                obs = env.reset()
                start_lat = None # Raund bitince orijini sıfırla
            continue

        # INFO'dan verileri çek
        current_lat = env_info['lat']
        current_lon = env_info['lon']
        current_alt_m = env_info['alt_m']

        target_lat = env_info['target_lat']
        target_lon = env_info['target_lon']
        target_alt_m = env_info['target_alt_m']
        
        start_lat = target_lat
        start_lon = target_lon
            
        # 1 Derece Enlem ≈ 111.32 Kilometredir
        lat_to_m = 111320.0
        lon_to_m = 111320.0 * np.cos(np.radians(start_lat))
        
        # X, Y koordinatlarını başlangıç noktasına göre hesapla
        x = (current_lon - start_lon) * lon_to_m
        y = (current_lat - start_lat) * lat_to_m
        z = current_alt_m

        target_x, target_y, target_z = 0.0, 0.0, target_alt_m
        
        # Uçağı Çiz (Hedef Çizimini Kaldırdık)
        rr.log("radar/f15", rr.Points3D([x, y, z], colors=[255, 0, 0], radii=150.0))
        rr.log("radar/hedef", rr.Points3D([target_x, target_y, target_z], colors=[0, 255, 0], radii=150.0))
        
        # Uçağın Kuyruk İzini Çiz
        trajectory.append([x, y, z])
        if len(trajectory) > 2:
            rr.log("radar/f15_rota", rr.LineStrips3D([trajectory], colors=[255, 100, 100]))

        # Telemetri Ekranı (Değişiklik yok)
        if step % 5 == 0:
            # 🔥 AGA BURASI: Artık obs'yi çarpmak yok, doğrudan saf JSBSim verisini (INFO) okuyoruz!
            irtifa_m = env_info['alt_m']
            hiz_ms = env_info['airspeed_ms'] 
            
            # Hedefle olan gerçek 3D metrik mesafeyi Rerun koordinatlarından (Pisagor ile) buluyoruz:
            #             
            # Sadece hedef sapma açısını obs'den çekmeye devam edebiliriz 
            # (Çünkü onu zaten -pi ile +pi arasında çok temiz hesaplamıştın)
            latest_frame = obs[0].flatten()[-15:] 
            sapma_radyan = latest_frame[13] * np.pi
            sapma_derece = sapma_radyan * (180.0 / np.pi)

            mesafe_m = np.sqrt(x**2 + y**2 + (current_alt_m - target_alt_m)**2)
            
            # Aksiyonları daha temiz yazdıralım
            aileron = action[0][0]
            elevator = action[0][1]
            throttle = action[0][2]
            
            print(f"✈️ Adım: {step:04d} | Hız: {hiz_ms:03.0f} m/s | İrtifa: {irtifa_m:05.0f} m | Hedefe: {mesafe_m:05.0f} m | Açı: {sapma_derece:+04.0f}° | Ödül: {rewards[0]:+06.2f} | Aks: [{aileron:+0.2f}, {elevator:+0.2f}, {throttle:+0.2f}]")            
        
        if dones[0]:
            print("💥 Raund bitti (Elendi, Hedefe Vardı veya Süre Doldu)! Yeniden doğuyor...")
            trajectory.clear()
            obs = env.reset()
            start_lat = None # Raund sıfırlandığında orijini de sıfırla
            continue
            
        time.sleep(4.0 / 120.0)

if __name__ == "__main__":
    test()