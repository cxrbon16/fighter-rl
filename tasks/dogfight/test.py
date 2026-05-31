import os
import time
import supersuit as ss
from stable_baselines3 import PPO
import numpy as np
import rerun as rr
from tasks.dogfight.dogfight import SelfPlayDogfightEnv 

def test_dogfight():
    print("Eğitilmiş Dogfight modeli test ediliyor...")
    
    # 1. Ortamı Başlat
    base_env = SelfPlayDogfightEnv(render_mode="debug")
    
    # Çoklu ajan wrapper'ları (Önceki yapınızla uyumlu)
    env = ss.black_death_v3(base_env) 
    env = ss.frame_stack_v1(env, stack_size=4) 
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class='stable_baselines3')    
    
    # Eğittiğiniz modelin yolunu buraya girin (eğer yoksa rastgele hareket edecektir)
    model_path = "models/ppo_dogfight_model.zip" 
    try:
        model = PPO.load(model_path)
        print("Model başarıyla yüklendi!")
    except:
        print("Model bulunamadı, uçaklar rastgele hareket edecek (Dummy Test).")
        model = None
        
    # 2. Rerun Başlatma
    rr.init("F16_Dogfight_Arena", spawn=True)    
    
    obs = env.reset()
    
    # Kuyruk izlerini tutmak için sözlük (İki ajan için ayrı ayrı)
    trajectories = {
        'agent_0': [],
        'agent_1': []
    }
    
    # Uçakların ilk başladığı noktayı orijin (0,0) kabul etmek için değişken
    # İt dalaşında merkez nokta sabit kalsın ki arena kaymasın
    origin_lat = None
    origin_lon = None
    
    print("Simülasyon başladı! İzlemek için Rerun motoruna geçin...")
    
    for step in range(10000):
        if model:
            action, _states = model.predict(obs, deterministic=True)
        else:
            # Model yoksa rastgele aksiyon üret (Test amaçlı)
            action = np.array([env.action_space.sample() for _ in range(env.num_envs)])
            
        obs, rewards, dones, infos = env.step(action)
        
        # PettingZoo'da dones bir array döner (ajan sayısı kadar)
        # Eğer herhangi biri bittiyse raund bitmiştir.
        if any(dones):
            print(f"💥 Raund bitti! Adım: {step} | Ortam sıfırlanıyor...")
            trajectories['agent_0'].clear()
            trajectories['agent_1'].clear()
            obs = env.reset()
            origin_lat = None # Orijini sıfırla
            continue

        # 3. Rerun Çizimi ve Loglama
        # `infos` içinden iki ajanın da verilerini çekmeliyiz.
        # Not: SelfPlayDogfightEnv'de step fonksiyonunun info döndürdüğünden emin ol.
        # Şimdilik BaseEnv'in fdm'ine doğrudan erişerek çiziyoruz (daha pratik).
        

        info_0, info_1 = infos[0], infos[1]

        # Sadece ilk adımda orijini ayarla
        if origin_lat is None:
            origin_lat = info_0['lat']
            origin_lon = info_0['lon']

        lat_to_m = 111320.0
        lon_to_m = 111320.0 * np.cos(np.radians(origin_lat))

        # --- AJAN 0 (MAVİ TAKIM) ---
        lat_0 = info_0['lat']
        lon_0 = info_0['lon']
        alt_0_m = info_0['alt_m']
        
        x_0 = (lon_0 - origin_lon) * lon_to_m
        y_0 = (lat_0 - origin_lat) * lat_to_m
        z_0 = alt_0_m
        
        rr.log("radar/agent_0/ucak", rr.Points3D([x_0, y_0, z_0], colors=[0, 100, 255], radii=150.0))
        trajectories['agent_0'].append([x_0, y_0, z_0])
        if len(trajectories['agent_0']) > 2:
            rr.log("radar/agent_0/rota", rr.LineStrips3D([trajectories['agent_0']], colors=[100, 150, 255]))

        # --- AJAN 1 (KIRMIZI TAKIM) ---
        lat_1 = info_1['lat']
        lon_1 = info_1['lon']
        alt_1_m = info_1['alt_m'] 
        
        x_1 = (lon_1 - origin_lon) * lon_to_m
        y_1 = (lat_1 - origin_lat) * lat_to_m
        z_1 = alt_1_m
        
        rr.log("radar/agent_1/ucak", rr.Points3D([x_1, y_1, z_1], colors=[255, 50, 50], radii=150.0))
        trajectories['agent_1'].append([x_1, y_1, z_1])
        if len(trajectories['agent_1']) > 2:
            rr.log("radar/agent_1/rota", rr.LineStrips3D([trajectories['agent_1']], colors=[255, 100, 100]))

        # 4. Telemetri ve Analiz (Her 5 adımda bir terminale bas)
        if step % 5 == 0:
            mesafe_m = np.sqrt((x_0 - x_1)**2 + (y_0 - y_1)**2 + (z_0 - z_1)**2)
            
            hiz_0_kts = info_0['airspeed_kts']
            hiz_1_kts = info_1['airspeed_kts']
            
            # Sadece ajan 0'ın ödülünü yazdırıyoruz
            odul_0 = float(rewards[0])
            
            log_metni = (f"⚔️ Adım: {step:04d} | Mesafe: {mesafe_m:05.0f}m | "
                         f"A0: {hiz_0_kts:03.0f}kts, {alt_0_m:05.0f}m | "
                         f"A1: {hiz_1_kts:03.0f}kts, {alt_1_m:05.0f}m | Ödül: {odul_0:+05.2f}")
            print(log_metni)

            rr.log("telemetri/metin", rr.TextLog(log_metni, level=rr.TextLogLevel.INFO))            
            
            # Rerun Grafikleri
            rr.log("telemetri/dogfight/aradaki_mesafe_m", rr.Scalars(mesafe_m))
            rr.log("telemetri/dogfight/irtifa_A0", rr.Scalars(alt_0_m))
            rr.log("telemetri/dogfight/irtifa_A1", rr.Scalars(alt_1_m))
            rr.log("telemetri/dogfight/odul_A0", rr.Scalars(odul_0))
            
        time.sleep(4.0 / 120.0)

if __name__ == "__main__":
    test_dogfight()