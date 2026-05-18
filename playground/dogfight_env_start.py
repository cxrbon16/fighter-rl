import numpy as np
from dogfight_env import DogfightParallelEnv

def run_pettingzoo_env(render_mode="none", total_steps=500):
    """
    PettingZoo Çoklu Ajan Ortamını başlatan ve rastgele aksiyonlarla koşturan fonksiyon.
    
    render_mode="none"  -> FlightGear kapalı, işlemci hızında maksimum sürat (Eğitim modu)
    render_mode="human" -> FlightGear açık, gerçek zamanlı izleme modu
    """
    print(f"\n=== Ortam Başlatılıyor | Mod: {render_mode.upper()} ===")
    
    # 1. Ortamı Örnekle (Instantiate)
    env = DogfightParallelEnv(render_mode=render_mode)
    
    # 2. Ortamı Sıfırla (Reset)
    # İlk gözlemler (observations) ve bilgiler (infos) sözlük (dict) olarak döner
    observations, infos = env.reset()
    
    step_count = 0
    
    # 3. Ana Simülasyon Döngüsü
    # env.agents listesi boşalana kadar (bölüm bitene kadar) döner
    while env.agents and step_count < total_steps:
        step_count += 1
        actions = {}
        
        # Her bir aktif ajan için sırayla aksiyon üret (AI yerine şimdilik rastgele)
        for agent_id in env.agents:
            # action_spaces[agent_id].sample() bize [-1.0, 1.0] arasında 3 adet rastgele sayı üretir
            actions[agent_id] = env.action_spaces[agent_id].sample()
        
        # Üretilen aksiyonları ortama sözlük olarak gönder
        # PettingZoo Parallel API gereği tüm ajanlar adımlarını eşzamanlı (parallel) atar
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # Her 50 adımda bir konsola iki F-15'in durumunu yazdır
        if step_count % 50 == 0 or step_count == 1:
            print(f"\n--- Adım: {step_count} ---")
            for agent_id in ["agent_1", "agent_2"]:
                if agent_id in observations:
                    obs = observations[agent_id]
                    # Gözlem uzayımız: [İrtifa, Hız, Mesafe, ATA, AA]
                    print(f"[{agent_id}] -> İrtifa: {obs[0]:.0f} ft | Hız: {obs[1]:.0f} kts | "
                          f"Mesafe: {obs[2]:.2f} NM | ATA: {np.degrees(obs[3]):.1f}° | Ödül: {rewards[agent_id]:.2f}")
        
        # Eğer ajanlardan biri elendiyse veya süre bittiyse döngüyü kır
        # terminations veya truncations içindeki herhangi bir değer True ise bölüm biter
        if any(terminations.values()) or any(truncations.values()):
            print("\n[BÖLÜM BİTTİ] Vurulma, çakılma veya zaman aşımı gerçekleşti.")
            break
            
    # 4. Ortamı Güvenli Kapat (Bellekten JSBSim instance'larını temizler)
    env.close()
    print("=== Ortam Başarıyla Kapatıldı ===\n")

if __name__ == "__main__":
    # TEST 1: FlightGear OLMADAN (Maksimum Hız Sınır Modu)
    # time.sleep() engeline takılmadığı için bu 500 adım işlemcinizin hızında göz açıp kapayıncaya kadar biter.
    # run_pettingzoo_env(render_mode="none", total_steps=500)
    
    print("="*50)
    
    # TEST 2: FlightGear AÇIKKEN (Gerçek Zamanlı İzleme Modu)
    # Bu modu test etmek istiyorsanız arka planda FlightGear'ın dinleme modunda açık olması gerekir.
    # Aksi takdirde FlightGear portu bulamadığı için takılabilir. Test etmek için aşağıdaki satırın yorumunu kaldırabilirsiniz.
    run_pettingzoo_env(render_mode="none", total_steps=10000)