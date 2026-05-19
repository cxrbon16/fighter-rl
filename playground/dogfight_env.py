from pettingzoo import ParallelEnv
from gymnasium import spaces
import jsbsim
import numpy as np
import time
from fg_string_sender import FGStringSender
from functools import lru_cache
import panda_viewer

class DogfightParallelEnv(ParallelEnv):
    metadata = {"render_modes": ["human", "none", "panda"], "name": "dogfight_v2"}

    def __init__(self, render_mode="none"):
        super().__init__()
        self.render_mode = render_mode
        
        self.possible_agents = ["agent_1", "agent_2"]
        self.agents = self.possible_agents[:]
        
        # [Aileron, Elevator, Throttle]
        self.action_spaces = {
            agent: spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
            for agent in self.possible_agents
        }
        
        self.observation_spaces = {
                agent: spaces.Box(
                    low=np.array([0.0, 0.0, -np.pi, -np.pi/2, 0.0, -np.pi, -np.pi, -50000.0]),
                    high=np.array([80000.0, 1500.0, np.pi, np.pi/2, 500000.0, np.pi, np.pi, 50000.0]),
                    dtype=np.float32
                )
                for agent in self.possible_agents
        }

        self.fdms = {}
        self.dt = 1.0 / 120.0
        
        self.mp_sender = None

        self.viewer = None
        if self.render_mode == "panda":
            from panda_viewer import DogfightViewer # Yukarıda yazdığımız dosyayı çektik
            self.viewer = DogfightViewer()
            

    @lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]

    def _get_combat_geometry(self, my_id):
        """Gerçek 3D vektör matematiği kullanarak ATA ve AA açılarını hesaplar"""
        my_fdm = self.fdms[my_id]
        enemy_id = "agent_2" if my_id == "agent_1" else "agent_1"
        enemy_fdm = self.fdms[enemy_id]
        
        # Dereceleri Radyana çeviriyoruz
        to_rad = np.pi / 180.0
        
        # Basit yerel koordinat düzlemi (X: Doğu, Y: Kuzey, Z: Yukarı)
        # Enlem/Boylam farklarını feet/metre cinsine yaklaştırıyoruz
        dx = (enemy_fdm['position/long-gc-deg'] - my_fdm['position/long-gc-deg'])   
        dy = (enemy_fdm['position/lat-geod-deg'] - my_fdm['position/lat-geod-deg']) * 60.0 * 6076.1
        dz = enemy_fdm['position/h-sl-ft'] - my_fdm['position/h-sl-ft']
        
        # Bağıl mesafe (Feet cinsinden alıp Deniz Miline çeviriyoruz)
        dist_ft = np.sqrt(dx**2 + dy**2 + dz**2)
        distance_nm = dist_ft / 6076.1 if dist_ft > 0 else 0.0
        
        if dist_ft == 0:
            return 0.0, 0.0, 0.0
            
        # Görüş Hattı Vektörü (Line of Sight - LOS)
        los_vector = np.array([dx, dy, dz]) / dist_ft
        
        # Ajanların kendi Heading (Rota) ve Pitch açılarından yön vektörlerini buluyoruz
        def get_heading_vector(fdm):
            h = fdm['attitude/psi-rad']  # Gerçek yönelim radyanı
            p = fdm['attitude/pitch-rad']
            return np.array([np.sin(h)*np.cos(p), np.cos(h)*np.cos(p), np.sin(p)])
            
        my_dir = get_heading_vector(my_fdm)
        enemy_dir = get_heading_vector(enemy_fdm)
        
        # 1. ATA (Antenna Train Angle): Benim burnumun rakibe olan açısı
        dot_ata = np.clip(np.dot(my_dir, los_vector), -1.0, 1.0)
        ata_angle = np.arccos(dot_ata) # Radyan [0, pi]
        
        # 2. AA (Aspect Angle): Rakibin kuyruğunun bana olan açısı
        dot_aa = np.clip(np.dot(enemy_dir, los_vector), -1.0, 1.0)
        aa_angle = np.arccos(dot_aa) # Radyan [0, pi]
        
        return distance_nm, ata_angle, aa_angle

    def _get_obs(self, agent_id):
        fdm = self.fdms[agent_id]
        
        # 1. Kendi Durumu (Ham Veriler)
        alt = fdm['position/h-sl-ft']
        vc = fdm['velocities/vc-kts']
        roll = fdm['attitude/phi-rad']   
        pitch = fdm['attitude/pitch-rad'] 
        
        # 2. Geometri (Ham Veriler)
        dist, ata, aa = self._get_combat_geometry(agent_id)
        
        # 3. Rakiple İrtifa Farkı
        enemy_id = "agent_2" if agent_id == "agent_1" else "agent_1"
        enemy_alt = self.fdms[enemy_id]['position/h-sl-ft']
        alt_diff = alt - enemy_alt 
        
        # --- NORMALİZASYON (HAYAT KURTARAN KISIM) ---
        # Sayıları PPO'nun sevdiği küçük aralıklara (yaklaşık -1 ile 1 arası) basıyoruz.
        norm_alt = alt / 30000.0          # Max 30.000 fit görsek, [0, 1] arası olur.
        norm_vc = vc / 1000.0             # Max 1000 knot görsek, [0, 1] arası olur.
        norm_dist = dist / 20000.0        # Radarın max menzili 20k fit olsa, [0, 1] arası.
        norm_alt_diff = alt_diff / 10000.0 # Fark genelde -10k ile 10k arasındadır -> [-1, 1]
        
        # Açılar radyan olduğu için zaten -pi ile +pi arasında. 
        # Onları da pi'ye bölerek tam [-1, 1] aralığına oturtuyoruz. Saf temizlik!
        norm_roll = roll / np.pi
        norm_pitch = pitch / np.pi
        norm_ata = ata / np.pi
        norm_aa = aa / np.pi

        # 4. Gözlem Dizisi
        obs = np.array([
            norm_alt, norm_vc, norm_roll, norm_pitch, 
            norm_dist, norm_ata, norm_aa, norm_alt_diff
        ], dtype=np.float32)
        
        # --- GÜVENLİK DUVARI (FIREWALL) ---
        # JSBSim fizikten dolayı patlayıp NaN veya Sonsuz verirse, ağı zehirlemesini engelle.
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Güvenlik ağı: Hiçbir değerin yanlışlıkla -5 ile 5 bandından çıkmamasını garanti et.
        obs = np.clip(obs, -5.0, 5.0)
        
        return obs
    
    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.fdms = {}
        
        # --- RASTGELE BAŞLANGIÇ KOŞULLARI (DOMAIN RANDOMIZATION) ---
        # 1. İrtifa ve Hız
        start_alt_1 = np.random.uniform(10000.0, 20000.0)
        start_alt_2 = np.random.uniform(10000.0, 20000.0)
        
        start_speed_1 = np.random.uniform(300.0, 500.0)
        start_speed_2 = np.random.uniform(300.0, 500.0)

        # 2. RASTGELE EULER AÇILARI (PITCH, ROLL, YAW)
        start_pitch_1 = np.random.uniform(-5.0, 5.0)
        start_pitch_2 = np.random.uniform(-5.0, 5.0)
        
        start_roll_1 = np.random.uniform(-15.0, 15.0)
        start_roll_2 = np.random.uniform(-15.0, 15.0)
        
        start_yaw_1 = np.random.uniform(0.0, 6.0)
        start_yaw_2 = np.random.uniform(0.0, 6.0)

        # --- agent_1 ---
        f15_1 = jsbsim.FGFDMExec(None)
        f15_1['fcs/gear-cmd-norm'] = 0.0
        f15_1['gear/unit[0]/pos-norm'] = 0.0
        f15_1['gear/unit[1]/pos-norm'] = 0.0
        f15_1['gear/unit[2]/pos-norm'] = 0.0
        
        if self.render_mode == "human":
            f15_1.set_output_directive('/home/ayganyavuz/Desktop/dogfighting_rl/playground/fg_output.xml')
            
        f15_1.load_model('f15')
        f15_1['ic/lat-geod-deg'] = 37.6190
        f15_1['ic/long-gc-deg'] = -122.3749
        f15_1['ic/h-sl-ft'] = start_alt_1
        f15_1['ic/vc-kts'] = start_speed_1
        
        f15_1['ic/theta-deg'] = start_pitch_1
        f15_1['ic/phi-deg'] = start_roll_1
        f15_1['ic/psi-true-deg'] = start_yaw_1 
        
        f15_1['propulsion/engine[0]/set-running'] = 1
        f15_1['propulsion/engine[1]/set-running'] = 1
        f15_1.run_ic()
        f15_1['fcs/gear-cmd-norm'] = 0.0       
        f15_1['gear/unit[0]/pos-norm'] = 0.0   
        f15_1['gear/unit[1]/pos-norm'] = 0.0   
        f15_1['gear/unit[2]/pos-norm'] = 0.0   
    
        self.fdms["agent_1"] = f15_1
        
        # --- agent_2 ---
        if self.render_mode == "human":
            self.string_sender = FGStringSender(dest_ip="127.0.0.1", dest_port=5555)
            
        f15_2 = jsbsim.FGFDMExec(None)
        f15_2.load_model('f15')
        
        lat_offset = np.random.uniform(-0.05, 0.05)
        lon_offset = np.random.uniform(-0.05, 0.05)
        
        f15_2['ic/lat-geod-deg'] = 37.6190 + lat_offset
        f15_2['ic/long-gc-deg'] = -122.3749 + lon_offset
        f15_2['ic/h-sl-ft'] = start_alt_2
        f15_2['ic/vc-kts'] = start_speed_2
        
        f15_2['ic/theta-deg'] = start_pitch_2
        f15_2['ic/phi-deg'] = start_roll_2
        f15_2['ic/psi-true-deg'] = start_yaw_2
        
        f15_2['propulsion/engine[0]/set-running'] = 1
        f15_2['propulsion/engine[1]/set-running'] = 1

        f15_2.run_ic()
        
        f15_2['fcs/gear-cmd-norm'] = 0.0       
        f15_2['gear/unit[0]/pos-norm'] = 0.0   
        f15_2['gear/unit[1]/pos-norm'] = 0.0   
        f15_2['gear/unit[2]/pos-norm'] = 0.0   

        self.fdms["agent_2"] = f15_2
        
        # 🔥 DÜZELTİLEN KISIM: Hem irtifa hem hız hafızası eksiksiz doluyor
        self.prev_alts = {}
        self.prev_vcs = {}
        for agent in self.agents:
            self.prev_alts[agent] = self.fdms[agent]['position/h-sl-ft']
            self.prev_vcs[agent] = self.fdms[agent]['velocities/vc-kts'] 
            
        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        
        return observations, infos

    def step(self, actions):
        if not actions:
            return {}, {}, {}, {}, {}

        # 1. Komutları uçağa ilet
        for agent_id, action in actions.items():
            fdm = self.fdms[agent_id]
            fdm['fcs/aileron-cmd-norm'] = float(action[0])
            fdm['fcs/elevator-cmd-norm'] = float(action[1])
            throttle = float((action[2] + 1.0) / 2.0)
            fdm['fcs/throttle-cmd-norm[0]'] = throttle
            fdm['fcs/throttle-cmd-norm[1]'] = throttle

        # 4 Frame kuralımız
        for _ in range(4):
            self.fdms["agent_1"].run()
            self.fdms["agent_2"].run()
        
        rewards = {}
        terminations = {}
        truncations = {}
        
        # 2. ÇİFT DELTA ÖDÜL MATEMATİĞİ (Tüm olası uçaklar için peşinen hesapla)
        for agent in self.possible_agents:
            current_alt = self.fdms[agent]['position/h-sl-ft']
            current_vc = self.fdms[agent]['velocities/vc-kts'] 
            
            delta_alt = current_alt - self.prev_alts[agent]
            delta_vc = current_vc - self.prev_vcs[agent]
            
            alt_reward = delta_alt / 20.0
            vc_reward = delta_vc / 10.0
            
            if delta_alt < 0:
                alt_reward = (delta_alt / 100.0) * 2.0 
            
            if delta_vc < 0:
                vc_reward = (delta_vc / 10.0) * 1.5
                
            reward = alt_reward + vc_reward
            
            if current_vc < 150.0:
                reward -= 5.0
            
            rewards[agent] = reward

            self.prev_alts[agent] = current_alt
            self.prev_vcs[agent] = current_vc
            
            # --- KRİTİK TERMİNASYON KONTROLLERİ ---
            terminations[agent] = False
            if current_alt < 1000.0:
                terminations[agent] = True
                rewards[agent] -= 100.0 # Çakılma cezası
                
            truncations[agent] = self.fdms[agent].get_sim_time() > 30.0

        # 🔥 ADIM A: Hangi uçak öldüyse veya süresi bittiyse onu self.agents listesinden SİL!
        for agent in self.possible_agents:
            if (terminations[agent] or truncations[agent]) and (agent in self.agents):
                self.agents.remove(agent)

        # 🔥 ADIM B: SuperSuit'e sadece ve sadece YAŞAYAN ajanların verilerini gönder!
        out_observations = {agent: self._get_obs(agent) for agent in self.agents}
        out_rewards = {agent: rewards[agent] for agent in self.agents}
        out_terminations = {agent: terminations[agent] for agent in self.agents}
        out_truncations = {agent: truncations[agent] for agent in self.agents}
        out_infos = {agent: {} for agent in self.agents}

        if self.render_mode == "human":
            if hasattr(self, "string_sender"):
                self.string_sender.send_state(self.fdms["agent_2"])
            time.sleep(self.dt * 4)

        if self.render_mode == "panda" and self.viewer is not None:
            # JSBSim'in Küresel (Enlem/Boylam) verilerini 3D Kartezyen düzleme uyduruyoruz.
            # Dereceleri 100000 ile çarparak metremsi bir X,Y düzlemine yayıyoruz.
            
            p1_state = {
                'x': (self.fdms["agent_1"]['position/long-gc-deg'] - (-122.3749)) * 100000,
                'y': (self.fdms["agent_1"]['position/lat-geod-deg'] - 37.6190) * 100000,
                'z': self.fdms["agent_1"]['position/h-sl-ft'],
                'roll': self.fdms["agent_1"]['attitude/phi-deg'],
                'pitch': self.fdms["agent_1"]['attitude/theta-deg'],
                'yaw': self.fdms["agent_1"]['attitude/psi-deg']
            }
            
            p2_state = {
                'x': (self.fdms["agent_2"]['position/long-gc-deg'] - (-122.3749)) * 100000,
                'y': (self.fdms["agent_2"]['position/lat-geod-deg'] - 37.6190) * 100000,
                'z': self.fdms["agent_2"]['position/h-sl-ft'],
                'roll': self.fdms["agent_2"]['attitude/phi-deg'],
                'pitch': self.fdms["agent_2"]['attitude/theta-deg'],
                'yaw': self.fdms["agent_2"]['attitude/psi-deg']
            }
            
            # 3D Motoru güncelle
            self.viewer.update_world(p1_state, p2_state)


        # Temizlenmiş, filtrelenmiş paketleri fırlatıyoruz
        return out_observations, out_rewards, out_terminations, out_truncations, out_infos