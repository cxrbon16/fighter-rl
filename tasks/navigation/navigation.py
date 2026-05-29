import numpy as np
from gymnasium import spaces
from lib.base_env import BaseF15Env

class NavigationTaskEnv(BaseF15Env):
    metadata = {"render_modes": ["human", "none", "panda"], "name": "dogfight_nav_v0"}

    def __init__(self, render_mode="none"):
        super().__init__(render_mode=render_mode)
        
        # 🔥 OBS SPACE: 12 Aerodinamik + 2 Navigasyon (Mesafe ve Açı) = 14 Boyut
        self.observation_spaces = {
            agent: spaces.Box(low=-5.0, high=5.0, shape=(14,), dtype=np.float32)
            for agent in self.possible_agents
        }
        
        self.targets = {}
        self.prev_dist = {}

    def _get_initial_conditions(self):
        # Navigasyonda uçak biraz daha serbest olabilir
        return {
            'alt': np.random.uniform(10000.0, 20000.0),
            'vc': np.random.uniform(350.0, 450.0),
            'pitch': np.random.uniform(-10.0, 10.0),
            'roll': np.random.uniform(-30.0, 30.0),
            'yaw': np.random.uniform(-0.2, 0.2)
        }

    def _task_reset(self):
        # Her reset'te yeni bir hedef noktası generate et
        for agent in self.agents:
            fdm = self.fdms[agent]
            # Uçağın bulunduğu yerin etrafında rastgele bir hedef (enlem/boylam ofseti)
            lat = fdm['position/lat-geod-deg']
            lon = fdm['position/long-gc-deg']
            
            # 0.1 derece yaklaşık 6-7 mil eder, başlangıç için ideal menzil
            self.targets[agent] = (lat + np.random.uniform(-0.05, 0.05), 
                                   lon + np.random.uniform(-0.05, 0.05), 
                                   np.random.uniform(10000.0, 20000.0)) # Hedef irtifa
            
            self.prev_dist[agent] = self._calculate_dist(agent)

    def _calculate_dist(self, agent):
        fdm = self.fdms[agent]
        d_lat = (fdm['position/lat-geod-deg'] - self.targets[agent][0]) * 364000 # feet (yaklaşık)
        d_long = (fdm['position/long-gc-deg'] - self.targets[agent][1]) * 364000
        return np.sqrt(d_lat**2 + d_long**2)

    def _get_obs(self, agent_id):
        # Base sensörleri hesapla (eski kodunla aynı)
        fdm = self.fdms[agent_id]
        
        # Temel verileri al
        alt = fdm['position/h-sl-ft']
        v_fps = fdm['velocities/vt-fps']
        specific_energy = (v_fps**2) / (2.0 * 32.17) + alt 
        g_force = fdm['accelerations/Nz'] 
        
        # Normalizasyonlar (Standart F-15 Veri Paketi)
        norm_alt = alt / 30000.0          
        norm_vc = fdm['velocities/vc-kts'] / 1000.0             
        norm_roll = fdm['attitude/phi-rad'] / np.pi
        norm_pitch = fdm['attitude/pitch-rad'] / np.pi
        norm_alpha = fdm['aero/alpha-rad'] / np.pi         
        norm_beta = fdm['aero/beta-rad'] / np.pi           
        norm_g = g_force / 10.0                            
        norm_climb = fdm['velocities/h-dot-fps'] / 1000.0  
        norm_p = fdm['velocities/p-rad_sec'] / np.pi       
        norm_q = fdm['velocities/q-rad_sec'] / np.pi       
        norm_r = fdm['velocities/r-rad_sec'] / np.pi       
        norm_energy = specific_energy / 100000.0           

        # 12 Boyutlu matrisi döndür
        obs = np.array([
            norm_alt, norm_vc, norm_roll, norm_pitch, 
            norm_alpha, norm_beta, norm_g, norm_climb,
            norm_p, norm_q, norm_r, norm_energy
        ], dtype=np.float32)
        
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

        # Navigasyon Sensörleri
        dist = self._calculate_dist(agent_id)
        
        # Açı hesaplama: Hedefin uçağa göre yönü
        d_lat = (self.targets[agent_id][0] - fdm['position/lat-geod-deg'])
        d_long = (self.targets[agent_id][1] - fdm['position/long-gc-deg'])
        target_bearing = np.arctan2(d_long, d_lat)
        current_heading = fdm['attitude/psi-rad']
        bearing_error = (target_bearing - current_heading + np.pi) % (2 * np.pi) - np.pi
        
        # Yeni verileri matrise ekle
        nav_obs = np.array([dist / 50000.0, bearing_error / np.pi], dtype=np.float32)
        
        return np.concatenate([obs, nav_obs])

    def _calculate_rewards_and_dones(self, actions):
        rewards = {}
        terminations = {}
        truncations = {}
        
        for agent in self.possible_agents:
            if agent not in self.agents:
                 continue
            
            # İRTİFAYI ÇEKİYORUZ
            current_alt = self.fdms[agent]['position/h-sl-ft']
            current_dist = self._calculate_dist(agent)
            
            # Ödül: Mesafedeki değişime göre
            rewards[agent] = (self.prev_dist[agent] - current_dist) / 100.0
            self.prev_dist[agent] = current_dist
            
            terminations[agent] = False
            
            # 1. BİTİŞ KOŞULU: HEDEFE ULAŞMA (BAŞARI)
            if current_dist < 1000.0:
                rewards[agent] += 400.0
                terminations[agent] = True
                
            # 2. BİTİŞ KOŞULU: YERE ÇAKILMA (İşte FPS'i kurtaracak satırlar!)
            if current_alt < 1000.0:
                rewards[agent] -= 70.0 # Yere çakılma cezası
                terminations[agent] = True
            
            agent_actions = actions[agent]
            
            action_penalty = 0.03 * (agent_actions[0]**2 + agent_actions[1]**2 + agent_actions[2] ** 2) 
            rewards[agent] -= action_penalty
            
            # 3. BİTİŞ KOŞULU: SÜRE DOLMASI
            truncations[agent] = self.fdms[agent].get_sim_time() > 120.0
            
        return rewards, terminations, truncations
    
    def _update_info(self, infos: dict):
        """Base sınıftan gelen saf F-15 telemetrisine, göreve özel Hedef (Target) verilerini ekler."""
        for agent_id, agent_info in infos.items():
            # Eğer debug modundaysak agent_info dolu gelir, içine target'ı gömelim
            if agent_info: 
                hedef = self.targets[agent_id]
                agent_info['target_lat'] = hedef[0]
                agent_info['target_lon'] = hedef[1]
                agent_info['target_alt_m'] = hedef[2] * 0.3048 # Fitten metreye
                
        return infos