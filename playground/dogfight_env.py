from pettingzoo import ParallelEnv
from gymnasium import spaces
import jsbsim
import numpy as np
import time
from fg_string_sender import FGStringSender

class DogfightParallelEnv(ParallelEnv):
    metadata = {"render_modes": ["human", "none"], "name": "dogfight_v2"}

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

        # YENİ SENSÖRLER: [İrtifa, Hız, Roll(Yatış), Pitch(Yunuslama), Rakibe Mesafe, ATA, AA, Rakiple İrtifa Farkı]
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
        
        # Kendi Durumu
        alt = fdm['position/h-sl-ft']
        vc = fdm['velocities/vc-kts']
        roll = fdm['attitude/phi-rad']   # Uçağın yatış açısı (Çok Kritik!)
        pitch = fdm['attitude/pitch-rad'] # Uçağın burun açısı (Çok Kritik!)
        
        # Geometri
        dist, ata, aa = self._get_combat_geometry(agent_id)
        
        # Rakiple İrtifa Farkı (Enerji avantajını anlaması için)
        enemy_id = "agent_2" if agent_id == "agent_1" else "agent_1"
        enemy_alt = self.fdms[enemy_id]['position/h-sl-ft']
        alt_diff = alt - enemy_alt # Pozitifse ben yukarıdayım, negatifse o yukarıda
        
        return np.array([alt, vc, roll, pitch, dist, ata, aa, alt_diff], dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.fdms = {}
        
        # --- RASTGELE BAŞLANGIÇ KOŞULLARI (DOMAIN RANDOMIZATION) ---
        # Uçaklar 10.000 ile 20.000 fit arası rastgele başlasın
        start_alt_1 = np.random.uniform(10000.0, 20000.0)
        start_alt_2 = np.random.uniform(10000.0, 20000.0)
        
        # Hızları 300 ile 500 knot arası değişsin
        start_speed_1 = np.random.uniform(300.0, 500.0)
        start_speed_2 = np.random.uniform(300.0, 500.0)

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
        f15_1['ic/psi-true-deg'] = 0.0 # Kuzeye baksın
        
        f15_1['propulsion/engine[0]/set-running'] = 1
        f15_1['propulsion/engine[1]/set-running'] = 1
        f15_1.run_ic()
        f15_1['simulation/do_simple_trim'] = 1
        self.fdms["agent_1"] = f15_1
        
        # --- agent_2 ---
        if self.render_mode == "human":
            self.string_sender = FGStringSender(dest_ip="127.0.0.1", dest_port=5555)
            
        f15_2 = jsbsim.FGFDMExec(None)
        f15_2.load_model('f15')
        # Agent 2'yi rastgele bir ofsetle (örneğin biraz daha doğuda/kuzeyde) başlat
        lat_offset = np.random.uniform(-0.05, 0.05)
        lon_offset = np.random.uniform(-0.05, 0.05)
        
        f15_2['ic/lat-geod-deg'] = 37.6190 + lat_offset
        f15_2['ic/long-gc-deg'] = -122.3749 + lon_offset
        f15_2['ic/h-sl-ft'] = start_alt_2
        f15_2['ic/vc-kts'] = start_speed_2
        
        # Agent 2'nin yönünü tamamen rastgele yap (0 ile 360 derece arası)
        f15_2['ic/psi-true-deg'] = np.random.uniform(0.0, 360.0) 
        
        f15_2['propulsion/engine[0]/set-running'] = 1
        f15_2['propulsion/engine[1]/set-running'] = 1
        f15_2.run_ic()
        f15_2['simulation/do_simple_trim'] = 1
        self.fdms["agent_2"] = f15_2
        
        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        
        self.prev_alts = {}
        for agent in self.agents:
            self.prev_alts[agent] = self.fdms[agent]['position/h-sl-ft']
        
        return observations, infos
    
    def step(self, actions):
        if not actions:
            return {}, {}, {}, {}, {}

        # 1. Komutları uçağa ilet (Değişiklik yok)
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

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        
        rewards = {}
        terminations = {}
        truncations = {}
        
        is_done = False
        
        # 2. DELTA İRTİFA (TIRMANMA) ÖDÜLÜ
        for agent in self.agents:
            # Uçağın anlık irtifasını al
            current_alt = self.fdms[agent]['position/h-sl-ft']
            
            # ÖNCEKİ ADIMLA ARADAKİ FARKI (DELTA) BUL
            delta_alt = current_alt - self.prev_alts[agent]
            
            # ÖDÜL MATEMATİĞİ: 
            # 10 feet tırmandıysa -> +0.1 puan. 10 feet düştüyse -> -0.1 puan.
            # (100'e bölerek puanı normalize ediyoruz ki ağ patlamasın)
            rewards[agent] = delta_alt / 100.0 
            
            # Bir sonraki adımda kullanmak üzere eski irtifayı güncelle!
            self.prev_alts[agent] = current_alt
            
            terminations[agent] = False
            
            # Uçak yere çakılırsa
            if current_alt < 1000.0:
                is_done = True
                terminations[agent] = True
                rewards[agent] -= 100.0 # Çakılma cezası
                
            # 30 saniyelik süre sınırı
            truncations[agent] = self.fdms[agent].get_sim_time() > 30.0

        infos = {agent: {} for agent in self.agents}

        if is_done or truncations["agent_1"]:
            self.agents = []
        
        if self.render_mode == "human":
            if hasattr(self, "string_sender"):
                self.string_sender.send_state(self.fdms["agent_2"])
            time.sleep(self.dt * 4)

        return observations, rewards, terminations, truncations, infos