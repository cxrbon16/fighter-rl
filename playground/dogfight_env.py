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
        
        # 🔥 SADECE TEK AJAN KALDI
        self.possible_agents = ["agent_1"]
        self.agents = self.possible_agents[:]
        
        # [Aileron, Elevator, Throttle]
        self.action_spaces = {
            agent: spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
            for agent in self.possible_agents
        }
        
        # 🔥 OBS SPACE KÜÇÜLTÜLDÜ (12 Boyutlu Saf Aerodinamik Sensör Paketi)
        self.observation_spaces = {
                agent: spaces.Box(
                    low=-5.0,
                    high=5.0,
                    shape=(12,),
                    dtype=np.float32
                )
                for agent in self.possible_agents
        }

        self.fdms = {}
        self.dt = 1.0 / 120.0
        
        self.mp_sender = None

        self.viewer = None
        if self.render_mode == "panda":
            from panda_viewer import DogfightViewer 
            self.viewer = DogfightViewer()
            
    @lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]

    def _get_obs(self, agent_id):
        fdm = self.fdms[agent_id]
        
        # --- TEMEL DURUM ---
        alt = fdm['position/h-sl-ft']
        
        # --- AERODİNAMİK SENSÖRLER ---
        v_fps = fdm['velocities/vt-fps']
        specific_energy = (v_fps**2) / (2.0 * 32.17) + alt 
        g_force = fdm['accelerations/Nz'] 
        
        # --- NORMALİZASYON SÜRECİ ---
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

        # 12 Boyutlu Düşmansız Sensör Matrisi
        obs = np.array([
            norm_alt, norm_vc, norm_roll, norm_pitch, 
            norm_alpha, norm_beta, norm_g, norm_climb,
            norm_p, norm_q, norm_r, norm_energy
        ], dtype=np.float32)
        
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        obs = np.clip(obs, -5.0, 5.0)
        
        return obs
    
    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.fdms = {}
        
        # --- BEBEK ADIMLARI: DARALTILMIŞ BAŞLANGIÇ (Curriculum Learning) ---
        start_alt_1 = np.random.uniform(14000.0, 16000.0)
        start_speed_1 = np.random.uniform(380.0, 420.0)

        start_pitch_1 = np.random.uniform(-2.0, 2.0)
        start_roll_1 = np.random.uniform(-5.0, 5.0)
        start_yaw_1 = np.random.uniform(-0.2, 0.2) 

        # --- agent_1 Başlat ---
        f15_1 = jsbsim.FGFDMExec(None)
        
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
        
        # İniş takımlarını topla
        f15_1['fcs/gear-cmd-norm'] = 0.0       
        f15_1['gear/unit[0]/pos-norm'] = 0.0   
        f15_1['gear/unit[1]/pos-norm'] = 0.0   
        f15_1['gear/unit[2]/pos-norm'] = 0.0   
    
        self.fdms["agent_1"] = f15_1
        
        # Hafıza sözlüklerini güncelle
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

        # 4 Frame (veya 8 Frame) kuralımız
        for _ in range(8):
            self.fdms["agent_1"].run()
        
        rewards = {}
        terminations = {}
        truncations = {}
        
        # 2. ÖDÜL MATEMATİĞİ (İrtifa kazanmaya büyük odak)
        for agent in self.possible_agents:
            current_alt = self.fdms[agent]['position/h-sl-ft']
            current_vc = self.fdms[agent]['velocities/vc-kts'] 
            
            delta_alt = current_alt - self.prev_alts[agent]
            
            # 🔥 İrtifa kazanmayı daha çok ödüllendiriyoruz
            alt_reward = delta_alt / 10.0 
                                    
            rewards[agent] = alt_reward

            self.prev_alts[agent] = current_alt
            self.prev_vcs[agent] = current_vc
            
            terminations[agent] = False
            if current_alt < 1000.0:
                terminations[agent] = True
                rewards[agent] -= 70.0 
                
            truncations[agent] = self.fdms[agent].get_sim_time() > 30.0

        # Terminate / Truncate temizliği
        for agent in self.possible_agents:
            if (terminations[agent] or truncations[agent]) and (agent in self.agents):
                self.agents.remove(agent)

        out_observations = {agent: self._get_obs(agent) for agent in self.agents}
        out_rewards = {agent: rewards[agent] for agent in self.agents}
        out_terminations = {agent: terminations[agent] for agent in self.agents}
        out_truncations = {agent: truncations[agent] for agent in self.agents}
        out_infos = {agent: {} for agent in self.agents}

        # --- GÖRSELLEŞTİRME ---
        if self.render_mode == "human" and hasattr(self, "string_sender"):
            # human modda fg_string_sender şu an f15_2 arıyordu, kaldırdık
            pass 

        if self.render_mode == "panda" and self.viewer is not None:
            p1_state = {
                'x': (self.fdms["agent_1"]['position/long-gc-deg'] - (-122.3749)) * 100000,
                'y': (self.fdms["agent_1"]['position/lat-geod-deg'] - 37.6190) * 100000,
                'z': self.fdms["agent_1"]['position/h-sl-ft'],
                'roll': self.fdms["agent_1"]['attitude/phi-deg'],
                'pitch': self.fdms["agent_1"]['attitude/theta-deg'],
                'yaw': self.fdms["agent_1"]['attitude/psi-deg']
            }
            
            # Viewer hata vermesin diye ikinci uçağı yerin dibine hayalet olarak gömüyoruz
            p2_dummy_state = {
                'x': 0.0, 'y': 0.0, 'z': -50000.0, 
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0
            }
            
            self.viewer.update_world(p1_state, p2_dummy_state)

        return out_observations, out_rewards, out_terminations, out_truncations, out_infos