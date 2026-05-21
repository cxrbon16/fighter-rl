from gymnasium import spaces
import numpy as np
from lib.base_env import BaseF15Env


class ClimbTaskEnv(BaseF15Env):
    metadata = {"render_modes": ["human", "none", "panda"], "name": "dogfight_climb_v0"}

    def __init__(self, render_mode="none"):
        # Base sınıfı ayağa kaldır
        super().__init__(render_mode=render_mode)
        
        # Göreve Özel: Sadece 12 Boyutlu Sensör Paketi
        self.observation_spaces = {
            agent: spaces.Box(low=-5.0, high=5.0, shape=(12,), dtype=np.float32)
            for agent in self.possible_agents
        }
        
        self.prev_alts = {}
        self.prev_vcs = {}

    def _get_initial_conditions(self):
        # Tırmanış görevine özel daraltılmış başlangıç penceresi
        return {
            'alt': np.random.uniform(14000.0, 16000.0),
            'vc': np.random.uniform(380.0, 420.0),
            'pitch': np.random.uniform(-2.0, 2.0),
            'roll': np.random.uniform(-5.0, 5.0),
            'yaw': np.random.uniform(-0.2, 0.2)
        }

    def _task_reset(self):
        # Base sınıf reset attıktan sonra hafızayı tazele
        for agent in self.agents:
            self.prev_alts[agent] = self.fdms[agent]['position/h-sl-ft']
            self.prev_vcs[agent] = self.fdms[agent]['velocities/vc-kts']

    def _get_obs(self, agent_id):
        fdm = self.fdms[agent_id]
        
        alt = fdm['position/h-sl-ft']
        v_fps = fdm['velocities/vt-fps']
        specific_energy = (v_fps**2) / (2.0 * 32.17) + alt 
        g_force = fdm['accelerations/Nz'] 
        
        # Normalizasyon
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

        obs = np.array([
            norm_alt, norm_vc, norm_roll, norm_pitch, 
            norm_alpha, norm_beta, norm_g, norm_climb,
            norm_p, norm_q, norm_r, norm_energy
        ], dtype=np.float32)
        
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        return np.clip(obs, -5.0, 5.0)

    def _calculate_rewards_and_dones(self):
        rewards = {}
        terminations = {}
        truncations = {}
        
        for agent in self.possible_agents:
            if agent not in self.agents:
                continue
                
            current_alt = self.fdms[agent]['position/h-sl-ft']
            current_vc = self.fdms[agent]['velocities/vc-kts'] 
            
            delta_alt = current_alt - self.prev_alts[agent]
            rewards[agent] = delta_alt / 10.0 

            self.prev_alts[agent] = current_alt
            self.prev_vcs[agent] = current_vc
            
            terminations[agent] = False
            if current_alt < 1000.0:
                terminations[agent] = True
                rewards[agent] -= 70.0 
                
            truncations[agent] = self.fdms[agent].get_sim_time() > 30.0
            
        return rewards, terminations, truncations