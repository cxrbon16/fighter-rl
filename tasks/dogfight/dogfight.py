import numpy as np
from gymnasium import spaces
from lib.base_env import BaseEnv

class SelfPlayDogfightEnv(BaseEnv):
    metadata = {"render_modes": ["debug", "human", "none"], "name": "dogfight_v0"}

    def __init__(self, render_mode="none"):
        super().__init__(number_of_agents=2, render_mode=render_mode, aircraft="f16")
        
        self.observation_spaces = {
            agent: spaces.Box(low=-1.0, high=1.0, shape=(28,), dtype=np.float32)
            for agent in self.possible_agents
        }
        
        self.tracking_time = {
            agent: 0 for agent in self.possible_agents
        }

    def _get_initial_conditions(self):                    
        return {   
            agent:
            {
                'alt': np.random.uniform(10000.0, 20000.0),
                'vc': np.random.uniform(350.0, 450.0),
                'pitch': np.random.uniform(-10.0, 10.0),
                'roll': np.random.uniform(-30.0, 30.0),
                'yaw': np.random.uniform(0.0, 360.0),
                'lat': 37.6190 + np.random.uniform(-0.10, +0.10),
                'long': -122.3749 + np.random.uniform(-0.10, +0.10),
            } for agent in self._get_possible_agents()
        }
            
    def _task_reset(self):
        pass

    def _get_obs(self, agent_id):
        opponent_agent_id = next(uid for uid in self.fdms.keys() if uid != agent_id)
        
        fdm = self.fdms[agent_id]
        opp_fdm = self.fdms[opponent_agent_id]

        # Direct observations
        alt = fdm['position/h-sl-ft']
        g_force = fdm['accelerations/Nz'] 

        norm_alt = alt / 50000.0          
        norm_vc = fdm['velocities/vc-kts'] / 1500.0             
        norm_roll = fdm['attitude/phi-rad'] / np.pi
        norm_pitch = fdm['attitude/pitch-rad'] / np.pi
        norm_alpha = fdm['aero/alpha-rad'] / np.pi         
        norm_beta = fdm['aero/beta-rad'] / np.pi           
        norm_g = g_force / 12.0                            
        norm_climb = fdm['velocities/h-dot-fps'] / 1000.0
        norm_p = fdm['velocities/p-rad_sec'] / np.pi       
        norm_q = fdm['velocities/q-rad_sec'] / np.pi       
        norm_r = fdm['velocities/r-rad_sec'] / np.pi    

        elev = fdm["fcs/elevator-cmd-norm"]
        ail = fdm["fcs/aileron-cmd-norm"]
        rud = fdm["fcs/rudder-cmd-norm"]
        thr = fdm["fcs/throttle-cmd-norm"]   

        norm_mach = fdm["velocities/mach"] / 2.0
        norm_qbar = fdm["aero/qbar-psf"] / 2500.0  
        norm_thrust = fdm["propulsion/engine/thrust-lbs"] / 30000.0

        # Relative observations
        delta_alt = opp_fdm['position/h-sl-ft'] - alt
        norm_delta_alt = delta_alt / 30000.0

        self_x, self_y, self_z = fdm['position/ecef-x-ft'], fdm['position/ecef-y-ft'], fdm['position/ecef-z-ft']
        opp_x, opp_y, opp_z = opp_fdm['position/ecef-x-ft'], opp_fdm['position/ecef-y-ft'], opp_fdm['position/ecef-z-ft']
        dist_sq = (opp_x - self_x)**2 + (opp_y - self_y)**2 + (opp_z - self_z)**2
        distance_ft = np.sqrt(dist_sq)
        
        norm_distance = distance_ft / 100000.0 

        norm_opp_vc = opp_fdm['velocities/vc-kts'] / 1000.0
        norm_opp_roll = opp_fdm['attitude/phi-rad'] / np.pi
        norm_opp_pitch = opp_fdm['attitude/pitch-rad'] / np.pi

        # Antenna Train Angle and Aspect Angle calculations
        lat1 = fdm['position/lat-geod-rad']
        lon1 = fdm['position/long-gc-rad']
        alt1 = fdm['position/h-sl-ft']

        lat2 = opp_fdm['position/lat-geod-rad']
        lon2 = opp_fdm['position/long-gc-rad']
        alt2 = opp_fdm['position/h-sl-ft']

        R_ft = 20925000.0

        dN = (lat2 - lat1) * R_ft
        dE = (lon2 - lon1) * R_ft * np.cos(lat1)
        dD = alt1 - alt2

        los_vector = np.array([dN, dE, dD])
        los_dist = np.linalg.norm(los_vector)

        v1_vector = np.array([
            fdm['velocities/v-north-fps'],
            fdm['velocities/v-east-fps'],
            fdm['velocities/v-down-fps']
        ])
        
        v2_vector = np.array([
            opp_fdm['velocities/v-north-fps'],
            opp_fdm['velocities/v-east-fps'],
            opp_fdm['velocities/v-down-fps']
        ])

        v1_speed = np.linalg.norm(v1_vector)
        v2_speed = np.linalg.norm(v2_vector)

        if v1_speed > 0 and los_dist > 0:
            ata_cos = np.dot(v1_vector, los_vector) / (v1_speed * los_dist)
            # np.clip hayat kurtarır! Float hassasiyetinden dolayı değer -1.0000001 olursa arccos hata verir (NaN döner).
            ata_rad = np.arccos(np.clip(ata_cos, -1.0, 1.0))
        else:
            ata_rad = 0.0

        if v2_speed > 0 and los_dist > 0:
            aa_cos = np.dot(v2_vector, los_vector) / (v2_speed * los_dist)
            aa_rad = np.arccos(np.clip(aa_cos, -1.0, 1.0))
        else:
            aa_rad = 0.0

        norm_ata = ata_rad / np.pi
        norm_aa = aa_rad / np.pi

        # Energy Calculations
        G_FT_S2 = 32.174

        vt_fps = fdm['velocities/vt-fps']
        alt_ft = fdm['position/h-sl-ft']
        
        specific_energy = alt_ft + (vt_fps**2) / (2 * G_FT_S2)

        opp_vt_fps = opp_fdm['velocities/vt-fps']
        opp_alt_ft = opp_fdm['position/h-sl-ft']
        
        opp_specific_energy = opp_alt_ft + (opp_vt_fps**2) / (2 * G_FT_S2)

        delta_energy = specific_energy - opp_specific_energy

        norm_my_energy = specific_energy / 120000.0
        norm_opp_energy = opp_specific_energy / 120000.0
        
        norm_delta_energy = delta_energy / 50000.0

        obs = np.array([
            # 1. Kendi Durumumuz (Kinematik ve Duruş)
            norm_alt,           # Kendi irtifamız (Normalize)
            norm_vc,            # Kendi gösterge hızımız
            norm_mach,          # Kendi Mach hızımız (Ses hızı oranı)
            norm_roll,          # Yatış açımız (Roll)
            norm_pitch,         # Yunuslama açımız (Pitch)
            norm_p,             # Yatış ivmemiz (Roll Rate)
            norm_q,             # Yunuslama ivmemiz (Pitch Rate)
            norm_r,             # Sapma ivmemiz (Yaw Rate)
            norm_climb,         # Dikey hızımız (Tırmanış/Dalış)
            
            # 2. Aerodinamik ve Motor Verileri (Limitler)
            norm_alpha,         # Hücum açımız (AoA - Stall limiti)
            norm_beta,          # Yanal kayma açımız (Sideslip)
            norm_g,             # Hissettiğimiz G Kuvveti
            norm_qbar,          # Dinamik hava basıncı (Manevra kabiliyeti)
            norm_thrust,        # Motor itkimiz
            
            # 3. Kontrol Yüzeyleri (Proprioception - Neye basıyoruz?)
            elev,               # İrtifa dümeni konumu
            ail,                # Kanatçık konumu
            rud,                # İstikamet dümeni konumu
            thr,                # Gaz kolu konumu
            
            # 4. Rakip ile Olan Geometrik İlişki (Taktiksel)
            norm_distance,      # Rakiple aramızdaki 3D mesafe
            norm_delta_alt,     # İrtifa farkımız (Avantaj kimde?)
            norm_ata,           # Antenna Train Angle (Burnumuz rakibe bakıyor mu?)
            norm_aa,            # Aspect Angle (Rakibin neresindeyiz?)
            
            # 5. Rakibin Temel Durumu (Radar Verisi)
            norm_opp_vc,        # Rakibin hızı
            norm_opp_roll,      # Rakibin yatış açısı (Nereye dönecek?)
            norm_opp_pitch,     # Rakibin yunuslama açısı
            
            # 6. Enerji Manevra Teorisi (Specific Energy)
            norm_my_energy,     # Kendi toplam özgül enerjimiz
            norm_opp_energy,    # Rakibin toplam özgül enerjisi
            norm_delta_energy   # Aramızdaki enerji farkı
            
        ], dtype=np.float32)

        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        
        obs = np.clip(obs, -1.0, 1.0)

        return obs

    def _calculate_rewards_and_dones(self, actions):
        rewards = {agent: 0.0 for agent in self.possible_agents}
        terminations = {agent: False for agent in self.possible_agents}
        truncations = {agent: False for agent in self.possible_agents}
        
        for agent_id in self.possible_agents:
            if agent_id not in self.agents:
                 continue
                 
            fdm = self.fdms[agent_id]
            
            # Survival
            current_alt = fdm['position/h-sl-ft']
            g_force = fdm['accelerations/Nz']
            
            vt_fps = fdm['velocities/vt-fps']
            specific_energy = current_alt + (vt_fps**2) / (2 * 32.174)
            
            opponent_id = next(uid for uid in self.fdms.keys() if uid != agent_id)
            opp_fdm = self.fdms[opponent_id]
            
            # Delta Energy
            opp_vt_fps = opp_fdm['velocities/vt-fps']
            opp_alt = opp_fdm['position/h-sl-ft']
            opp_specific_energy = opp_alt + (opp_vt_fps**2) / (2 * 32.174)
            delta_energy = specific_energy - opp_specific_energy
            norm_delta_energy = delta_energy / 50000.0
            
            # Distance calcs.
            self_x, self_y, self_z = fdm['position/ecef-x-ft'], fdm['position/ecef-y-ft'], fdm['position/ecef-z-ft']
            opp_x, opp_y, opp_z = opp_fdm['position/ecef-x-ft'], opp_fdm['position/ecef-y-ft'], opp_fdm['position/ecef-z-ft']
            current_dist = np.sqrt((opp_x - self_x)**2 + (opp_y - self_y)**2 + (opp_z - self_z)**2)
            
            # ATA and AA calcs.
            lat1, lon1 = fdm['position/lat-geod-rad'], fdm['position/long-gc-rad']
            lat2, lon2 = opp_fdm['position/lat-geod-rad'], opp_fdm['position/long-gc-rad']
            R_ft = 20925000.0
            
            dN = (lat2 - lat1) * R_ft
            dE = (lon2 - lon1) * R_ft * np.cos(lat1)
            dD = current_alt - opp_alt
            los_vector = np.array([dN, dE, dD])
            los_dist = np.linalg.norm(los_vector)

            v1_vector = np.array([fdm['velocities/v-north-fps'], fdm['velocities/v-east-fps'], fdm['velocities/v-down-fps']])
            v2_vector = np.array([opp_fdm['velocities/v-north-fps'], opp_fdm['velocities/v-east-fps'], opp_fdm['velocities/v-down-fps']])
            
            v1_speed = np.linalg.norm(v1_vector)
            v2_speed = np.linalg.norm(v2_vector)

            ata_rad = 0.0
            if v1_speed > 0 and los_dist > 0:
                ata_cos = np.dot(v1_vector, los_vector) / (v1_speed * los_dist)
                ata_rad = np.arccos(np.clip(ata_cos, -1.0, 1.0))

            aa_rad = 0.0
            if v2_speed > 0 and los_dist > 0:
                aa_cos = np.dot(v2_vector, los_vector) / (v2_speed * los_dist)
                aa_rad = np.arccos(np.clip(aa_cos, -1.0, 1.0))
                
            norm_ata = ata_rad / np.pi
            norm_aa = aa_rad / np.pi

            # Reward Shaping
            step_reward = 0.0

            # Survival Reward
            step_reward += 0.05 
            
            # G Limit
            if g_force > 9.0 or g_force < -3.0:
                step_reward -= 0.5  

            # Stability                
            agent_actions = actions[agent_id]
            action_penalty = 0.03 * (agent_actions[0]**2 + agent_actions[1]**2 + agent_actions[2]**2)
            step_reward -= action_penalty

            # Positioning & Geometry
            offensive_score = (1.0 - norm_ata) + (1.0 - norm_aa)
            step_reward += (offensive_score - 1.0) * 0.1
            
            # Energy 
            if norm_delta_energy > 0:
                step_reward += 0.05
                
            # Lethality (WEZ - Weapon Engagement Zone)
            in_wez = current_dist < 3000.0 and ata_rad < (10.0 * np.pi / 180.0)
            
            if in_wez:
                 step_reward += 2.0
                 self.tracking_time[agent_id] += 1
            else:
                 self.tracking_time[agent_id] = max(0, self.tracking_time[agent_id] - 1) 

            if current_alt < 1000.0:
                step_reward -= 100.0
                terminations[agent_id] = True
                
            elif current_dist > 150000.0: 
                step_reward -= 50.0
                terminations[agent_id] = True
                
            elif self.tracking_time[agent_id] > 50:
                 step_reward += 100.0
                 terminations[agent_id] = True
                 terminations[opponent_id] = True
                 rewards[opponent_id] -= 100.0

            if fdm.get_sim_time() > 180.0:
                truncations[agent_id] = True

            rewards[agent_id] += step_reward
            
        return rewards, terminations, truncations