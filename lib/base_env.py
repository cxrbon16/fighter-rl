from pettingzoo import ParallelEnv
from gymnasium import spaces
import jsbsim
from functools import lru_cache
from pathlib import Path

jsbsim.FGJSBBase().debug_lvl = 0

class BaseF15Env(ParallelEnv):
    metadata = {"render_modes": ["human", "none", "panda"], "name": "base_f15_v0"}

    def __init__(self, render_mode="none"):
        super().__init__()
        self.render_mode = render_mode
        
        self.possible_agents = ["agent_1"]
        self.agents = self.possible_agents[:]
        
        # F-15 için Aksiyon Uzayı Sabittir [Aileron, Elevator, Throttle]
        self.action_spaces = {
            agent: spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=float)
            for agent in self.possible_agents
        }
        
        # Sensör Uzayı Göreve (Task) göre değişeceği için boş bırakıyoruz
        self.observation_spaces = {}

        self.fdms = {}
        self.dt = 1.0 / 120.0
        
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

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.fdms = {}
        
        # Alt sınıfın (Task) belirlediği başlangıç koordinatlarını ve hızını al
        initial_conditions = self._get_initial_conditions()
        
        f15_1 = jsbsim.FGFDMExec(None)
        if self.render_mode == "human":
            config_path = Path(__file__).resolve().parent.parent / 'config' / 'agent1_protocol.xml'
            config_path_str = str(config_path)
            f15_1.set_output_directive(config_path_str)
            
        f15_1.load_model('f15')
        f15_1['ic/lat-geod-deg'] = 37.6190
        f15_1['ic/long-gc-deg'] = -122.3749
        
        # Başlangıç dinamiklerini enjekte et
        f15_1['ic/h-sl-ft'] = initial_conditions['alt']
        f15_1['ic/vc-kts'] = initial_conditions['vc']
        f15_1['ic/theta-deg'] = initial_conditions['pitch']
        f15_1['ic/phi-deg'] = initial_conditions['roll']
        f15_1['ic/psi-true-deg'] = initial_conditions['yaw']
        
        f15_1['propulsion/engine[0]/set-running'] = 1
        f15_1['propulsion/engine[1]/set-running'] = 1
        f15_1.run_ic()
        
        f15_1['fcs/gear-cmd-norm'] = 0.0       
        f15_1['gear/unit[0]/pos-norm'] = 0.0   
        f15_1['gear/unit[1]/pos-norm'] = 0.0   
        f15_1['gear/unit[2]/pos-norm'] = 0.0   
    
        self.fdms["agent_1"] = f15_1
        
        # Alt sınıfa (Task) ait özel reset işlemlerini (hafıza vb.) tetikle
        self._task_reset()
        
        observations = {agent: self._get_obs(agent) for agent in self.agents}
        
        infos = {agent: self._get_info(agent) for agent in self.agents}
        infos = self._update_info(infos)
        
        return observations, infos

    def step(self, actions):
        if not actions:
            return {}, {}, {}, {}, {}

        # 1. Komutları Fiziğe İlet
        for agent_id, action in actions.items():
            fdm = self.fdms[agent_id]
            fdm['fcs/aileron-cmd-norm'] = float(action[0])
            fdm['fcs/elevator-cmd-norm'] = float(action[1])
            throttle = float((action[2] + 1.0) / 2.0)
            fdm['fcs/throttle-cmd-norm[0]'] = throttle
            fdm['fcs/throttle-cmd-norm[1]'] = throttle

        # 2. 8 Frame Fizik Simülasyonu
        for _ in range(8):
            self.fdms["agent_1"].run()

        # 3. Alt Sınıftan (Task) Ödül ve Bitiş Durumlarını Al
        rewards, terminations, truncations = self._calculate_rewards_and_dones(actions)

        # Temizlik
        for agent in self.possible_agents:
            if (terminations.get(agent) or truncations.get(agent)) and (agent in self.agents):
                self.agents.remove(agent)

        observations = {agent: self._get_obs(agent) for agent in self.agents}

        infos = {agent: self._get_info(agent) for agent in self.agents}
        infos = self._update_info(infos)

        self._render_frame()

        return observations, rewards, terminations, truncations, infos

    def _render_frame(self):
        if self.render_mode == "panda" and self.viewer is not None:
            p1_state = {
                'x': (self.fdms["agent_1"]['position/long-gc-deg'] - (-122.3749)) * 100000,
                'y': (self.fdms["agent_1"]['position/lat-geod-deg'] - 37.6190) * 100000,
                'z': self.fdms["agent_1"]['position/h-sl-ft'],
                'roll': self.fdms["agent_1"]['attitude/phi-deg'],
                'pitch': self.fdms["agent_1"]['attitude/theta-deg'],
                'yaw': self.fdms["agent_1"]['attitude/psi-deg']
            }
            p2_dummy_state = {'x': 0.0, 'y': 0.0, 'z': -50000.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}
            self.viewer.update_world(p1_state, p2_dummy_state)
    
    def _get_info(self, agent_id):
        """Debug modundaysa saf telemetri verilerini infos sözlüğüne doldurur."""
        if self.render_mode == "debug":
            fdm = self.fdms[agent_id]
            return {
                "lat": fdm['position/lat-geod-deg'],
                "lon": fdm['position/long-gc-deg'],
                "alt_ft": fdm['position/h-sl-ft'],
                "alt_m": fdm['position/h-sl-ft'] * 0.3048,
                "roll_deg": fdm['attitude/phi-deg'],
                "pitch_deg": fdm['attitude/theta-deg'],
                "yaw_deg": fdm['attitude/psi-deg'],
                "airspeed_kts": fdm['velocities/vc-kts'],
                "airspeed_ms": fdm['velocities/vc-kts'] * 0.514444
            }
        
        # Debug modunda değilsek FPS'i korumak için boş sözlük dön
        return {}

    def _get_initial_conditions(self):
        raise NotImplementedError

    def _task_reset(self):
        pass # Opsiyonel, sadece ihtiyaç varsa ezilir

    def _get_obs(self, agent_id):
        raise NotImplementedError

    def _calculate_rewards_and_dones(self):
        raise NotImplementedError
    
    def _update_info(self, infos: dict):
        return infos