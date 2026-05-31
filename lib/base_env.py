from pettingzoo import ParallelEnv
from gymnasium import spaces
import jsbsim
from functools import lru_cache
from pathlib import Path
from typing import List, Dict

jsbsim.FGJSBBase().debug_lvl = 0

class BaseEnv(ParallelEnv):
    metadata = {"render_modes": ["debug", "human", "none"], "name": "base_env"}

    def __init__(self, number_of_agents, render_mode="none", aircraft="f16"):
        super().__init__()
        
        self.render_mode = render_mode
        self.aircraft = aircraft
        self.number_of_agents = number_of_agents

        self.possible_agents = [f"agent_{idx + 1}" for idx in range(number_of_agents)]
        self.agents = self.possible_agents[:]
        
        # (aileron, elevator, throttle), all three in range (-1.0, 1.0)
        self.action_spaces = {
            agent: spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=float)
            for agent in self.possible_agents
        }
        
        # That will be determined by the subclass.
        self.observation_spaces = {}

        # flight dynamics models, we will have fdms for each agent.
        self.fdms = {}
    
    @lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.fdms = {}
        
        # Take the initial conditions from the Task subclass. That gives us a vector of conditions for each agent in the env.
        initial_conditions_list: Dict[Dict] = self._get_initial_conditions()
        
        # that will fill the fdms dictionary for each agent.
        for agent in self.agents:

            fdm = jsbsim.FGFDMExec(None)
            initial_conditions = initial_conditions_list[agent]

            if self.render_mode == "human":
                config_path = Path(__file__).resolve().parent.parent / 'config' / f'{agent}_protocol.xml'
                config_path_str = str(config_path)
                fdm.set_output_directive(config_path_str)
                
            fdm.load_model(self.aircraft)
            fdm['ic/lat-geod-deg'] = initial_conditions.get("lat", 37.6190)
            fdm['ic/long-gc-deg'] = initial_conditions.get("long", -122.3749)
            
            
            fdm['ic/h-sl-ft'] = initial_conditions['alt']
            fdm['ic/vc-kts'] = initial_conditions['vc']
            fdm['ic/theta-deg'] = initial_conditions['pitch']
            fdm['ic/phi-deg'] = initial_conditions['roll']
            fdm['ic/psi-true-deg'] = initial_conditions['yaw']
            
            if self.aircraft == "f15":
                fdm['propulsion/engine[0]/set-running'] = 1
                fdm['propulsion/engine[1]/set-running'] = 1
                fdm.run_ic()
                
                fdm['fcs/gear-cmd-norm'] = 0.0       
                fdm['gear/unit[0]/pos-norm'] = 0.0   
                fdm['gear/unit[1]/pos-norm'] = 0.0   
                fdm['gear/unit[2]/pos-norm'] = 0.0   

            elif self.aircraft == "f16":
                fdm['propulsion/engine[0]/set-running'] = 1
                fdm.run_ic()

                fdm['fcs/gear-cmd-norm'] = 0.0      
                fdm['gear/unit[0]/pos-norm'] = 0.0   
                fdm['gear/unit[1]/pos-norm'] = 0.0   
                fdm['gear/unit[2]/pos-norm'] = 0.0   
        
            self.fdms[agent] = fdm
        
        # subclass task reset method.
        self._task_reset()
        
        observations = {agent: self._get_obs(agent) for agent in self.agents}
        
        infos = {agent: self._get_info(agent) for agent in self.agents}
        infos = self._update_info(infos)
        
        return observations, infos

    def step(self, actions: Dict[str, List[int]]):
        if not actions:
            return {}, {}, {}, {}, {}

        # Take the actions.        
        for agent, action in actions.items():
            fdm = self.fdms[agent]
            fdm['fcs/aileron-cmd-norm'] = float(action[0])
            fdm['fcs/elevator-cmd-norm'] = float(action[1])
            throttle = float((action[2] + 1.0) / 2.0)
            fdm['fcs/throttle-cmd-norm[0]'] = throttle
            if self.aircraft == 'f15':
                fdm['fcs/throttle-cmd-norm[1]'] = throttle

        # Wait for 8 simulation step.
        for _ in range(8):
            for agent_id, fdm in self.fdms.items():
                fdm.run()

        # Take rewards, terminations, truncations from the subclass.
        rewards, terminations, truncations = self._calculate_rewards_and_dones(actions)

        # Apply terminanations and truncations.
        for agent in self.possible_agents:
            if (terminations.get(agent) or truncations.get(agent)) and (agent in self.agents):
                self.agents.remove(agent)

        # Get observations from the subclass.
        observations = {agent: self._get_obs(agent) for agent in self.agents}

        # Get extra informations from the subclass.
        infos = {agent: self._get_info(agent) for agent in self.agents}
        infos = self._update_info(infos)

        return observations, rewards, terminations, truncations, infos
    
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
    
    def _get_possible_agents(self):
        return self.possible_agents

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