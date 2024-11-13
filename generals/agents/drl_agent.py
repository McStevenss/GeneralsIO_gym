from stable_baselines3 import ppo, PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common import callbacks
from stable_baselines3.common import policies
from .agent import Agent
from generals.core.game import Action, Observation
import numpy as np

class RL_agent(Agent):
    def __init__(self, id: str = "DeepRL", color: tuple[int, int, int] = (133, 37, 69), model_path:str = "/home/mans/personal_projects/RL_generals/logs/models/generals_2048/best_model.zip"):
        super().__init__(id, color)

        self.model_path = model_path
        self.color = color

        self.model = PPO.load(self.model_path)

        # self.model = PPO.load(self.model_path)

    def load_model(self, env):
        self.model = PPO.load(self.model_path,env)

        self.model.observation_space = env.observation_space
  
    def act(self, observation: Observation):

        _observation = observation["observation"] if "observation" in observation else observation

        if isinstance(_observation, dict):
            # Convert the dict into a format suitable for the policy
            _observation = {key: np.expand_dims(value, axis=0) for key, value in _observation.items()}


        action, _states = self.model.predict(observation=_observation, deterministic=True)

        if len(action) == 1:
            action = action[0]
        if isinstance(action, np.ndarray):
            cell_x, cell_y, direction, pass_turn, split = action

            action = {
                "pass": 0, 
                "cell": np.array([cell_x, cell_y]),
                "direction": direction,
                "split": split
            }

        print(action)
        return action
    

    def reset(self):
        pass