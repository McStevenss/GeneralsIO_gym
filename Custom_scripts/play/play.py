from stable_baselines3 import ppo, PPO
import torch
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym

from generals.agents import ExpanderAgent

import numpy as np
from stable_baselines3.common.monitor import Monitor

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy, MaskableMultiInputActorCriticPolicy
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback

# model_path = "/home/mans/personal_projects/RL_generals/logs/models/maskable_generals_100/best_model.zip"
# model_path = "/home/mans/personal_projects/RL_generals/logs/models/maskable_generals_250/best_model.zip"
# model_path = "/home/mans/personal_projects/RL_generals/logs/models/maskable_generals_500/final_model.zip"
model_path = "/home/mans/personal_projects/RL_generals/logs/models/maskable_generals_200/final_model.zip"
model = MaskablePPO.load(model_path)
npc = ExpanderAgent()


def mask_fn(env: gym.Env) -> np.ndarray:
    action_mask = env.get_valid_action_mask()
    return action_mask


env = gym.make("gym-generals-simplified-v0", agent_id="DeepRL", npc=npc, render_mode="human")
env = ActionMasker(env, mask_fn)



print(model.observation_space)

obs, _ = env.reset()
# print(obs)
terminated = False
steps = 0
while not terminated:
    # Retrieve current action mask
    action_masks = mask_fn(env)
    action, _states = model.predict(obs, action_masks=action_masks)
    obs, reward, terminated, truncated, info = env.step(action)
    steps += 1
    env.render()
print(f"Game done in {steps} steps")