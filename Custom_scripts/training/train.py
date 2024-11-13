# from stable_baselines3 import ppo, PPO
import argparse
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

# Initialize agents
npc = ExpanderAgent()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

torch.device
N_EVAL_EPISODES=1
N_EVAL_INTERVAL = 1000
N_STEPS=200
N_TIMESTEPS = 350_000

def mask_fn(env: gym.Env) -> np.ndarray:
    action_mask = env.get_valid_action_mask()
    return action_mask

def reward_function_simplified(
        observation,
        action,
        done,
        info,
    ) -> int:
    agent_id = "DeepRL"
    opponent_id = "Expander"
    reward = 0
    agent_land_factor = observation["owned_land_count"] * 0.01
    opponent_land_factor = observation["opponent_land_count"] * 0.01

    reward += agent_land_factor - opponent_land_factor

    reward += (info[agent_id]["army"] * 0.01) - (info[opponent_id]["army"] * 0.01)

  
    if info[agent_id]["is_winner"]:
        reward += 100
        print("WE WON") 

    if info[opponent_id]["is_winner"]:
        reward -= 10

    return reward


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--continue_train_path', type=str,
        help='path to model file .zip',
        default="",
        required=False
    )
    args = parser.parse_args()

    ######################
    #### Create Envs #####
    ######################
    env = gym.make("gym-generals-simplified-v0", agent_id="DeepRL", npc=npc, reward_fn=reward_function_simplified, max_length=N_STEPS)
    env = ActionMasker(env, mask_fn)

    eval_env = gym.make("gym-generals-simplified-v0", agent_id="DeepRL", npc=npc, reward_fn=reward_function_simplified, max_length=N_STEPS)
    eval_env = ActionMasker(eval_env, mask_fn)

    ############################
    #### Load/Create Model #####
    ############################
    if args.continue_train_path != "":
        print("Loading model supplied in args")
        model = MaskablePPO.load(args.continue_train_path, env=env, device=device, tensorboard_log="logs/tensorboard/finetunes")
    else:
        model = MaskablePPO(MaskableMultiInputActorCriticPolicy,
                            env,
                            verbose=1,
                            n_steps=N_STEPS,
                            batch_size=N_STEPS,
                            n_epochs=10,
                            ent_coef=0.000,
                            learning_rate=0.0003,
                            gamma=0.95,
                            tensorboard_log="logs/tensorboard/",
                            device=device)

    eval_callback = MaskableEvalCallback(
        eval_env=eval_env,
        n_eval_episodes=N_EVAL_EPISODES,
        eval_freq=N_EVAL_INTERVAL,
        best_model_save_path=f'logs/models/maskable_generals_{N_STEPS}',
        deterministic=True
    )

    model.learn(
        total_timesteps=N_TIMESTEPS,
        tb_log_name=f"maskable_ppo_generals_{N_STEPS}_{N_TIMESTEPS}",
        callback=eval_callback,
        progress_bar=True
    )

    model.save(f'logs/models/maskable_generals_{N_STEPS}/final_model.zip')