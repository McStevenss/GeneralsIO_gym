from collections.abc import Callable
from copy import deepcopy
from typing import Any, SupportsFloat, TypeAlias

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from generals.agents import Agent, AgentFactory
from generals.core.game import Action, Game, Info
from generals.core.grid import GridFactory
from generals.core.observation import Observation
from generals.core.replay import Replay
from generals.gui import GUI
from generals.gui.properties import GuiMode

Reward: TypeAlias = float
RewardFn: TypeAlias = Callable[[Observation, Action, bool, Info], Reward]

class SimplifiedGymnasiumGenerals(gym.Env):
    metadata = {
        "render_modes": ["human"],
        "render_fps": 6,
    }

    def __init__(
        self,
        grid_factory: GridFactory | None = None,
        npc: Agent | None = None,
        agent_id: str | None = None,  # Optional, just to obtain id and color
        agent_color: tuple | None = None,
        truncation: int | None = None,
        reward_fn: RewardFn | None = None,
        render_mode: str | None = None,
        max_length: int | None = None
    ):
        self.render_mode = render_mode
        self.grid_factory = grid_factory if grid_factory is not None else GridFactory()
        self.reward_fn = reward_fn if reward_fn is not None else SimplifiedGymnasiumGenerals._default_reward

        # Agents
        if npc is None:
            print('No NPC agent provided. Creating "Random" NPC as a fallback.')
            npc = AgentFactory.make_agent("random")
        else:
            assert isinstance(npc, Agent), "NPC must be an instance of Agent class."
        self.npc = npc
        self.agent_id = agent_id
        self.agent_color = agent_color
        self.agent_ids = [self.agent_id, self.npc.id]
        self.agent_data = {
            self.agent_id: {"color": (67, 70, 86) if agent_color is None else agent_color},
            self.npc.id: {"color": self.npc.color},
        }
        assert self.agent_id != npc.id, "Agent ids must be unique - you can pass custom ids to agent constructors."

        # Game
        grid = self.grid_factory.grid_from_generator()
        self.game = Game(grid, [self.agent_id, self.npc.id])
        # self.observation_space = self.game.observation_space

        self.observation_space = spaces.Dict({
            'troops_per_cell': spaces.Box(low=0, high=1000, shape=(10, 10), dtype=np.int64),
            'general_cell': spaces.Box(low=0, high=10, shape=(2,), dtype=np.int64),
            'team_cells': spaces.Box(low=0, high=10, shape=(10, 10), dtype=np.int64),
            'opponent_land_count': spaces.Box(low=0, high=100, shape=(1,), dtype=np.int64),
            'owned_land_count': spaces.Box(low=0, high=100, shape=(1,), dtype=np.int64),
        })

        # self.action_space = self.game.action_space
        # self.action_space = spaces.MultiDiscrete([10, 10, 4])  # [cell_x, cell_y, direction, split]
        
        #MaskablePPO takes the sum of dims, so 10,10,4 is not possible, actionspace has to be all possible actions which is 400 (10x10x4)
        self.action_space = spaces.Discrete(400) 

        self.truncation = truncation

        if max_length is None:
            self.MAX_STEPS = 10_000
        elif max_length is not None:
            self.MAX_STEPS = max_length


        self.steps = 0

    def render(self):
        if self.render_mode == "human":
            _ = self.gui.tick(fps=self.metadata["render_fps"])

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)
        if options is None:
            options = {}

        if "grid" in options:
            grid = self.grid_factory.grid_from_string(options["grid"])
        else:
            self.grid_factory.rng = self.np_random
            grid = self.grid_factory.grid_from_generator()

        # Create game for current run
        self.game = Game(grid, self.agent_ids)

        # Create GUI for current render run
        if self.render_mode == "human":
            # self.gui = GUI(self.game, self.agent_data, GuiMode.TRAIN)
            self.gui = GUI(self.game, self.agent_data, GuiMode.TRAIN)

        if "replay_file" in options:
            self.replay = Replay(
                name=options["replay_file"],
                grid=grid,
                agent_data=self.agent_data,
            )
            self.replay.add_state(deepcopy(self.game.channels))
        elif hasattr(self, "replay"):
            del self.replay

        # self.observation_space = self.game.observation_space
        self.observation_space = spaces.Dict({
            'troops_per_cell': spaces.Box(low=0, high=1000, shape=(10, 10), dtype=np.int64),
            'general_cell': spaces.Box(low=0, high=10, shape=(2,), dtype=np.int64),
            'team_cells': spaces.Box(low=0, high=10, shape=(10, 10), dtype=np.int64),
            'opponent_land_count': spaces.Box(low=0, high=100, shape=(1,), dtype=np.int64),
            'owned_land_count': spaces.Box(low=0, high=100, shape=(1,), dtype=np.int64)
            })
        self.action_space = self.game.action_space

        observation = self.game.agent_observation(self.agent_id).as_dict()
        observation = self.convert_to_simplified(observation)
        info: dict[str, Any] = {}

        self.steps = 0
        # print(observation, info)
        return observation, info
    
    def get_valid_action_mask(self):
        observation = self.game.agent_observation(self.agent_id)
        # print(observation.as_dict(with_mask=False))
        action_mask = observation.action_mask()

        # valid_actions = np.argwhere(action_mask == 1)
        # print("Existing valid actions:", len(valid_actions), [action for action in valid_actions])
        #NOTE: FROM REDDIT:
        #So for example, if your action space is with dims [2,2,2]
        #and you want to ignore first action from each dimension
        #you'll have to return following array: [True, False, True, False, True, False].

        #AKA we have 10x10x4, we want to flatten it out to a 400 flat array
        # print(observation.as_dict(), action_mask)
        flattened_action_mask = action_mask.flatten()
        return flattened_action_mask
    
    def convert_action(self,action):
        if type(action) == dict:
            converted_action = action  
        else:
            converted_action = self.decode_action(action)
            # print("converted action:",converted_action)
        return converted_action
    
    def decode_action(self, action):

        # Calculate cell_x, cell_y, and direction from the flattened action
        cell_x = action // (10 * 4)          # 0 to 9
        cell_y = (action // 4) % 10          # 0 to 9
        direction = action % 4               # 0 to 3

        converted_action = {
            "pass":0,
            "cell": np.array([cell_x,cell_y]),
            "direction": direction,
            "split": 0
        }
        return converted_action

    def convert_to_simplified(self,_observation):

        _observation = _observation["observation"] if "observation" in _observation else _observation
        game_map = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)

        cities = np.argwhere(_observation["cities"] == True)
        team1 = np.argwhere(_observation["owned_cells"] == True)
        team2 = np.argwhere(_observation["opponent_cells"] == True)
        generals = np.argwhere(_observation["generals"] == True)
        if len(generals) == 0:
            generals = [[-1, -1]]
        for y, x in cities:
            game_map[y, x] = 9

        for y, x in team1:
            game_map[y, x] = 1

        for y, x in team2:
            game_map[y, x] = 2

        new_observation = {
            'troops_per_cell': np.array(_observation["armies"], dtype=np.int64),
            'team_cells': game_map,
            'owned_land_count': np.array([_observation["owned_land_count"]], dtype=np.int64),
            'opponent_land_count': np.array([_observation["opponent_land_count"]], dtype=np.int64),
            'general_cell': np.array(generals[0], dtype=np.int64)
        }

        return new_observation

    def step(self, action: Action) -> tuple[Observation, SupportsFloat, bool, bool, dict[str, Any]]:
        # Get action of NPC
        npc_observation = self.game.agent_observation(self.npc.id).as_dict()
        npc_action = self.npc.act(npc_observation)
        actions = {self.agent_id: self.convert_action(action), self.npc.id: npc_action}
        observations, infos = self.game.step(actions)

        # From observations of all agents, pick only those relevant for the main agent
        obs = observations[self.agent_id].as_dict()
        obs = self.convert_to_simplified(obs)
        # info = infos[self.agent_id]
        info = infos
        reward = self.reward_fn(obs, action, self.game.is_done(), info)
        terminated = self.game.is_done()
        truncated = False
        if self.truncation is not None:
            truncated = self.game.time >= self.truncation

        if hasattr(self, "replay"):
            self.replay.add_state(deepcopy(self.game.channels))

        if terminated or truncated:
            if hasattr(self, "replay"):
                self.replay.store()

        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            terminated = True

        return obs, reward, terminated, truncated, info

    @staticmethod
    def _default_reward(
        observation: Observation,
        action: Action,
        done: bool,
        info: Info,
    ) -> Reward:
        return 0

    def close(self) -> None:
        if self.render_mode == "human":
            self.gui.close()
