import os
import sys
from pathlib import Path

import gym
import numpy as np

base_dir = Path(__file__).resolve().parent
sys.path.append(str(base_dir))
from dueling_dqn_agent_submit import DuelingDQNAgent

actions_map = {
    0: [[100], [0]],  # N
    1: [[100], [30]],  # NE
    2: [[100], [-30]],  # NW
    3: [[-100], [0]],  # S
    4: [[-100], [30]],  # SW
    5: [[-100], [-30]],  # SE
}

observation_space = gym.spaces.Box(low=0, high=255, shape=(1, 40, 40), dtype=np.uint8)
observation_space = observation_space.shape

# observation_space = shape=(40, 40, 1)
action_space = len(actions_map)

agent = DuelingDQNAgent(state_space=observation_space,
                        action_space=action_space,
                        max_memory_size=10000,
                        batch_size=128,
                        gamma=0.95,  # Increased from 0.90 to 0.95
                        lr=0.001,
                        dropout=0.2,
                        exploration_max=1.0,
                        exploration_min=0.02,
                        exploration_decay=0.995,  # Decreased from 0.99 to 0.995
                        pretrained=True,
                        is_train=False)

sys.path.pop(-1)  # just for safety


def my_controller(obs_list, action_space_list, obs_space_list):
    state = obs_list['obs']['agent_obs']
    action = agent.act(state)
    action_index = action.data.tolist()
    action_data = actions_map[action_index]
    return action_data
