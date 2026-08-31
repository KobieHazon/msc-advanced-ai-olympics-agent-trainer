from olympics_engine.generator import create_scenario
from olympics_engine.scenario import wrestling
from .base_game_environment import BaseGameEnvironment

MAX_REWARD = 100


class WrestlingGameEnvironment(BaseGameEnvironment):
    def __init__(self):
        self.game_scenario = create_scenario('wrestling')
        self.game_env = wrestling(self.game_scenario)

    def render(self):
        self.game_env.render()

    def reset(self):
        return self.game_env.reset()

    def get_agent_data(self, agent_index: int):
        return self.game_env.agent_list[agent_index]

    def get_shaped_reward(self, step_reward: int, is_done: int):
        if is_done and step_reward == 1:
            return MAX_REWARD
        if is_done and step_reward == 0:
            return -1 * MAX_REWARD
        return 0

    def __getattr__(self, item):
        return getattr(self.game_env, item)
