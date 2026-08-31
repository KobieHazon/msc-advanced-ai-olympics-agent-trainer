from olympics_engine.generator import create_scenario
from olympics_engine.scenario import football
from .base_game_environment import BaseGameEnvironment


class FootballGameEnvironment(BaseGameEnvironment):
    def __init__(self):
        self.game_scenario = create_scenario('football')
        self.game_env = football(self.game_scenario)

    def render(self):
        self.game_env.render()

    def reset(self):
        return self.game_env.reset()

    def get_agent_data(self, agent_index: int):
        return self.game_env.agent_list[agent_index]

    def get_shaped_reward(self, step_reward: int, is_done: int):
        if not is_done:
            return 0
        else:
            if step_reward == 1:
                return step_reward
            else:
                return step_reward - 100

    def __getattr__(self, item):
        return getattr(self.game_env, item)
