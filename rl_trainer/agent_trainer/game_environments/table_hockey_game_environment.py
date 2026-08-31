from olympics_engine.generator import create_scenario
from olympics_engine.scenario import table_hockey
from .base_game_environment import BaseGameEnvironment


class TableHockeyGameEnvironment(BaseGameEnvironment):
    def __init__(self):
        self.game_scenario = create_scenario('table-hockey')
        self.game_env = table_hockey(self.game_scenario)

    def render(self):
        self.game_env.render()

    def reset(self):
        return self.game_env.reset()

    def get_agent_data(self, agent_index: int):
        return self.game_env.agent_list[agent_index]

    def __getattr__(self, item):
        return getattr(self.game_env, item)
