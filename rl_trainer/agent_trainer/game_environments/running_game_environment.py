from olympics_engine.generator import create_scenario
from olympics_engine.scenario import Running_competition
from .base_game_environment import BaseGameEnvironment


class RunningGameEnvironment(BaseGameEnvironment):

    def __init__(self, map_id: int = 1):  # TODO: change to a random map
        self.running_scenario = create_scenario('running-competition')
        self.running_env = Running_competition(meta_map=self.running_scenario, map_id=map_id, vis=200, vis_clear=5,
                                               agent1_color='light red',
                                               agent2_color='blue')

    def render(self):
        self.running_env.render()

    def reset(self):
        return self.running_env.reset()

    def get_shaped_reward(self, step_reward: int, is_done: int):
        if not is_done:
            return 0
        else:
            if step_reward == 1:
                return step_reward
            else:
                return step_reward - 100

    def get_agent_data(self, agent_index: int):
        return self.running_env.agent_list[agent_index]

    def __getattr__(self, item):
        return getattr(self.running_env, item)
