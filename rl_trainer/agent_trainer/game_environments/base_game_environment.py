from abc import ABCMeta, abstractmethod


class BaseGameEnvironment(metaclass=ABCMeta):
    @abstractmethod
    def render(self):
        pass

    @abstractmethod
    def reset(self):  # TODO: add return typing, their different
        pass

    @abstractmethod
    def get_agent_data(
        self, agent_index: int
    ):  # look at create_scenario to understand what kind of agent data exists
        pass

    def get_shaped_reward(self, step_reward: int, is_done: bool) -> int:
        """Default terminal-only reward used by environments without shaping."""
        return step_reward if is_done else 0
