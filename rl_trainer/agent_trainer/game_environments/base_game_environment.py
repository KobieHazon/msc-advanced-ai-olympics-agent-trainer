from abc import ABCMeta, abstractmethod
from typing import List


class BaseGameEnvironment(metaclass=ABCMeta):
    @abstractmethod
    def render(self):
        pass

    @abstractmethod
    def reset(self):  # TODO: add return typing, their different
        pass

    @abstractmethod
    def get_agent_data(self, agent_index: int):  # look at create_scenario to understand what kind of agent data exists
        pass
