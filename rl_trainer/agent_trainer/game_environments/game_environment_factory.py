from typing import Set

from .football_game_environment import FootballGameEnvironment
from .running_game_environment import RunningGameEnvironment
from .table_hockey_game_environment import TableHockeyGameEnvironment
from .wrestling_game_environment import WrestlingGameEnvironment

NAME_TO_CLASS = {
    "running": RunningGameEnvironment,
    "football": FootballGameEnvironment,
    "wrestling": WrestlingGameEnvironment,
    "table-hockey": TableHockeyGameEnvironment,
}


class GameEnvironmentFactory:
    def __init__(self, games: Set[str]):
        if not games.issubset(NAME_TO_CLASS.keys()):
            raise ValueError('Unsupported game registered to factory')
        self.games = games

    def create_game_environment(self, name: str, *args, **kwargs):
        if name not in self.games:
            raise ValueError('Unsupported game environment')
        game_environment_class = NAME_TO_CLASS[name]
        return game_environment_class(*args, **kwargs)
