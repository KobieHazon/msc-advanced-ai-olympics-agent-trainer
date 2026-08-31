import argparse
import logging
from dataclasses import dataclass

SUPPORTED_GAMES = {'running', 'table-hockey', 'football', 'wrestling'}
SUPPORTED_AGENT_MODELS = {'ppo', 'random'}

logger = logging.getLogger(__name__)


@dataclass
class AgentTrainerArgs:
    game_name: str
    train_model: str
    load_train_model: bool
    train_model_path: str

    enemy_model: str
    load_enemy_model: bool
    enemy_model_path: str

    max_episodes: int
    episode_max_len: int

    torch_seed: int

    save_interval: int
    render_graphic: bool


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--game_name', default="running", type=str,
                        help='/'.join(SUPPORTED_GAMES))
    parser.add_argument('--train_model', default="ppo", type=str, help='/'.join(SUPPORTED_AGENT_MODELS))
    parser.add_argument("--load_train_model", action='store_true')
    parser.add_argument("--train_model_path", type=str)

    parser.add_argument('--enemy_model', default="random", type=str, help='/'.join(SUPPORTED_AGENT_MODELS))
    parser.add_argument("--load_enemy_model", action='store_true')
    parser.add_argument("--enemy_model_path", type=str)

    parser.add_argument('--max_episodes', default=100000, type=int)
    parser.add_argument('--episode_max_len', default=500, type=int)

    parser.add_argument('--torch_seed', default=1, type=int)

    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--render_graphic", action='store_true')
    return parser


def parse_args() -> AgentTrainerArgs:
    arg_parser = create_argument_parser()
    args = arg_parser.parse_args()
    agent_trainer_args = AgentTrainerArgs(
        game_name=args.game_name,
        train_model=args.train_model,
        load_train_model=args.load_train_model,
        train_model_path=args.train_model_path,
        enemy_model=args.enemy_model,
        load_enemy_model=args.load_enemy_model,
        enemy_model_path=args.enemy_model_path,
        max_episodes=args.max_episodes,
        episode_max_len=args.episode_max_len,
        torch_seed=args.torch_seed,
        save_interval=args.save_interval,
        render_graphic=args.render_graphic,
    )
    logger.info(f"Agent Trainer args are:\n{str(agent_trainer_args)}")
    return agent_trainer_args
