# AI Olympics Agent Trainer

A 2023 CS MSc Advanced AI team project for training and submitting agents to the IJCAI AI Olympics integrated environment. The recovered work combines a reusable PPO trainer, DQN and dueling-DQN experiments, a visual game classifier, game-specific policy dispatch, and final trained submission weights.

| Material | Classification | Notes |
| --- | --- | --- |
| `env/`, `olympics_engine/`, `utils/`, `agents/random/`, assets, and original documentation | Supplied framework | Preserved from the `jidiai/Competition_IJCAI2023` framework in the first commit. |
| `agents/dueling_dqn/` and `agents/game_classifier/` | Coauthored solution | Final submission agents and trained weights recovered from the team snapshot. |
| `rl_trainer/agent_trainer/` | Coauthored solution | Newest recovered reusable trainer and model/environment adapters. |
| `rl_trainer/rl-algo-competition/` and modified reference-trainer files | Coauthored experiments | Earlier team experiments retained for completeness; they are not the canonical run path. |
| `artifacts/ppo-vs-random-episode-9900/` | Curated generated result | Latest actor/critic pair selected from hundreds of intermediate checkpoints. |
| `docs/upstream-framework-readme.md` and `LICENSE` | Supplied documentation | Original framework README and MIT license. |

The July 2, 2023 snapshot is the newest recovered copy. It was compared by content with the June 20 competition package and the May `train` and `Project` directories. The older copies contain no later unique source. An older nested classifier ZIP was extracted and compared; its classifier weights match, but its submission source predates the retained folder.

Generated virtual environments, caches, local editor files, evaluation logs, TensorBoard event files, duplicate archives, and roughly 850 MB of superseded checkpoints are intentionally omitted. They are not source and are not required by the documented run path.

## What Is Included

- A configurable PPO-versus-random trainer for running, football, wrestling, and table hockey
- DQN and dueling-DQN agent implementations and a final running-agent checkpoint
- A convolutional game classifier that dispatches observations to game-specific PPO or random policies
- Final classifier, football actor, and running actor weights used by the recovered submission
- The supplied AI Olympics engine, wrappers, assets, and reference trainers needed to run the project
- One curated latest PPO actor/critic checkpoint from the recovered 9,900-episode run

## Requirements

- Python 3.10
- [`uv`](https://docs.astral.sh/uv/)
- A CPU is sufficient for tests and inference; CUDA is optional for training

## Setup

```bash
git clone https://github.com/KobieHazon/msc-advanced-ai-olympics-agent-trainer.git
cd msc-advanced-ai-olympics-agent-trainer
uv sync --dev
```

## Run

Run the final classifier agent against the supplied random agent:

```bash
SDL_VIDEODRIVER=dummy uv run python run_log.py \
  --my_ai game_classifier \
  --opponent random
```

Start a short PPO training run:

```bash
SDL_VIDEODRIVER=dummy uv run python -m rl_trainer.agent_trainer.agent_trainer \
  --game_name running \
  --train_model ppo \
  --enemy_model random \
  --max_episodes 10 \
  --episode_max_len 500
```

Training is stochastic and expensive. The command demonstrates the canonical recovered entry point; it does not reproduce the original long run in a few minutes.

## Validate

```bash
SDL_VIDEODRIVER=dummy uv run pytest
uv run ruff check .
uv run ruff format --check .
```

The tests load the recovered model weights, exercise both final submission controllers, validate the trainer factories and argument handling, and smoke-test an environment reset and step without rendering a window.

## Authors

- Kobie Hazon
- Ron Ben Shimol
- Shahar Bend

The recovered files do not establish a reliable per-line division of work, so the team-authored material is credited jointly.

## Known Limitations

- The full historical training run was not reproduced; validation covers inference, model loading, factories, and a short environment smoke path.
- `agent_trainer.py` is the newest canonical trainer. `agent_trainer_run.py` is retained as an earlier incomplete experiment.
- The historical Ray/RLlib experiment imports an obsolete Ray API and is preserved as source evidence, not advertised as a supported entry point.
- PyTorch checkpoints should only be loaded from trusted sources. The maintained loaders request weights-only deserialization.

## Rights

The supplied AI Olympics framework retains its MIT license and copyright notice in `LICENSE`. See `NOTICE.md` for the status of the team-authored coursework additions.
