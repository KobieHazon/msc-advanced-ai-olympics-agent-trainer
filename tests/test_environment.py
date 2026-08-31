from rl_trainer.agent_trainer.game_environments.running_game_environment import (
    RunningGameEnvironment,
)


def test_running_environment_reset_and_step():
    environment = RunningGameEnvironment(map_id=1)

    initial_state = environment.reset()
    next_state, reward, done, info = environment.step([[0, 0], [0, 0]])

    assert len(initial_state) == 2
    assert len(next_state) == 2
    assert len(reward) == 2
    assert isinstance(done, bool)
    assert info is not None
