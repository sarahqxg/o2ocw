import numpy as np
import pytest

from o2o.envs import ENV_NAMES, make_env


@pytest.mark.parametrize("name", ENV_NAMES)
def test_env_builds_and_steps(name):
    env = make_env(name, seed=0)
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    for _ in range(50):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        if term or trunc:
            obs, info = env.reset()
    env.close()


@pytest.mark.parametrize("name", ENV_NAMES)
def test_env_is_deterministic_under_seed(name):
    trajectories = []
    for _ in range(2):
        env = make_env(name, seed=123)
        obs, _ = env.reset(seed=123)
        traj = [obs]
        for _ in range(20):
            obs, r, term, trunc, info = env.step(env.action_space.sample())
            if term or trunc:
                obs, info = env.reset()
            traj.append(obs)
        env.close()
        trajectories.append(np.stack(traj))
    np.testing.assert_array_equal(trajectories[0], trajectories[1])


def test_unknown_env_rejected():
    with pytest.raises(ValueError):
        make_env("Pendulum-v1", seed=0)
