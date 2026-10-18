"""Environment construction for the two coursework environments."""
import gymnasium as gym

ENV_NAMES = ("CartPole-v1", "LunarLander-v3")


def make_env(name: str, seed: int, render: bool = False) -> gym.Env:
    """Build a seeded env. Rendering is off unless explicitly requested."""
    if name not in ENV_NAMES:
        raise ValueError(f"Unknown env {name!r}; expected one of {ENV_NAMES}.")
    env = gym.make(name, render_mode="rgb_array" if render else None)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env
