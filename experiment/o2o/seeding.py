"""Global seeding for reproducible runs."""
import random

import d3rlpy
import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, PyTorch and d3rlpy. Envs are seeded in `o2o.envs.make_env`."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    d3rlpy.seed(seed)
