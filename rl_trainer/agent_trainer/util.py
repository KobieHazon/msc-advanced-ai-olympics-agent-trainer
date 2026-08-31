from dataclasses import dataclass

import numpy as np


@dataclass
class Transition:
    state: np.array
    action: int
    a_log_prob: int
    reward: int
    next_state: np.array
    is_done: bool
