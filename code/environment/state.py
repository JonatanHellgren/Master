from typing import NamedTuple

import numpy as np

class State(NamedTuple):
    """
    Class to hold all information about the state in
    """
    grid: np.ndarray
    obs: np.ndarray
    n_objectives: int
    n_side_effects: int
