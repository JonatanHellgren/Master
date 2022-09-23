from typing import NamedTuple

class EnvParams(NamedTuple):
    """
    Object for storing environment parameters
    """
    # Parameter  : Type      Example
    size         : tuple   # (10, 10)
    n_foods      : int     # 15
    n_food_types : int     # 3
    objective    : int     # 3 for small and 6 for medium
    n_test       : int     # 100
    is_stochastic: bool
    is_pomdp     : bool
