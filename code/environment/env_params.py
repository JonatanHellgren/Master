from typing import NamedTuple

class EnvParams(NamedTuple):
    # Parameter    Type    # Example
    size         : tuple   # (10, 10)
    n_foods      : int     # 15
    n_food_types : int     # 3
    n_test       : int     # 100
    stochastic   : bool
