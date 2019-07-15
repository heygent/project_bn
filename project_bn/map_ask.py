from typing import List, Tuple

from aima_probability import BayesNet, is_hidden, sum_out
from project_bn.maxout_factor import make_maxout_factor
from project_bn.mpe_ask import mpe_ask_factors


def map_ask(Ms: List[str], e: dict, bn: BayesNet) -> Tuple[dict, float]:
    factors = []
    for var in reversed(bn.variables):
        factors.append(make_maxout_factor(var, e, bn))
        if is_hidden(var, Ms, e):
            factors = sum_out(var, factors, bn)

    return mpe_ask_factors(factors, bn)
