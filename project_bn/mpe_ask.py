from collections import Counter
from typing import List

from aima_probability import BayesNet, Factor, make_factor
from project_bn.maxout_factor import MaxoutFactor


def mpe_ask(e, bn: BayesNet):
    factors = [make_factor(v, e, bn) for v in bn.variables]
    return mpe_ask_factors(factors, bn)


def mpe_ask_factors(factors: List[Factor], bn):
    factors = [MaxoutFactor(f) for f in factors]
    var_count = Counter(var for f in factors for var in f.variables)

    assert len(factors) >= 1

    while factors:
        factor = factors.pop()

        for var in factor.variables:
            # Se la variabile è contenuta solo in quel fattore, posso fare
            # il maxout.
            if var_count[var] == 1:
                factor = factor.max_out(var, bn)
                var_count[var] -= 1

        if factors:
            to_multiply = factors.pop()
            var_count -= Counter(
                set(factor.variables).intersection(to_multiply.variables)
            )
            factor = factor.pointwise_product(to_multiply, bn)
            factors.append(factor)
        else:
            break

    # noinspection PyUnboundLocalVariable
    return factor.previous_assignments[()], factor.cpt[()]
