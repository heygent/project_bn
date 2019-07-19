from collections import Counter
from typing import List

from aima_probability import (
    BayesNet,
    Factor,
    make_factor,
    pointwise_product,
    is_hidden,
)
from project_bn.maxout_factor import MaxoutFactor, make_maxout_factor


# def mpe_ask(e, bn: BayesNet):
#     factors = [make_factor(v, e, bn) for v in bn.variables]
#     return mpe_ask_factors(factors, bn)


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

#
# def mpe_ask(e, bn: BayesNet):
#     factors = make_maxout_factor(bn.variables[-1], e, bn)
#     var_count = Counter(var for f in factors for var in f.variables)
#     multiplied = {bn.variables[-1]}
#     to_max_out = set(bn.variables) - e.keys()
#
#     for var in reversed(bn.variables[:-1]):
#
#         new_factor = make_maxout_factor(var, e, bn)
#         factor = factor.pointwise_product(new_factor, bn)
#
#         multiplied.add(var)
#
#         if multiplied.issuperset(factor.variables):
#             for v in multiplied.intersection(factor.variables, to_max_out):
#                 if len(to_max_out) == 1:
#                     break
#                 factor = factor.max_out(v, bn)
#                 to_max_out.remove(v)
#         if len(to_max_out) == 1:
#             break
#
#     return factor.previous_assignments[()], factor.cpt[()]


def max_out(var, factors, bn):
    """Eliminate var from all factors by summing over its values."""
    result, var_factors = [], []
    for f in factors:
        if var in f.variables:
            var_factors.append(f)
        else:
            result.append(f)
    result.append(pointwise_product(var_factors, bn).max_out(var, bn))
    return result


def mpe_ask(e, bn):
    factors = [
        make_maxout_factor(var, e, bn) for var in reversed(bn.variables)
    ]
    to_max_out = [var for var in reversed(bn.variables) if var not in e]

    for var in to_max_out:
        factors = max_out(var, factors, bn)

    factor = factors[0]
    return factor.previous_assignments[()], factor.cpt[()]
