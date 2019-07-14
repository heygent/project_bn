from aima_probability import *
from project_bn.custom_factor import *


def mpe(X, e, bn):
    assert X not in e, "Query variable must be distinct from evidence"
    factors = []
    for var in reversed(bn.variables):
        factors.append(make_custom_factor(var, e, bn))
        if is_hidden(var, X, e):
            factors = max_out(var, factors, bn)
    return pointwise_product(factors, bn).normalize()


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

