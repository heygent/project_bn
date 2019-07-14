from aima_probability import *
from bif_serializer import parse_bif
from project_bn.maxout_factor import *


def stupid_mpe_ask(e, bn: BayesNet):
    factors = [
        make_maxout_factor(var, e, bn) for var in reversed(bn.variables)
    ]
    for var in bn.variables:
        factors = max_out(var, factors, bn)

    return factors[0]


def mpe_ask(e, bn: BayesNet):
    factors = []
    visited = set()
    factors_vars = set()

    for var in reversed(bn.variables):
        factors.append(make_maxout_factor(var, e, bn))
        factors_vars.update(factors[-1].variables)
        visited.add(var)

        if visited.issuperset(factors_vars):
            for var in visited.intersection(factors_vars):
                factors = max_out(var, factors, bn)


def mpe_ask2(e, bn: BayesNet):

    factor = make_maxout_factor(bn.variables[-1], e, bn)
    multiplied = {bn.variables[-1]}
    to_max_out = set(bn.variables) - e.keys()

    for var in reversed(bn.variables[:-1]):

        new_factor = make_maxout_factor(var, e, bn)
        factor = factor.pointwise_product(new_factor, bn)

        multiplied.add(var)

        if multiplied.issuperset(factor.variables):
            for v in multiplied.intersection(factor.variables, to_max_out):
                if len(to_max_out) == 1:
                    break
                factor = factor.max_out(v, bn)
                to_max_out.remove(v)
        if len(to_max_out) == 1:
            break

    return factor


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


if __name__ == "__main__":
    net = BayesNet(parse_bif("resources/earthquake.xml"))
    mpe_ask2(dict(Alarm=True), net)
