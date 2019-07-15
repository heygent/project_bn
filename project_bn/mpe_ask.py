from collections import Counter
from typing import List

from bif_serializer import parse_bif_spec
from aima_probability import BayesNet
from project_bn.maxout_factor import make_maxout_factor, MaxoutFactor


def mpe_ask(e, bn: BayesNet):
    factors = [make_maxout_factor(v, e, bn) for v in bn.variables]
    return mpe_ask_factors(factors, bn)


def mpe_ask_factors(factors: List[MaxoutFactor], bn):
    factors = factors.copy()
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

    return factor.previous_assignments[()], factor.cpt[()]


if __name__ == "__main__":
    net = BayesNet(parse_bif_spec("resources/earthquake.xml"))
    mpe_ask(dict(JohnCalls=True, MaryCalls=True), net)
