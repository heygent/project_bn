from aima_probability import *

__all__ = ["MaxoutFactor", "make_maxout_factor"]


class MaxoutFactor(Factor):
    """A factor in a joint distribution."""

    def __init__(self, variables, cpt, previous_assignments=None):
        super().__init__(variables, cpt)
        self.previous_assignments = previous_assignments or {
            key: {} for key in cpt.keys()
        }

    def pointwise_product(self, other: "MaxoutFactor", bn) -> "MaxoutFactor":
        """Multiply two factors, combining their variables."""
        variables, cpt = self._pointwise_product(other, bn)
        assignments = {
            event_values(e, variables): {
                **self.previous_assignments[event_values(e, self.variables)],
                **other.previous_assignments[event_values(e, other.variables)],
            }
            for e in all_events(variables, bn, {})
        }
        return MaxoutFactor(variables, cpt, assignments)

    def sum_out(self, var, bn) -> "MaxoutFactor":
        return MaxoutFactor(*self._sum_out(var, bn))

    def max_out(self, var, bn) -> "MaxoutFactor":
        """Make a factor eliminating var by summing over its values."""
        variables = [X for X in self.variables if X != var]
        cpt = {}
        mpe = {}

        for e in all_events(variables, bn, {}):
            cpt_key = event_values(e, variables)
            max_p = 0
            max_value = None
            for val in bn.variable_values(var):
                val_p = self.p({**e, var: val})
                if max_p < val_p:
                    max_value = val
                    max_p = val_p

            cpt[cpt_key] = max_p
            mpe[cpt_key] = {
                **self.previous_assignments[
                    event_values({**e, var: max_value}, self.variables)
                ],
                var: max_value,
            }

            print("----------")
            print(f"max_value == {max_value}")
            print(f"max_p     == {max_p}")

            print(f"e == {e}")
            print(f"cpt_key == {cpt_key}")
            print(f"mpe[{cpt_key}] == {mpe[cpt_key]}")
            print(f"cpt[{cpt_key}] == {cpt[cpt_key]}")
            print("----------")

        return MaxoutFactor(variables, cpt, mpe)


def make_maxout_factor(var, e, bn: BayesNet) -> "MaxoutFactor":
    """Return the factor for var in bn's joint distribution given e.
    That is, bn's full joint distribution, projected to accord with e,
    is the pointwise product of these factors for bn's variables."""
    node = bn.variable_node(var)
    variables = [X for X in [var] + node.parents if X not in e]
    cpt = {
        event_values(e1, variables): node.p(e1[var], e1)
        for e1 in all_events(variables, bn, e)
    }
    return MaxoutFactor(variables, cpt)
