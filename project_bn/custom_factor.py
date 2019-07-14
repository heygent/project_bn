from aima_probability import *


class CustomFactor:
    """A factor in a joint distribution."""

    def __init__(self, variables, cpt):
        self.variables = variables
        self.cpt = cpt
        #self.max_factor = max_factor

    def pointwise_product(self, other, bn) -> "CustomFactor":
        """Multiply two factors, combining their variables."""
        variables = list(set(self.variables) | set(other.variables))
        cpt = {
            event_values(e, variables): self.p(e) * other.p(e)
            for e in all_events(variables, bn, {})
        }
        return CustomFactor(variables, cpt)

    def max_out(self, var, bn) -> "CustomFactor":
        """Make a factor eliminating var by summing over its values."""
        variables = [X for X in self.variables if X != var]
        cpt = {}
        mpe = {}
        lista_prob = []
        p_max = 0

        for e in all_events(variables, bn, {}):
            cpt_key = (event_values(e, variables))
            # print("e = ", e)
            for val in bn.variable_values(var):
                lista_prob.append(self.p({**e, var: val}))
                # print("val = ", val)
                if p_max < self.p({**e, var: val}):
                    p_max = self.p({**e, var: val})
                    max_value = val
                    mpe[var] = max_value

            cpt_value = max(lista_prob)
            cpt[cpt_key] = cpt_value
            print("eccolo", mpe)



        # cpt = {
        #     event_values(e, variables):
        #     max(self.p({**e, var: val}) for val in bn.variable_values(var))
        #     for e in all_events(variables, bn, {})
        #
        # }

        # print("sua", cpt)

        return CustomFactor(variables, cpt)

    def normalize(self) -> ProbDist:
        """Return my probabilities; must be down to one variable."""
        assert len(self.variables) == 1
        return ProbDist(
            self.variables[0], {k: v for ((k,), v) in self.cpt.items()}
        )

    def p(self, e):
        """Look up my value tabulated for e."""
        return self.cpt[event_values(e, self.variables)]


def make_custom_factor(var, e, bn: BoolBayesNet) -> "CustomFactor":
    """Return the factor for var in bn's joint distribution given e.
    That is, bn's full joint distribution, projected to accord with e,
    is the pointwise product of these factors for bn's variables."""
    node = bn.variable_node(var)
    variables = [X for X in [var] + node.parents if X not in e]
    cpt = {
        event_values(e1, variables): node.p(e1[var], e1)
        for e1 in all_events(variables, bn, e)
    }
    return CustomFactor(variables, cpt)
