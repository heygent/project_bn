from aima_probability import *


def create_random_evidence(N: int, net: BayesNet, percentage):

    keys = random.sample(net.variables, N)
    real_percentage = (N // 100) * percentage
    evidences_keys = keys[:real_percentage]
    map_keys = keys[real_percentage:]
    evidences = {}
    for key in evidences_keys:
        domain = net.variable_values(key)
        evidences[key] = random.choice(domain)

    return evidences, map_keys
