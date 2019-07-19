from aima_probability import *
from project_bn.mpe_ask import mpe_ask
from project_bn.map_ask import map_ask
from bif_serializer import parse_bif_spec
from project_bn.create_random_evidence import create_random_evidence
from project_bn.benchmark import benchmark

import os


def main():

    # print(
    #     "-------------------------------------------------------------------------------------"
    # )
    # print("CHILD - DOMINIO PICCOLO - EVIDENZE 25% MAP 25% ")
    # net = BayesNet(parse_bif_spec("resources/child.xml"))
    # evidences, map_variables = create_random_evidence(
    #     len(net.nodes), net, 25, 25
    # )
    # print(len(evidences), len(map_variables)), print(len(net.nodes))
    # with benchmark("mpe"):
    #     ris_mpe = mpe_ask(evidences, net)
    # print(ris_mpe)
    # with benchmark("map"):
    #     ris_map = map_ask(map_variables, evidences, net)
    # print(ris_map)
    # print(
    #     "-------------------------------------------------------------------------------------"
    # )

    # print(
    #     "-------------------------------------------------------------------------------------"
    # )
    # print("CHILD - DOMINIO PICCOLO 50% evidenze 10% map")
    # net = BayesNet(parse_bif_spec("resources/child.xml"))
    # evidences, map_variables = create_random_evidence(
    #     len(net.nodes), net, 50, 10
    # )
    # with benchmark("mpe"):
    #     ris_mpe = mpe_ask(evidences, net)
    # print(ris_mpe)
    # with benchmark("map"):
    #     ris_map = map_ask(map_variables, evidences, net)
    # print(ris_map)
    # print(
    #     "-------------------------------------------------------------------------------------"
    # )
    #
    # print(
    #     "-------------------------------------------------------------------------------------"
    # )
    # print("CHILD - DOMINIO PICCOLO 75% evidenze 10% map")
    # net = BayesNet(parse_bif_spec("resources/child.xml"))
    # evidences, map_variables = create_random_evidence(
    #     len(net.nodes), net, 75, 10
    # )
    # with benchmark("mpe"):
    #     ris_mpe = mpe_ask(evidences, net)
    # print(ris_mpe)
    # with benchmark("map"):
    #     ris_map = map_ask(map_variables, evidences, net)
    # print(ris_map)
    # print(
    #     "-------------------------------------------------------------------------------------"
    # )
    #
    # print(
    #     "-------------------------------------------------------------------------------------"
    # )
    # print("CHILD - DOMINIO PICCOLO 10% evidenze 25% map")
    # net = BayesNet(parse_bif_spec("resources/child.xml"))
    # evidences, map_variables = create_random_evidence(
    #     len(net.nodes), net, 10, 25
    # )
    # with benchmark("mpe"):
    #     ris_mpe = mpe_ask(evidences, net)
    # print(ris_mpe)
    # with benchmark("map"):
    #     ris_map = map_ask(map_variables, evidences, net)
    # print(ris_map)
    # print(
    #     "-------------------------------------------------------------------------------------"
    # )
    #
    # print(
    #     "-------------------------------------------------------------------------------------"
    # )
    # print("CHILD - DOMINIO PICCOLO 10% evidenze 50% map")
    # net = BayesNet(parse_bif_spec("resources/child.xml"))
    # evidences, map_variables = create_random_evidence(
    #     len(net.nodes), net, 10, 50
    # )
    # with benchmark("mpe"):
    #     ris_mpe = mpe_ask(evidences, net)
    # print(ris_mpe)
    # with benchmark("map"):
    #     ris_map = map_ask(map_variables, evidences, net)
    # print(ris_map)
    # print(
    #     "-------------------------------------------------------------------------------------"
    # )
    #
    # print(
    #     "-------------------------------------------------------------------------------------"
    # )
    # print("CHILD - DOMINIO PICCOLO 10% evidenze 75% map")
    # net = BayesNet(parse_bif_spec("resources/child.xml"))
    # evidences, map_variables = create_random_evidence(
    #     len(net.nodes), net, 10, 75
    # )
    # with benchmark("mpe"):
    #     ris_mpe = mpe_ask(evidences, net)
    # print(ris_mpe)
    # with benchmark("map"):
    #     ris_map = map_ask(map_variables, evidences, net)
    # print(ris_map)
    # print(
    #     "-------------------------------------------------------------------------------------"
    # )
    #
    # print(
    #     "-------------------------------------------------------------------------------------"
    # )
    # print(
    #     "INSURANCE STESSE EVIDENZE  - DIVERSA COMPLESSITÀ 25% evidenze - 10% map"
    # )
    # net = BayesNet(parse_bif_spec("resources/insurance.xml"))
    # evidences, map_variables = create_random_evidence(
    #     len(net.nodes), net, 25, 10
    # )
    # with benchmark("mpe"):
    #     ris_mpe = mpe_ask(evidences, net)
    # print(ris_mpe)
    # with benchmark("map"):
    #     ris_map = map_ask(map_variables, evidences, net)
    # print(ris_map)
    # print(
    #     "-------------------------------------------------------------------------------------"
    # )
    #
    # print(
    #     "-------------------------------------------------------------------------------------"
    # )
    print(
        "HAILFINDER - DOMINIO MEDIO - STESSE EVIDENZE 25% evidenze - 10% map"
    )
    net = BayesNet(parse_bif_spec("resources/hailfinder.xml"))
    evidences, map_variables = create_random_evidence(
        len(net.nodes), net, 25, 25
    )
    with benchmark("mpe"):
        ris_mpe = mpe_ask(evidences, net)
    print(ris_mpe)
    with benchmark("map"):
        ris_map = map_ask(map_variables, evidences, net)
    print(ris_map)
    print(
        "-------------------------------------------------------------------------------------"
    )
    #
    # print(
    #     "-------------------------------------------------------------------------------------"
    # )
    # print("win95pts - DOMINIO GRANDE - STESSE EVIDENZE 25% evidenze - 10% map")
    # net = BayesNet(parse_bif_spec("resources/win95pts.xml"))
    # evidences, map_variables = create_random_evidence(
    #     len(net.nodes), net, 25, 10
    # )
    # with benchmark("mpe"):
    #     ris_mpe = mpe_ask(evidences, net)
    # print(ris_mpe)
    # with benchmark("map"):
    #     ris_map = map_ask(map_variables, evidences, net)
    # print(ris_map)
    # print(
    #     "-------------------------------------------------------------------------------------"
    # )


if __name__ == "__main__":
    main()
