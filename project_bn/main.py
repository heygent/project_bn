from aima_probability import *
from project_bn.mpe_ask import mpe_ask
from bif_serializer import parse_bif_spec
from project_bn.create_random_evidence import create_random_evidence
import os


def main():


    # confronto  su rete piccola al variare dell'evidenza
    #75%  evidenza
    #CHILD nodi=20 archi = 25 parametri = 230

    net = BayesNet(parse_bif_spec("resources/child.xml"))
    evidences = create_random_evidence(len(net.nodes), net)
    ris = mpe_ask(evidences, net)
    print(ris)



    #confronto



if __name__ == "__main__":
    main()
