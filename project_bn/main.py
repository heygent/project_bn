from aima_probability import *
from project_bn.mpe_ask import mpe_ask
from project_bn.map_ask import map_ask
from bif_serializer import parse_bif_spec
from project_bn.benchmark import benchmark
import os


def main():
    # dir_path = os.path.dirname(os.path.realpath(__file__))
    # print(dir_path)
    net = BayesNet(parse_bif_spec("resources/earthquake.xml"))

    # burglary_ = burglary

    with benchmark("ciao") as bench:
        ris = mpe_ask(dict(Alarm=True), net)

    for pars in param:
        mpe_ask(*pars)
        map_ask()

    print(ris.show_approx())


if __name__ == '__main__':
    main()
