from aima_probability import *
from project_bn.mpe_ask import mpe_ask
from bif_serializer import parse_bif_spec
import os


def main():
    # dir_path = os.path.dirname(os.path.realpath(__file__))
    # print(dir_path)
    net = BayesNet(parse_bif_spec("resources/earthquake.xml"))
    # burglary_ = burglary
    ris = mpe_ask(dict(Alarm=True), net)

    print(ris.show_approx())


if __name__ == '__main__':
    main()
