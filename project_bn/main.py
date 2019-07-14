from aima_probability import *
from project_bn.mpe import mpe
from bif_serializer import parse_bif
import os


def main():
    # dir_path = os.path.dirname(os.path.realpath(__file__))
    # print(dir_path)
    net = BayesNet(parse_bif("resources/adder.xml"))
    # burglary_ = burglary
    ris = mpe('Burglary', dict(Alarm=True), net)

    print(ris.show_approx())


if __name__ == '__main__':
    main()
