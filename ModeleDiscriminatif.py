import csv

import numpy as np

from Modele import Modele


class ModeleDiscriminatif(Modele):
    """
    Modele discriminatif de classification binaire
    """

    def __init__(self):
        Modele.__init__(self)
        self.beta = np.array(0)
        return

    def read_data(self, filename, sep='\t'):
        file_reader = csv.reader(open(filename), delimiter=sep)
        file_content = [[float(xa), float(xb), float(y)] for (xa, xb, y) in file_reader]
        # noinspection PyPep8Naming
        XY = np.array(file_content)
        # noinspection PyPep8Naming
        X_intercept = np.linspace(1, 1, len(XY))
        self.X = np.column_stack((X_intercept, XY[:, :-1]))
        self.Y = XY[:, -1]
