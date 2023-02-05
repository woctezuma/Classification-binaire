import csv
from pathlib import Path

import numpy as np

from Modele import Modele


class ModeleGeneratif(Modele):
    """Modele génératif pour la classification binaire."""

    def __init__(self):
        Modele.__init__(self)
        self.pi = 0
        self.mu1 = np.array(0)
        self.mu0 = np.array(0)
        return

    def read_data(self, filename, sep='\t'):
        with Path(filename).open() as f:
            file_reader = csv.reader(f, delimiter=sep)
            file_content = [
                [float(xa), float(xb), float(y)] for (xa, xb, y) in file_reader
            ]
        # noinspection PyPep8Naming
        XY = np.array(file_content)
        self.X = XY[:, :-1]
        self.Y = XY[:, -1]
