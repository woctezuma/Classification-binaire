import math

import numpy as np

from ModeleGeneratif import ModeleGeneratif


class LDA(ModeleGeneratif):
    """
    Linear Discriminant Analysis (LDA)
    """

    def __init__(self):
        ModeleGeneratif.__init__(self)
        self.sigma = np.array(0)
        return

    def set_name(self):
        return 'LDA'

    def train(self, filename):
        self.read_data(filename)
        # noinspection PyPep8Naming
        N1 = sum(self.Y)
        n = len(self.Y)
        self.pi = N1 / n
        self.mu1 = np.array([np.dot(self.X[:, i], self.Y) for i in range(len(self.X[0, :]))])
        self.mu0 = np.array([np.dot(self.X[:, i], (1 - self.Y)) for i in range(len(self.X[0, :]))])
        self.mu1 /= N1
        self.mu0 /= n - N1
        self.sigma = sum([np.outer(z, z)
                          for z in [(self.X[i, :] - self.Y[i] * self.mu1 - (1 - self.Y[i]) * self.mu0)
                                    for i in range(len(self.Y))]])
        self.sigma /= n
        return

    # noinspection PyPep8Naming
    def get_error(self, filename):
        self.read_data(filename)
        beta = np.linalg.solve(self.sigma, self.mu1 - self.mu0)
        temp = np.linalg.solve(self.sigma, self.mu1 + self.mu0)
        gamma = -0.5 * np.dot((self.mu1 - self.mu0).T, temp) + math.log(self.pi / (1 - self.pi))
        Y_pred = np.array(self.Y)
        for i in range(len(self.Y)):
            Y_pred[i] = 1.0 / (1 + math.exp(-gamma - np.dot(self.X[i, :], beta)))
        return sum(abs(self.Y - Y_pred)) / len(self.Y)

    def get_separator(self):
        abscisse = np.linspace(min(self.X[:, -2]), max(self.X[:, -2]))
        valeur = 0.
        beta = np.linalg.solve(self.sigma, self.mu1 - self.mu0)
        temp = np.linalg.solve(self.sigma, self.mu1 + self.mu0)
        gamma = -0.5 * np.dot((self.mu1 - self.mu0).T, temp) + math.log(self.pi / (1 - self.pi))
        ordonnee = (valeur + gamma + beta[-2] * abscisse) / (-beta[-1])
        return abscisse, ordonnee


if __name__ == "__main__":
    my_classifier = LDA()
    my_classifier.main()
