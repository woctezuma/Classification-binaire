import math

import numpy as np

from ModeleGeneratif import ModeleGeneratif


class LDA(ModeleGeneratif):
    '''
    Linear Discriminant Analysis (LDA)
    '''

    def __init__(self):
        ModeleGeneratif.__init__(self)
        self.sigma = np.array(0)
        return

    def set_name(self):
        return 'LDA'

    def train(self, filename):
        self.creer(filename)
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

    def getErreur(self, filename):
        self.creer(filename)
        beta = np.linalg.solve(self.sigma, self.mu1 - self.mu0)
        temp = np.linalg.solve(self.sigma, self.mu1 + self.mu0)
        gamma = -0.5 * np.dot((self.mu1 - self.mu0).T, temp) + math.log(self.pi / (1 - self.pi))
        Ypred = np.array(self.Y)
        for i in range(len(self.Y)):
            Ypred[i] = 1.0 / (1 + math.exp(-gamma - np.dot(self.X[i, :], beta)))
        return sum(abs(self.Y - Ypred)) / len(self.Y)

    def getSeparateur(self):
        abscisse = np.linspace(min(self.X[:, -2]), max(self.X[:, -2]))
        valeur = 0.
        beta = np.linalg.solve(self.sigma, self.mu1 - self.mu0)
        temp = np.linalg.solve(self.sigma, self.mu1 + self.mu0)
        gamma = -0.5 * np.dot((self.mu1 - self.mu0).T, temp) + math.log(self.pi / (1 - self.pi))
        ordonnee = (valeur + gamma + beta[-2] * abscisse) / (-beta[-1])
        return (abscisse, ordonnee)


def main():
    from load_data import get_train_filename, get_test_filename, get_suffixe_graphique

    for dataset_letter in ['A', 'B', 'C']:
        my_classifier = LDA()
        my_classifier.train(get_train_filename(dataset_letter))

        my_classifier.afficherErreur(get_train_filename(dataset_letter))
        my_classifier.afficher(get_train_filename(dataset_letter) + get_suffixe_graphique(my_classifier.get_name()))

        my_classifier.afficherErreur(get_test_filename(dataset_letter))
        my_classifier.afficher(get_test_filename(dataset_letter) + get_suffixe_graphique(my_classifier.get_name()))

    return True


if __name__ == "__main__":
    main()
