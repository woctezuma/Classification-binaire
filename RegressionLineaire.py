import numpy as np

from ModeleDiscriminatif import ModeleDiscriminatif


class RegressionLineaire(ModeleDiscriminatif):
    """
    Régression linéaire
    """

    def set_name(self):
        return 'RegressionLineaire'

    def train(self, filename):
        self.read_data(filename)
        self.beta = np.linalg.solve(np.dot(self.X.T, self.X), np.dot(self.X.T, self.Y))
        return

    def get_error(self, filename):
        self.read_data(filename)
        # noinspection PyPep8Naming
        Y_pred = np.dot(self.X, self.beta)
        return sum(abs(self.Y - Y_pred)) / len(self.Y)

    def get_separator(self):
        abscisse = np.linspace(min(self.X[:, -2]), max(self.X[:, -2]))
        valeur = 0.5
        ordonnee = (valeur - self.beta[0] - self.beta[1] * abscisse) / self.beta[2]
        return abscisse, ordonnee


if __name__ == "__main__":
    my_classifier = RegressionLineaire()
    my_classifier.main()
