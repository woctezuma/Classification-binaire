import numpy as np

from ModeleDiscriminatif import ModeleDiscriminatif


class RegressionLineaire(ModeleDiscriminatif):
    '''
    Régression linéaire
    '''

    def set_name(self):
        return 'RegressionLineaire'

    def train(self, filename):
        self.creer(filename)
        self.beta = np.linalg.solve(np.dot(self.X.T, self.X), np.dot(self.X.T, self.Y))
        return

    def getErreur(self, filename):
        self.creer(filename)
        Ypred = np.dot(self.X, self.beta)
        return sum(abs(self.Y - Ypred)) / len(self.Y)

    def getSeparateur(self):
        abscisse = np.linspace(min(self.X[:, -2]), max(self.X[:, -2]))
        valeur = 0.5
        ordonnee = (valeur - self.beta[0] - self.beta[1] * abscisse) / self.beta[2]
        return (abscisse, ordonnee)


def main():
    from load_data import get_train_filename, get_test_filename, get_suffixe_graphique

    for dataset_letter in ['A', 'B', 'C']:
        my_classifier = RegressionLineaire()
        my_classifier.train(get_train_filename(dataset_letter))

        my_classifier.afficherErreur(get_train_filename(dataset_letter))
        my_classifier.afficher(get_train_filename(dataset_letter) + get_suffixe_graphique(my_classifier.get_name()))

        my_classifier.afficherErreur(get_test_filename(dataset_letter))
        my_classifier.afficher(get_test_filename(dataset_letter) + get_suffixe_graphique(my_classifier.get_name()))

    return True


if __name__ == "__main__":
    main()
