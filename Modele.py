import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


class Modele:
    """Modele pour la classification binaire."""

    def __init__(self):
        self.X = np.array(0)
        self.Y = np.array(0)
        self.name = self.set_name()
        return

    def set_name(self):
        raise NotImplementedError

    def get_name(self):
        return self.name

    def read_data(self, filename, sep='\t'):
        raise NotImplementedError

    def train(self, filename):
        raise NotImplementedError

    def get_error(self, filename):
        raise NotImplementedError

    def get_separator(self):
        raise NotImplementedError

    def display_figure(self, output_name):
        plt.figure()
        labels_1 = self.Y == 1
        plt.plot(self.X[labels_1, -2], self.X[labels_1, -1], 'bo')
        labels_0 = self.Y == 0
        plt.plot(self.X[labels_0, -2], self.X[labels_0, -1], 'ro')
        (abscisse, ordonnee) = self.get_separator()
        plt.plot(abscisse, ordonnee, 'k')
        plt.savefig(output_name)
        return

    def display_error(self, filename):
        print(f'Erreur ({filename}) : {self.get_error(filename):.2f}')
        return

    def main(self):
        from load_data import get_plot_suffixe, get_test_filename, get_train_filename

        for dataset_letter in ['A', 'B', 'C']:
            self.train(get_train_filename(dataset_letter))

            self.display_error(get_train_filename(dataset_letter))
            self.display_figure(
                get_train_filename(dataset_letter) + get_plot_suffixe(self.get_name()),
            )

            self.display_error(get_test_filename(dataset_letter))
            self.display_figure(
                get_test_filename(dataset_letter) + get_plot_suffixe(self.get_name()),
            )

        return True
