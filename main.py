from RegressionLogistique import RegressionLogistique


def main():
    from load_data import get_train_filename, get_test_filename, get_suffixe_graphique

    for dataset_letter in ['A', 'B', 'C']:
        my_classifier = RegressionLogistique()
        my_classifier.train(get_train_filename(dataset_letter))

        my_classifier.afficherErreur(get_train_filename(dataset_letter))
        my_classifier.afficher(get_train_filename(dataset_letter) + get_suffixe_graphique(my_classifier.get_name()))

        my_classifier.afficherErreur(get_test_filename(dataset_letter))
        my_classifier.afficher(get_test_filename(dataset_letter) + get_suffixe_graphique(my_classifier.get_name()))

    return True


if __name__ == "__main__":
    main()
