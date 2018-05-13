from RegressionLogistique import RegressionLogistique


def main():
    from load_data import get_train_filename, get_test_filename, get_plot_suffixe

    for dataset_letter in ['A', 'B', 'C']:
        my_classifier = RegressionLogistique()
        my_classifier.train(get_train_filename(dataset_letter))

        my_classifier.display_error(get_train_filename(dataset_letter))
        my_classifier.display_figure(get_train_filename(dataset_letter) + get_plot_suffixe(my_classifier.get_name()))

        my_classifier.display_error(get_test_filename(dataset_letter))
        my_classifier.display_figure(get_test_filename(dataset_letter) + get_plot_suffixe(my_classifier.get_name()))

    return True


if __name__ == "__main__":
    main()
