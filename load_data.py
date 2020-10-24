def get_dataset_path(dataset_letter):
    return 'data/classification' + dataset_letter


def get_train_filename(dataset_letter='A'):
    return get_dataset_path(dataset_letter) + '.train'


def get_test_filename(dataset_letter='A'):
    return get_dataset_path(dataset_letter) + '.test'


def get_plot_suffixe(class_name=''):
    matches = {
        'RegressionLineaire': 'RegLin',
        'RegressionLogistique': 'RegLog',
        'LDA': 'LDA',
        'QDA': 'QDA',
    }

    try:
        plot_suffixe = '.' + matches[class_name]
    except KeyError:
        plot_suffixe = ''

    plot_suffixe += '.png'

    return plot_suffixe


def main():
    for dataset_letter in ['A', 'B', 'C']:

        for text in [get_dataset_path(dataset_letter),
                     get_train_filename(dataset_letter),
                     get_test_filename(dataset_letter),
                     get_plot_suffixe('')]:
            print(text)

    return True


if __name__ == "__main__":
    main()
