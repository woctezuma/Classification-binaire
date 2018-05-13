def get_dataset_path(dataset_letter):
    dataset_path = 'data/classification' + dataset_letter
    return dataset_path


def get_train_filename(dataset_letter='A'):
    train_filename = get_dataset_path(dataset_letter) + '.train'
    return train_filename


def get_test_filename(dataset_letter='A'):
    test_filename = get_dataset_path(dataset_letter) + '.test'
    return test_filename


def get_suffixe_graphique(class_name=''):
    matches = dict()
    matches['RegressionLineaire'] = 'RegLin'
    matches['RegressionLogistique'] = 'RegLog'
    matches['LDA'] = 'LDA'
    matches['QDA'] = 'QDA'

    try:
        suffixe_graphique = '.' + matches[class_name]
    except KeyError:
        suffixe_graphique = ''

    suffixe_graphique += '.png'

    return suffixe_graphique


def main():
    for dataset_letter in ['A', 'B', 'C']:

        for str in [get_dataset_path(dataset_letter),
                    get_train_filename(dataset_letter),
                    get_test_filename(dataset_letter),
                    get_suffixe_graphique('')]:
            print(str)

    return True


if __name__ == "__main__":
    main()
