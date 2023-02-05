def get_dataset_path(dataset_letter):
    dataset_path = 'data/classification' + dataset_letter
    return dataset_path


def get_train_filename(dataset_letter='A'):
    train_filename = get_dataset_path(dataset_letter) + '.train'
    return train_filename


def get_test_filename(dataset_letter='A'):
    test_filename = get_dataset_path(dataset_letter) + '.test'
    return test_filename


def get_plot_suffixe(class_name=''):
    matches = {}
    matches['RegressionLineaire'] = 'RegLin'
    matches['RegressionLogistique'] = 'RegLog'
    matches['LDA'] = 'LDA'
    matches['QDA'] = 'QDA'

    try:
        plot_suffixe = '.' + matches[class_name]
    except KeyError:
        plot_suffixe = ''

    plot_suffixe += '.png'

    return plot_suffixe


def main():
    for dataset_letter in ['A', 'B', 'C']:
        for text in [
            get_dataset_path(dataset_letter),
            get_train_filename(dataset_letter),
            get_test_filename(dataset_letter),
            get_plot_suffixe(''),
        ]:
            print(text)

    return True


if __name__ == "__main__":
    main()
