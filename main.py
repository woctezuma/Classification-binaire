from RegressionLineaire import RegressionLineaire
from RegressionLogistique import RegressionLogistique
from LDA import LDA
from QDA import QDA

if __name__ == "__main__":
    train_filename = 'myTrainFile.txt' # TODO
    test_filename = 'myTestFile.txt' # TODO
    suffixe_graphique = '.RegLog.png'

    my_classifier = RegressionLogistique()
    my_classifier.train(train_filename)

    my_classifier.afficherErreur(train_filename)
    my_classifier.afficher(train_filename+suffixe_graphique)
    
    my_classifier.afficherErreur(test_filename)
    my_classifier.afficher(test_filename+suffixe_graphique)
