import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class Modele:
    '''
    Modele pour la classification binaire
    '''
    
    def __init__(self):
        self.X = np.array(0)
        self.Y = np.array(0)
        return
    
    def creer(self, filename, sep ='\t'):
        raise NotImplemented
    
    def train(self, filename):
        raise NotImplemented
    
    def getErreur(self, filename):
        raise NotImplemented
    
    def getSeparateur(self):
        raise NotImplemented
    
    def afficher(self, outputName):
        plt.figure()
        labels_1 = (self.Y==1)
        plt.plot(self.X[labels_1,-2], self.X[labels_1,-1], 'bo')
        labels_0 = (self.Y==0)
        plt.plot(self.X[labels_0,-2], self.X[labels_0,-1], 'ro')
        (abscisse, ordonnee) = self.getSeparateur()
        plt.plot(abscisse, ordonnee, 'k')
        plt.savefig(outputName)
        return
    
    def afficherErreur(self, filename):
        print 'Erreur (%s) : %.2f' % (filename, self.getErreur(filename))
        return
