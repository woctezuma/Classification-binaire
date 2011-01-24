import numpy as np
from ModeleDiscriminatif import ModeleDiscriminatif

class RegressionLineaire(ModeleDiscriminatif):
	'''
	Régression linéaire
	'''
	
	def train(self, filename):
	    self.creer(filename)
	    self.beta = np.linalg.solve(np.dot(self.X.T,self.X), np.dot(self.X.T,self.Y))
	    return
	
	def getErreur(self, filename):
	    self.creer(filename)
	    Ypred = np.dot(self.X, self.beta)
	    return sum(abs(self.Y-Ypred))/len(self.Y)
	
	def getSeparateur(self):
	    abscisse = np.linspace(min(self.X[:,-2]),max(self.X[:,-2]))
	    valeur = 0.5
	    ordonnee = (valeur-self.beta[0]-self.beta[1]*abscisse)/self.beta[2]
	    return (abscisse, ordonnee)
