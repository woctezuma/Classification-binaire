import math
import numpy as np
from ModeleDiscriminatif import ModeleDiscriminatif

class RegressionLogistique(ModeleDiscriminatif):
	'''
	Régression logistique
	'''
	
	def train(self, filename):
		self.creer(filename)
		self.beta = np.linspace(0, 0, len(self.X[0]))
		mu = 0.5*(self.Y+0.5)
		num_iter = 0
		while(num_iter<25):
			W = np.diag(mu*(1-mu))
			delta_beta = np.linalg.solve(np.dot(self.X.T, np.dot(W,self.X)), np.dot(self.X.T,(self.Y-mu)))
			self.beta += delta_beta
			if all(delta_beta < 1e-8):
			    break
			for i in range(len(mu)):
			    mu[i] = 1.0/(1+math.exp(-np.dot(self.X[i,:],self.beta)))
			num_iter+=1
		return
	
	def getErreur(self, filename):
		self.creer(filename)
		mu = np.linspace(0,0,len(self.Y))
		for i in range(len(mu)):
		    mu[i] = 1.0/(1+math.exp(-np.dot(self.X[i,:],self.beta)))
		Ypred = mu
		return sum(abs(self.Y-Ypred))/len(self.Y)
	
	def getSeparateur(self):
	    abscisse = np.linspace(min(self.X[:,-2]),max(self.X[:,-2]))
	    valeur = 0.
	    ordonnee = (valeur-self.beta[0]-self.beta[1]*abscisse)/self.beta[2]
	    return (abscisse, ordonnee)
