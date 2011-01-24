import math
import numpy as np
from ModeleGeneratif import ModeleGeneratif

def conique(gamma, beta, alpha, absc, ordo):
    return gamma+np.dot(np.array([absc,ordo]).T,beta)+0.5*np.dot(np.array([absc,ordo]).T,np.dot(alpha,np.array([absc,ordo])))

class QDA(ModeleGeneratif):
    '''
    Quadratic Discriminant Analysis (QDA)
    '''
    
    def __init__(self):
        ModeleGeneratif.__init__(self)
        self.sigma1 = np.array(0)
        self.sigma0 = np.array(0)
        return
    
    def train(self, filename):
        self.creer(filename)
        N1 = sum(self.Y)
        n = len(self.Y)
        self.pi = N1/n
        self.mu1 = np.array([np.dot(self.X[:,i],self.Y) for i in range(len(self.X[0,:]))])
        self.mu0 = np.array([np.dot(self.X[:,i],(1-self.Y)) for i in range(len(self.X[0,:]))])
        self.mu1 /= N1
        self.mu0 /= n-N1
        self.sigma1 = sum([np.outer(z,z)
                           for z in [(self.X[i,:]-self.mu1)
                                    for i in range(len(self.Y)) if self.Y[i]==1]])
        self.sigma0 = sum([np.outer(z,z)
                           for z in [(self.X[i,:]-self.mu0)
                                     for i in range(len(self.Y)) if self.Y[i]==0]])
        self.sigma1 /= n
        self.sigma0 /= n
        return
    
    def getErreur(self, filename):
        self.creer(filename)
        invsigma1 = np.linalg.inv(self.sigma1)
        invsigma0 = np.linalg.inv(self.sigma0)
        alpha = 0.5*(invsigma0-invsigma1)
        beta = np.dot(invsigma1, self.mu1) - np.dot(invsigma0, self.mu0)
        gamma = math.log(self.pi/(1-self.pi))+0.5*math.log(np.linalg.det(self.sigma0)/np.linalg.det(self.sigma1))-0.5*np.dot(self.mu1.T,np.dot(invsigma1,self.mu1))+0.5*np.dot(self.mu0.T,np.dot(invsigma0,self.mu0))
        Ypred = np.array(self.Y)
        for i in range(len(self.Y)):
            Ypred[i] = 1.0/(1+math.exp(-gamma-np.dot(self.X[i,:],beta)-0.5*np.dot(self.X[i,:].T,np.dot(alpha,self.X[i,:]))))
        return sum(abs(self.Y-Ypred))/len(self.Y)

    def getSeparateur(self):
        invsigma1 = np.linalg.inv(self.sigma1)
        invsigma0 = np.linalg.inv(self.sigma0)
        alpha = 0.5*(invsigma0-invsigma1)
        beta = np.dot(invsigma1, self.mu1) - np.dot(invsigma0, self.mu0)
        gamma = math.log(self.pi/(1-self.pi))+0.5*math.log(np.linalg.det(self.sigma0)/np.linalg.det(self.sigma1))-0.5*np.dot(self.mu1.T,np.dot(invsigma1,self.mu1))+0.5*np.dot(self.mu0.T,np.dot(invsigma0,self.mu0))
        abscisse_possible = np.linspace(min(self.X[:,-2]), max(self.X[:,-2]))
        ordonnee_possible = np.linspace(min(self.X[:,-1]), max(self.X[:,-1]), 500)
        abscisse = []
        ordonnee = []
        for i in range(2,len(abscisse_possible)):
            MaConique = [abs(conique(gamma,beta,alpha,abscisse_possible[i],ordo)) for ordo in ordonnee_possible]
            MonMinimum = min(MaConique)
            MonRang = np.nonzero(MaConique==MonMinimum)[0]
            if MonMinimum<0.3 and (len(ordonnee)==0 or abs(ordonnee[-1]-ordonnee_possible[MonRang])<1):
                abscisse.append(abscisse_possible[i])
                ordonnee.append(ordonnee_possible[MonRang])
        return (abscisse, ordonnee)
