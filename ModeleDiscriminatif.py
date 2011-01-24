import csv
import numpy as np
from Modele import Modele

class ModeleDiscriminatif(Modele):
        '''
        Modele discriminatif de classification binaire
        '''
        
        def __init__(self):
                Modele.__init__(self)
                self.beta = np.array(0)
                return
        
        def creer(self, filename, sep ='\t'):
                fileReader = csv.reader(open(filename),delimiter=sep)
                fileContent = [[float(xa),float(xb),float(y)]
                               for (xa,xb,y) in fileReader]
                XY = np.array(fileContent)
                Xintercept =  np.linspace(1,1,len(XY))
                self.X = np.column_stack((Xintercept, XY[:,:-1]))
                self.Y = XY[:, -1]
