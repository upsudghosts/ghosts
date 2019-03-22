filename1 = "public_data/Mortality_train.data"
dataset = np.loadtxt(filename1, delimiter=" ")
from sys import argv
import numpy as np
from sklearn.base import BaseEstimator
#INITIALISER X POUR TESTER
class zePreproDeLaMort(BaseEstimator):
    
    def __init__(self):
        self.uncensoredLines = []
        
    def fit(self, X, y = None):
        for i in range(len(X)):
            if X[i,-1] == 0:
                self.uncensoredLines.append(i)
        return np.asarray(self.uncensoredLines)
   
    def transform(self, X, y = None):
        cpt = 0
        res = []
        val = []
        for j in range(len(dataset)):
            if j == X[cpt]:
                for i in dataset[j]:
                    val.append(i)
                res.append(val)
                cpt+=1
                val = []
        return np.asarray(res)
  
    def fit_transform(self, X, y = None):
        lines = fit(X)
        transform = transform(lines)
        return transform



    def testing():
        prepro = zePreproDeLaMort()
        fitTab = prepro.fit(dataset)
        transformTab = prepro.transform(fitTab)
        #print(transformTab[9:12])
    testing()
    
