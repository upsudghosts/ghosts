import numpy as np

def drop_censored(X, Y):
        print(len(X))
        censored_lines = []
        for i in range(len(Y)):
            if Y[i,-1] == 1:
                censored_lines.append(i)
            
        
        cpt = 0
        resX = []
        valX = []
        resY = []
        valY = []
        print(len(censored_lines))
        for i in range(len(X)):
            #print(cpt)
            if i == censored_lines[cpt] and cpt < len(censored_lines)-1:
                for j in X[i]:
                    valX.append(j)
                for j in Y[i]:
                    valY.append(j)
                resX.append(valX)
                resY.append(valY)
                cpt+=1
                valX = []
                valY = []
                
        solX = np.asarray(resX)
        solY = np.asarray(resY)
        return solX, solY
