import numpy as np

def drop_censored(X, Y):
    censored_indexes = np.where(Y==1)[0] #age != 0
    X_uncensored = np.delete(X, censored_indexes, axis=0)
    Y_uncensored = np.delete(Y, censored_indexes, axis=0)
    return X_uncensored, Y_uncensored 