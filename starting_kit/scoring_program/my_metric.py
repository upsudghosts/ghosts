'''Examples of organizer-provided metrics.
You can just replace this code by your own.
Make sure to indicate the name of the function that you chose as metric function
in the file metric.txt. E.g. mse_metric, because this file may contain more
than one function, hence you must specify the name of the function that is your metric.'''

import numpy as np
import pandas as pd
import scipy as sp
from lifelines.utils import concordance_index
from sklearn.metrics import mean_absolute_error
from scipy.stats import logistic
from sksurv.metrics import concordance_index_censored


    #return c
def c_index_old(solution, prediction):
    '''Concordance index from the lifelines library
    '''
    # the concordance_index function requires a panda dataFrame
    df = pd.DataFrame(solution, columns=['time', 'event'])
    return concordance_index(df['time'], prediction, df['event'])

def mse_metric(solution, prediction):
    '''Mean-square error.
    Works even if the target matrix has more than one column'''
    df = pd.DataFrame(solution, columns=['time', 'event'])
    mse = np.mean((solution-prediction)**2)
    return np.mean(mse)

def custom_c_index(solution, prediction):
    solution_events = solution[:,1].astype('bool')
    solution_times  = solution[:,0]
    #c = concordance_index_censored(solution_events, solution_times, prediction)[0]
    c = c_index_old(solution, prediction)
    n_c = np.clip(2*c - 1, a_min=0, a_max=1)
    mse = mean_absolute_error(solution_times, prediction)
    worstR = 50
    r = (1 - (mse / worstR))
    # print('C = '+str(c))
    # print("N_C = "+str(n_c))
    # print('R = '+str(r))
    # print('MSE = '+str(mse)+' log = '+str(logistic.cdf(mse)))
    alpha = 0.1
    pen = 0.1
    if r < 0 :
        score = c + pen * r
    else :
        score = np.clip(0.2*np.clip(r, a_min=0.1, a_max=1)+(r+alpha)*c, a_min=0, a_max=1)
    return score
