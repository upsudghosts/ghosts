'''
Sample predictive model.
You must supply at least 4 methods:
- fit: trains the model.
- predict: uses the model to perform predictions.
- save: saves the model.
- load: reloads the model.
'''
import numpy as np   # We recommend to use numpy arrays
from os.path import isfile
from sklearn.base import BaseEstimator
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle 

from sklearn.tree import DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor

from lifelines import CoxPHFitter
from lifelines.datasets import load_rossi

import tobit

class model(BaseEstimator):
    def __init__(self):
        '''
        This constructor is supposed to initialize data members.
        Use triple quotes for function documentation.
        '''
        self.num_train_samples=0
        self.num_feat=1
        self.num_labels=1
        self.is_trained= False
        
        self.what = 8

        # Baseline decision tree :
        if self.what == 1:
            self.baseline_clf = GaussianNB()
        elif self.what == 2:
            self.baseline_clf = Ridge()
        elif self.what == 3:
            self.baseline_clf = DecisionTreeRegressor(max_depth=4)
        elif self.what == 4:
            self.baseline_clf = RandomForestClassifier()
        elif self.what == 5:
            self.baseline_clf = NearestCentroid()
        elif self.what == 6:
            self.baseline_clf = tobit.TobitModel()
        elif self.what == 7:
            self.baseline_clf = CoxPHFitter()
        elif self.what == 8:
            self.baseline_clf = GradientBoostingRegressor()

    def fit(self, X, y):
        '''
        This function should train the model parameters.
        Here we do nothing in this example...
        Args:
            X: Training data matrix of dim num_train_samples * num_feat.
            y: Training label matrix of dim num_train_samples * num_labels.
        Both inputs are numpy arrays.
        For classification, labels could be either numbers 0, 1, ... c-1 for c classe
        or one-hot encoded vector of zeros, with a 1 at the kth position for class k.
        The AutoML format support on-hot encoding, which also works for multi-labels problems.
        Use data_converter.convert_to_num() to convert to the category number format.
        For regression, labels are continuous values.
        '''

        self.num_train_samples = X.shape[0]
        if X.ndim>1: self.num_feat = X.shape[1]
        print("FIT: dim(X)= [{:d}, {:d}]".format(self.num_train_samples, self.num_feat))
        num_train_samples = y.shape[0]
        if y.ndim>1: self.num_labels = y.shape[1]
        print("FIT: dim(y)= [{:d}, {:d}]".format(num_train_samples, self.num_labels))
        if (self.num_train_samples != num_train_samples):
            print("ARRGH: number of samples in X and y do not match!")

        '''For this baseline we will do a regular regression wihout taking
        censored data into account, the target value of the regression is contained in
        the first column of the y array'''
        # We get the target for the regression (y[:,1] contains the events)
        # y[:,0] is a slicing method, it takes all the lines ':' , and only the first column '0'
        # of the 2-d ndarray y

        # Once we have our regression target, we simply fit our model :
        if self.what == 6:
            self.baseline_clf.fit(X, y)
        elif self.what == 7: # doesnt work for now
            X = pd.DataFrame(X)
            self.baseline_clf.fit(X, duration_col='day')
        else:
            y1 = y[:,0]
            self.baseline_clf.fit(X, y1) # or y1

        self.is_trained=True

    def predict(self, X):
        '''
        This function should provide predictions of labels on (test) data.
        Here we just return zeros...
        Make sure that the predicted values are in the correct format for the scoring
        metric. For example, binary classification problems often expect predictions
        in the form of a discriminant value (if the area under the ROC curve it the metric)
        rather that predictions of the class labels themselves. For multi-class or multi-labels
        problems, class probabilities are often expected if the metric is cross-entropy.
        Scikit-learn also has a function predict-proba, we do not require it.
        The function predict eventually can return probabilities.
        '''
        num_test_samples = X.shape[0]
        if X.ndim>1: num_feat = X.shape[1]
        print("PREDICT: dim(X)= [{:d}, {:d}]".format(num_test_samples, num_feat))
        if (self.num_feat != num_feat):
            print("ARRGH: number of features in X does not match training data!")
        print("PREDICT: dim(y)= [{:d}, {:d}]".format(num_test_samples, self.num_labels))
        # We ask the model to predict new data X :
        pred = self.baseline_clf.predict(X)
        print('DEBUG : '+str(pred.shape))
        return pred

    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "wb"))

    def load(self, path="./"):
        modelfile = path + '_model.pickle'
        if isfile(modelfile):
            with open(modelfile, 'rb') as f:
                self = pickle.load(f)
            print("Model reloaded from: " + modelfile)
        return self
