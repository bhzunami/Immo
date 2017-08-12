import pandas as pd
import pdb
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import seaborn as sns
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import chi2, f_regression
from sklearn.linear_model import LassoLarsCV, Ridge, RidgeCV, LassoCV, Lasso, LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn import metrics
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import export_graphviz
import logging
import scipy
import gc
from multiprocessing import Pool
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from collections import defaultdict

from .helper import generate_matrix, ape, mape, mdape, gen_subplots, plot, train_statistics


class StackedRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, estimators, weights=None):
        self._estimators = estimators

        if weights is None:
            weights = [1/len(estimators)] * len(estimators)

        self.set_weights(weights)

    def fit(self, X, y):
        for estimator in self._estimators:
            estimator.fit(X, y)

        return self

    def set_weights(self, weights):
        if len(weights) != len(self._estimators):
            raise Exception("Number of weights not equal to number of estimators.")
        if np.sum(weights) != 1: # TODO add epsilon? e.g. np.sum([0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1]) != 1
            raise Exception("Given weights don't add up to 1!")

        self._weights = weights

        return self

    def predict_(self, X):
        return [np.array(estimator.predict(X)) * weight for estimator, weight in zip(self._estimators, self._weights)]

    def predict(self, X):
        if self._weights is None:
            raise Exception("Set estimator weights first!")


        return np.sum(self.predict_(X), axis=0)
