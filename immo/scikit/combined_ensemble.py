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

from helper import generate_matrix, ape, mape, mdape, gen_subplots, plot, train_statistics


def dpathes_to_hash(dpathes):
    return [hash(frozenset(row.nonzero()[1])) for row in dpathes]

# this monster is needed because python can't pickle lambdas
class PseudoLambda(object):
    def __init__(self, *args):
        self.args = args
    def __call__(self, tree):
        return dpathes_to_hash(tree.decision_path(self.args[0]))

def pool_dpath_trees(trees, X, n_parts=1000, n_pools=4):
    parts = []
    # split X up into parts, otherwise we get a memory error
    for i in range(0, len(X), n_parts):
        with Pool(n_pools) as p:
            parts.append(p.map(PseudoLambda(X[i:i+n_parts]), trees))

    return [sum(list(x), []) for x in list(zip(*parts))]


class CombinedEnsemble(BaseEstimator):
    def __init__(self, ensemble_estimator, estimator2 = None):
        self.dpath_trees = []
        self.dpath_trees_grouped = []
        self.X = None # pandas dataframe
        self.y = None # pandas dataframe

        # ensemble_estimator must be a Forest Estimator
        self.ensemble_estimator = ensemble_estimator
        # estimator2 may be any kind of estimator
        self.estimator2 = estimator2
        # example:
        # ensemble_estimator=lambda: ExtraTreesRegressor(n_estimators=100, min_samples_leaf=3)
        # estimator2=lambda: LinearRegression(normalize=True)

    def fit(self, X, y, **kwargs):
        self.ensemble_estimator.fit(X, y, **kwargs)

        self.X = X
        self.y = y

        self.dpath_trees = pool_dpath_trees(self.ensemble_estimator.estimators_, X)

        # group indexes by hash
        self.dpath_trees_grouped = [defaultdict(list) for _ in range(len(self.dpath_trees))]
        for tree_idx, tree in enumerate(self.dpath_trees):
            for idx, hash_val in enumerate(tree):
                self.dpath_trees_grouped[tree_idx][hash_val].append(idx)

        return self

    def predict(self, X, verbose=False):

        if verbose: logging.info("get decision pathes")
        predict_dpathes = pool_dpath_trees(self.ensemble_estimator.estimators_, X)

        if verbose: logging.info("match decision pathes")
        matching_rows = [[] for _ in range(len(X))]
        for test_tree, dpath_tree in zip(predict_dpathes, self.dpath_trees_grouped):
            for test_i, test in enumerate(test_tree):
                matching_rows[test_i] += dpath_tree[test]
        matching_rows = [list(set(x)) for x in matching_rows] # get unique values

        all_y_pred = []
        for i, row in enumerate(matching_rows):
            if verbose and i%1000==0: logging.info("fit and predict {}/{}".format(i, len(matching_rows)))

            all_y_pred.append(clone(self.estimator2) \
                                .fit(self.X.iloc[row], self.y.iloc[row]) \
                                .predict([X.iloc[i]])[0])

        if verbose: logging.info("fit and predict finished")

        return all_y_pred
