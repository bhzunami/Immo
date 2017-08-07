"""
https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/

https://www.analyticsvidhya.com/blog/2015/06/tuning-random-forest-model/

https://www.analyticsvidhya.com/blog/2016/04/complete-tutorial-tree-based-modeling-scratch-in-python/
"""
import logging
import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt
# Scikit
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.externals import joblib
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
import pdb
from .helper import generate_matrix, plot, train_statistics

RNG = np.random.RandomState(42)
from .pipeline import Pipeline

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
import xgboost as xgb


RNG = np.random.RandomState(42)

class PredictPipeline(Pipeline):
    def __init__(self, goal, settings, directory):
        super().__init__(goal, settings, directory)

        predict_pipeline = [
            #self.outlier_detection,
            #self.adaBoost,
            #self.xgboost,
            #self.extraTreeRegression,
            self.mix,
        ]

        self.pipeline = self.preparation_pipeline() + predict_pipeline

    def adaBoost(self, ads):
        X, y = generate_matrix(ads, 'price')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        boost = AdaBoostRegressor(DecisionTreeRegressor(),
                                  n_estimators=18, random_state=RNG)
        boost.fit(X_train, y_train)
        joblib.dump(boost, '{}/adaboost.pkl'.format(self.model_folder))
        y_pred = boost.predict(X_test)
        train_statistics(y_test, y_pred, title="Adaboost")
        plot(y_test, y_pred, self.image_folder, show=True, title="adaboost")
        return ads


    def xgboost(self, ads):
        print("Start xgboost")
        X, y = generate_matrix(ads, 'price')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        xgb_model = xgb.XGBRegressor(silent=False, max_depth=100,
                                     learning_rate=0.1, n_estimators=350, n_jobs=-1)

        xgb_model.fit(X_train, y_train)
        xgb_model._Booster.save_model('{}/xgbooster.pkl'.format(self.model_folder))

        y_pred_xgb = xgb_model.predict(X_test)
        train_statistics(y_test, y_pred_xgb, title="Stacked xgb")
        return ads

    def extraTreeRegression(self, ads):
        X, y = generate_matrix(ads, 'price')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        model = ExtraTreesRegressor(n_estimators=700, n_jobs=-1, random_state=RNG)
        model.fit(X_train, y_train)
        joblib.dump(model, '{}/extra_tree_regressior.pkl'.format(self.model_folder))
        y_pred = model.predict(X_test)
        train_statistics(y_test, y_pred, title="ExtraTree_train")
        plot(y_test, y_pred, self.image_folder, show=True, title="ExtraTree_train")
        return ads
        
    def mix(self, ads):
        X, y = generate_matrix(ads, 'price')
        X, y = X.values, y.values



        for train_index, test_index in KFold(n_splits=3, shuffle=True).split(X):
            logging.info('MIX new split')
            logging.info("="*20)

            adaboost = joblib.load('{}/adaboost.pkl'.format(self.model_folder))
            xgb_model = xgb.XGBRegressor(silent=False, max_depth=100,
                                        learning_rate=0.1, n_estimators=350, n_jobs=-1)
            booster = xgb.Booster()
            booster.load_model('{}/xgbooster.pkl'.format(self.model_folder))
            xgb_model._Booster = booster

            tree = joblib.load('{}/extra_tree_regressior.pkl'.format(self.model_folder))

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
       
            a_predict = adaboost.predict(X_test)
            xg_predict = xgb_model.predict(X_test)
            t_predict = tree.predict(X_test)
            y_pred = np.array(0.2*a_predict + 0.6*xg_predict + 0.2*t_predict)
            train_statistics(y_test, y_pred, title="BEST")

            y_pred = np.array(0.3*a_predict + 0.3*xg_predict + 0.4*t_predict)
            train_statistics(y_test, y_pred, title="0.3, 0.3, 0.4")

            y_pred = np.array(0.8*a_predict + 0.1*xg_predict + 0.1*t_predict)
            train_statistics(y_test, y_pred, title="0.8, 0.1, 0.1")

            y_pred = np.array(0.2*a_predict + 0.6*xg_predict + 0.2*t_predict)
            train_statistics(y_test, y_pred, title="0.2, 0.6, 0.2")

            y_pred = np.array(0.4*a_predict + 0.4*xg_predict + 0.2*t_predict)
            train_statistics(y_test, y_pred, title="0.4, 0.4, 0.2")

            y_pred = np.array(0.1*a_predict + 0.8*xg_predict + 0.1*t_predict)
            train_statistics(y_test, y_pred, title="0.1, 0.8, 0.1")

            y_pred = np.array(0.6*a_predict + 0.1*xg_predict + 0.3*t_predict)
            train_statistics(y_test, y_pred, title="0.6, 0.1, 0.3")

            y_pred = np.array(0.2*a_predict + 0.7*xg_predict + 0.1*t_predict)
            train_statistics(y_test, y_pred, title="0.2, 0.7, 0.1")

            y_pred = np.array(0.2*a_predict + 0.0*xg_predict + 0.8*t_predict)
            train_statistics(y_test, y_pred, title="0.2, 0.0, 0.8")

            y_pred = np.array(0.3*a_predict + 0.4*xg_predict + 0.3*t_predict)
            train_statistics(y_test, y_pred, title="0.3, 0.4, 0.3")

        return ads


