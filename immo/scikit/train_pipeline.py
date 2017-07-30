"""
https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/

https://www.analyticsvidhya.com/blog/2015/06/tuning-random-forest-model/

https://www.analyticsvidhya.com/blog/2016/04/complete-tutorial-tree-based-modeling-scratch-in-python/
"""
import os
import pdb
import logging
import json
import argparse
import numpy as np
#import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import ast
# Scikit
from sklearn.ensemble import ExtraTreesRegressor, IsolationForest
from sklearn.model_selection import train_test_split, KFold
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LassoLarsCV, Ridge, RidgeCV, LassoCV, Lasso, LinearRegression, LogisticRegression

# NLTK
#import nltk
#from nltk.corpus import stopwords # Import the stop word list
#from nltk.stem import SnowballStemmer

from .a_detection import AnomalyDetection
from .helper import generate_matrix, ape, mape, mdape, gen_subplots, plot, train_statistics

RNG = np.random.RandomState(42)
from .pipeline import Pipeline

#import xgboost as xgb
#import lightgbm as lgb
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.grid_search import GridSearchCV

from .combined_ensemble import CombinedEnsemble
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
# import xgboost as xgb
# import lightgbm as lgb

def detect_language(text):
    """ detect the language by the text where
    the most stopwords for a language are found
    """
    languages_ratios = {}
    tokens = nltk.wordpunct_tokenize(text)
    words_set = set([word.lower() for word in tokens])
    for language in ['german', 'french', 'italian']:
        stopwords_set = set(stopwords.words(language))
        languages_ratios[language] = len(words_set.intersection(stopwords_set)) # language "score"

    return max(languages_ratios, key=languages_ratios.get)

class TrainPipeline(Pipeline):
    def __init__(self, goal, settings, directory):
        super().__init__(goal, settings, directory)

        # pipeline for creating and saving ads
        #self.pipeline = self.preparation_pipeline + []

        train_pipleine = [
            self.load_df("{}/ads_prepared.pkl".format(self.model_folder)),
            self.train_outlier_detection,
            self.outlier_detection,
            #self.train_living_area
            #self.xgboost
            #self.lgb
            #self.adaBoost
            self.extraTreeRegression2,
            #self.xgboost]
            #self.lgb]
            #self.adaBoost]
            #self.kneighbours
            #self.train_extraTreeRegression_withKFold
            #self.train_extraTeeRegression
            #self.combinedEnsemble_settings,
            #self.combinedEnsemble_train,
            #self.combinedEnsemble_load,
            #self.combinedEnsemble_test
        ]

        self.pipeline = train_pipleine if os.path.isfile("{}/ads_prepared.pkl".format(self.model_folder)) else self.preparation_pipeline + train_pipleine

    def old_outliers_detection(self, ads):
        from sklearn import svm
        clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
        clf.fit(ads)
        predicted = clf.predict(ads)
        logging.info(ads[np.where(predicted == 1)].shape)
        logging.info(ads[np.where(predicted == -1)].shape)

    def rmse_cv(self, model, X, y_train):
        kf = KFold(5, shuffle=True, random_state=42).get_n_splits(X.values)
        rmse = np.sqrt(-cross_val_score(model, X.values, y_train, scoring="neg_mean_squared_error", cv=kf))
        return(rmse)


    def test(self, ads):
        X, y = generate_matrix(ads, 'price')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6)

        lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=42))
        score = self.rmse_cv(lasso, X_train, y_train)
        print("\nLasso score: {:.4f} (+/-{:.4f})\n".format(score.mean(), score.std()))


        ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))

        KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

        GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                           max_depth=4, max_features='sqrt',
                                           min_samples_leaf=15, min_samples_split=10,
                                           loss='huber', random_state =5)

        model_xgb = xgb.XGBRegressor(colsample_bytree=0.2, gamma=0.0,
                                     learning_rate=0.05, max_depth=6,
                                     min_child_weight=1.5, n_estimators=7200,
                                     reg_alpha=0.9, reg_lambda=0.6,
                                     subsample=0.2,seed=42, silent=1,
                                     random_state =7)

        model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                                      learning_rate=0.05, n_estimators=720,
                                      max_bin = 55, bagging_fraction = 0.8,
                                      bagging_freq = 5, feature_fraction = 0.2319,
                                      feature_fraction_seed=9, bagging_seed=9,
                                      min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)


    def lgb(self, ads):

        X, y = generate_matrix(ads, 'price')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6)
        train_data=lgb.Dataset(X_train, label=y_train)

        #param = {'num_leaves':500, 'objective':'regression', 'max_depth':7, 'learning_rate':.05,'max_bin':200}
        params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': {'l2', 'auc'},
            'verbose': 0
        }
        #param['metric'] = ['auc', 'binary_logloss']

        num_round = 50
        lgbm = lgb.train(params, train_data, num_boost_round=10000)
        y_pred = lgbm.predict(X_test, num_iteration=lgbm.best_iteration)
        train_statistics(y_test, y_pred, title="Lgb")
        return ads

    def adaBoost(self, ads):
        """http://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_regression.html#sphx-glr-auto-examples-ensemble-plot-adaboost-regression-py

        https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/

        """
        X, y = generate_matrix(ads, 'price')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6)
        # learning rate 0.05 - 0.2
        for i in [5, 10, 20, 50, 70]:
            logging.info("Adaboost with estimator: {}".format(i))
            boost = AdaBoostRegressor(DecisionTreeRegressor(),
                                      n_estimators=i, random_state=RNG)
            boost.fit(X_train, y_train)
            y_pred = boost.predict(X_test)
            train_statistics(y_test, y_pred, title="Adaboost with estimator: {}".format(i))

    def kneighbours(self, ads):
        X, y = generate_matrix(ads, 'price')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6)

        for i in [50, 100, 200, 400, 500]:
            logging.info("KNeighborsRegressor with leaf size: {}".format(i))
            neigh = KNeighborsRegressor(n_neighbors=2, weights='distance', leaf_size=i, n_jobs=6)
            neigh.fit(X_train, y_train)
            y_pred = neigh.predict(X_test)
            train_statistics(y_test, y_pred, title="KNeighbour with leaf size: {}".format(i))

        return ads

    # Best 200 depth n_estimators=50
    def xgboost(self, ads):
        # http://www.xavierdupre.fr/app/pymyinstall/helpsphinx/notebooks/example_xgboost.html
        X, y = generate_matrix(ads, 'price')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6)

        xgb_model = xgb.XGBRegressor()
        clf = GridSearchCV(xgb_model,
                           {'max_depth': [100, 200, 300],
                            'n_estimators': [10, 50, 80]},
                           verbose=1,
                           n_jobs=1)

        clf.fit(X_train, y_train)
        logging.info(clf.best_score_, clf.best_params_)
        return ads
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test)

        params = {"max_depth":100, "eta":0.1}
        logging.info("Start makeing model")
        model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)
        joblib.dump(model, '{}/xgb.pkl'.format(self.model_folder))

        model.loc[:,["test-rmse-mean", "train-rmse-mean"]].plot()
        plt.show()
        logging.info("Start making regressio model")

        model_xgb = xgb.XGBRegressor(n_estimators=350, max_depth=100, learning_rate=0.1) #the params were tuned using xgb.cv
        logging.info("Start fit model")
        model_xgb.fit(X_train, y_train)
        joblib.dump(model_xgb, '{}/xgb_fit.pkl'.format(self.model_folder))
        logging.info("Predict")
        y_pred = model_xgb.predict(X_test)
        train_statistics(y_test, y_pred)
        #plot(y_test, y_pred, show=False, plot_name="xgboost")

        return ads

    def train_test_validate_split(self, ads):
        # Shuffle the ads
        # drop=True prevents .reset_index from creating a column containing the old index entries.
        ads = ads.sample(frac=1).reset_index(drop=True)

        # train 70%
        train_ads = ads.sample(frac=0.7, random_state=RNG)
        ads = ads.drop(train_ads.index)

        # test 20%
        test_ads = ads.sample(frac=0.7, random_state=RNG)
        test_ads.to_csv('train_ads.csv', header=True, encoding='utf-8')

        validate_ads = ads.drop(test_ads.index)
        validate_ads.to_csv('validate_ads.csv', header=True, encoding='utf-8')

        ads = train_ads
        return ads

    def train_living_area(self, ads):
        filterd_ads = ads.dropna(subset=['living_area'])
        filterd_ads = filterd_ads.drop(['price'], axis=1)
        logging.debug("Find best estimator for ExtraTreesRegressor")
        X, y = generate_matrix(filterd_ads, 'living_area')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
        best_md = None
        best_estimator = None
        # Use Warm start to use calculated trees
        model = ExtraTreesRegressor(n_estimators=100, warm_start=True, n_jobs=-1, random_state=RNG)
        for estimator in range(100,
                            self.settings['living_area_prediction']['limit'],
                            self.settings['living_area_prediction']['step']):
            logging.debug("Estimator: {}".format(estimator))
            model.n_estimators = estimator
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            md = mape(y_test, y_pred)
            logging.info("Train Livint area with estimator: {}".format(estimator))
            train_statistics(y_test, y_pred, 'Living area')
            if best_md is None or best_md > md:  # Wenn altes md grösser ist als neues md
                best_estimator = estimator
                best_md = md
                logging.info("Better result with estimator: {}".format(estimator))
                # Store model
                joblib.dump(model, '{}/living_area.pkl'.format(self.model_folder))

        self.settings['living_area_prediction']['estimator'] = best_estimator
        # Save best c for all features
        with open('{}/settings.json'.format(self.directory), 'w') as f:
            f.write(json.dumps(self.settings))
        return ads


    def train_outlier_detection(self, ads):

        """Check which contamination is the best for the features
        Run isolation forest with different contamination and check
        difference in the standard derivation.
        If the diff is < 1 we found our c
        """
        for feature in self.settings['anomaly_detection']['features']:
            # Do not run outlier detection if model exists
            if os.path.isfile('{}/isolation_forest_{}.pkl'.format(self.model_folder, feature)):
                logging.info("Outlier detection for feature {} exists.".format(feature))
                continue

            logging.info("Check feature: {}".format(feature))
            # Only use the this specific feature with our target
            tmp_ad = ads[[feature, self.goal]]
            # Initialize std and std_percent
            std = [np.std(tmp_ad[feature].astype(int))]
            std_percent = [100]
            difference = [0]
            # We are in train phase so best_c should always be 0
            best_c, self.settings['anomaly_detection'][feature] = 0, 0
            last_model, cls_ = None, None
            chosen_std_percent = 0

            # Run isolation forest for diffrent contamination
            for c in np.arange(0.01, self.settings['anomaly_detection']['limit'],
                              self.settings['anomaly_detection']['step']):
                last_model, cls_ = cls_ , IsolationForest(max_samples=0.6,
                                                        contamination=c,
                                                        n_estimators=self.settings['anomaly_detection']['estimator'],
                                                        random_state=RNG)
                logging.debug("Check C: {}".format(c))
                cls_.fit(tmp_ad.values)
                outlierIdx = cls_.predict(tmp_ad.values)
                # Remove entries which are detected as outliers
                filtered = tmp_ad.drop(tmp_ad.index[np.where(outlierIdx == -1)[0]])
                # Calculate standard derivation of the new filtered ads
                std.append(np.std(filtered[feature].astype(int)))
                std_percent.append((std[-1]/std[0])*100)
                logging.info("New std: with c {}: {}".format(c, std[-1]))
                logging.info("New std in percent with c {}: {}".format(c, std_percent[-1]))
                # Calculate diff from last standard derivation to check if we found our contamination
                diff = std_percent[-2] - std_percent[-1]
                logging.info("Diff when using c {}: {}".format(c, diff))
                # We stop after diff is the first time < 1. So best_c must be 0
                # But we can not stop the calculation because we want to make the whole diagramm
                if diff < 2.5 and best_c == 0:
                    # We do not need this c, we need the last c - step
                    best_c = np.around(c - self.settings['anomaly_detection']['step'], 2)
                    chosen_std_percent = std_percent[-2]
                    joblib.dump(last_model, '{}/isolation_forest_{}.pkl'.format(self.model_folder, feature))
                difference.append(diff)

            # Store best c in our settings to use it later
            self.settings['anomaly_detection'][feature] = best_c if best_c > 0 else 0 + self.settings['anomaly_detection']['step']

            # Plot stuff
            # Create directory if not exists
            if not os.path.exists('{}/outlier_detection'.format(self.image_folder)):
                logging.info("Directory for outliers does not exists. Create one")
                os.makedirs('{}/outlier_detection'.format(self.image_folder))
            logging.info("Best C for feature {}: {} chosen_std: {}".format(feature, best_c, chosen_std_percent))
            # Plot the standard derivation for this feature
            fig, ax1 = plt.subplots()
            plt.title('Reduction of σ for feature {}'.format(feature.replace("_", " ")))
            plt.plot(list(map(lambda x: x*100, np.arange(0.0, self.settings['anomaly_detection']['limit'],
                                    self.settings['anomaly_detection']['step']))),
                     std_percent, c='r')
            plt.ylabel('Decline of σ in %')
            plt.xlabel('% removed of {} outliers'.format(feature.replace("_", " ")))
            # Draw lines to choosen c
            plt.plot([best_c*100, best_c*100], [40, chosen_std_percent], linewidth=1, color='b', linestyle='--')
            plt.plot([0, best_c*100], [chosen_std_percent, chosen_std_percent], linewidth=1, color='b', linestyle='--')

            ax1.set_xlim([0, 10])
            ax1.set_ylim([35, 100])
            plt.axis('tight')
            plt.savefig('{}/outlier_detection/{}_std.png'.format(self.image_folder, feature))
            plt.close()

            # Plot the difference between the standard derivation
            fig, ax1 = plt.subplots()
            ax1.set_title('Difference of σ from previous σ for feature: {}'.format(feature.replace("_", " ")))
            ax1.plot(list(np.arange(0.0, self.settings['anomaly_detection']['limit'],
                                    self.settings['anomaly_detection']['step'])), difference, c='r')

            ax1.set_ylabel('% decline of σ to previous σ')
            ax1.set_xlabel('% removed of {} outliers'.format(feature.replace("_", " ")))
            plt.savefig('{}/outlier_detection/{}_diff_of_std.png'.format(self.image_folder, feature))
            plt.close()

        # Save best c for all features
        with open('{}/settings.json'.format(self.directory), 'w') as f:
            f.write(json.dumps(self.settings))
        return ads

    def train_extraTreeRegression_withKFold(self, ads):
        ads['price'] = np.log(ads['price'])

        X, y = generate_matrix(ads.reindex(), 'price')

        #X = X.drop(['price_log'], axis=1)
        y = np.exp(y)
        X, y = X.values, y.values

        kf = KFold(n_splits=10)
        sum_mape = 0
        sum_mdape = 0
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model = ExtraTreesRegressor(n_estimators=100, warm_start=False, n_jobs=-1, random_state=RNG)
            model.fit(X_train, y_train)
            y_pred = np.exp(model.predict(X_test))
            train_statistics(y_test, y_pred, title="ExtraTree_train_{}".format(100))
            sum_mape += mape(y_test, y_pred)
            sum_mdape += mdape(y_test, y_pred)
        logging.info("MAPE: {}".format(sum_mape/kf.get_n_splits(X)))
        logging.info("MdAPE: {}".format(sum_mdape/kf.get_n_splits(X)))
        return ads


    def extraTreeRegression2(self, ads):
        X, y = generate_matrix(ads, 'price')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        model = ExtraTreesRegressor(n_estimators=700, n_jobs=-1, random_state=RNG)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        md = mape(y_test, y_pred)
        train_statistics(y_test, y_pred, title="ExtraTree_train")
        plot(y_test, y_pred, self.image_folder, show=True, title="ExtraTree_train")



    def train_extraTeeRegression(self, ads):
        # remove = ['characteristics', 'description']
        # filterd_ads = ads.drop(remove, axis=1)
        X_train, y_train = generate_matrix(ads, 'price')

        test_ads = pd.read_csv('train_ads.csv', index_col=0, engine='c')
        X_test, y_test = generate_matrix(test_ads, 'price')

        valid_ads = pd.read_csv('validate_ads.csv', index_col=0, engine='c')
        X_valid, y_valid = generate_matrix(valid_ads, 'price')
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        best_md = None
        best_estimator = None
        model = ExtraTreesRegressor(n_estimators=100, warm_start=True, n_jobs=-1, random_state=RNG)
        logging.debug("Find best estimator for ExtraTreesRegressor")
        for estimator in range(100,
                               self.settings['extraTreeRegression']['limit'],
                               self.settings['extraTreeRegression']['step']):
            logging.debug("Estimator: {}".format(estimator))
            model.n_estimators = estimator

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            md = mape(y_test, y_pred)
            logging.info("Stats for estimator: {}".format(estimator))
            train_statistics(y_test, y_pred, title="ExtraTree_train_{}".format(estimator))
            logging.info("VALIDATION")
            y_pred = model.predict(X_valid)
            train_statistics(y_valid, y_pred, title="ExtraTree_train_{}".format(estimator))

            plot(y_test, y_pred, self.image_folder, show=False, title="ExtraTree_train_{}".format(estimator))
            # Wenn altes md grösser ist als neues md haben wir ein kleiners md somit bessers Ergebnis
            if best_md is None or best_md > md:
                best_estimator = estimator
                best_md = md
                logging.info("Better result with estimator: {}".format(estimator))
                # Store model
                joblib.dump(model, '{}/extraTree.pkl'.format(self.model_folder))

        self.settings['extraTreeRegression']['estimator'] = best_estimator
        # Save best c for all features
        with open('{}/settings.json'.format(self.directory), 'w') as f:
            f.write(json.dumps(self.settings))
        return ads

    def piero_extraTreeRegression(self, ads):
        ads['price'] = np.log(ads['price'])

        X, y = generate_matrix(ads, 'price')

        logging.info('split')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RNG)

        estimator = 100
        model = ExtraTreesRegressor(n_estimators=estimator, warm_start=True, n_jobs=-1, min_samples_leaf=3)

        logging.info('fit')
        model.fit(X_train, y_train)
        logging.info('predict')
        y_pred = model.predict(X_test)
        train_statistics(y_test, y_pred, title="ExtraTree_train_{}".format(estimator))

        plot(y_test, y_pred, self.image_folder, show=False, title="ExtraTree_train_{}".format(estimator))
        joblib.dump(model, '{}/extraTree.pkl'.format(self.model_folder))

        return ads



    def combinedEnsemble_settings(self, ads):
        self.n_estimators = 700
        self.min_samples_leaf = 5
        self.ensemble_estimator = 'extratrees'

        self.combinedEnsemble_identifier = 'combinedEnsemble_{}_{}_{}'.format(self.ensemble_estimator, self.n_estimators, self.min_samples_leaf)
        self.combinedEnsemble_identifier_long = 'combinedEnsemble ensemble_estimator={} n_estimators={}, min_samples_leaf={}'.format(self.ensemble_estimator, self.n_estimators, self.min_samples_leaf)

        self.estimators = {
            'linear': LinearRegression(normalize=True),
            'ridge': RidgeCV(alphas=[0.01, 0.03, 0.1, 0.3, 1, 3, 10]),
            'knn2': KNeighborsRegressor(n_neighbors=2, weights='distance'),
            'knn5': KNeighborsRegressor(n_neighbors=5, weights='distance'),
        }


        return ads

    def combinedEnsemble_train(self, ads):
        X, y = generate_matrix(ads, 'price')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RNG)

        model = CombinedEnsemble(
            ensemble_estimator=ExtraTreesRegressor(n_estimators=self.n_estimators, min_samples_leaf=self.min_samples_leaf, n_jobs=-1),
        )

        logging.info('Fit {}'.format(self.combinedEnsemble_identifier_long))
        model.fit(X_train, y_train)

        logging.info("Fit finished. Save model")
        joblib.dump(model, '{}/{}.pkl'.format(self.model_folder, self.combinedEnsemble_identifier))

        self.combinedEnsemble = model

        return ads

    def combinedEnsemble_load(self, ads):
        self.combinedEnsemble = joblib.load('{}/{}.pkl'.format(self.model_folder, self.combinedEnsemble_identifier))

        return ads

    def combinedEnsemble_test(self, ads):
        X, y = generate_matrix(ads, 'price')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RNG)

        model = self.combinedEnsemble

        logging.info("Begin testing stage 2 estimators.")
        logging.info("-"*80)
        logging.info("")

        for name, estimator in self.estimators.items():
            logging.info('Predict stage 2 estimator: {}'.format(name))
            model.estimator2 = estimator
            y_pred = model.predict(X_test, verbose=True)

            logging.info('Statistics for stage 2 estimator: {}'.format(name))
            train_statistics(y_test, y_pred, title="{} estimator2={}".format(self.combinedEnsemble_identifier_long, name))
            plot(y_test, y_pred, self.image_folder, show=False, title="{}_{}".format(self.combinedEnsemble_identifier, name))
            logging.info("-"*80)
            logging.info("")

        logging.info('Finished')

        return ads