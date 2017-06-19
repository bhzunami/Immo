import matplotlib
matplotlib.use('Agg')
import os
import pdb
import logging
import json
import argparse
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import ast
# Scikit
from sklearn.ensemble import ExtraTreesRegressor, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
# NLTK
import nltk
from nltk.corpus import stopwords # Import the stop word list
from nltk.stem import SnowballStemmer

from a_detection import AnomalyDetection
from helper import generate_matrix, ape, mape, mdape, gen_subplots, plot, train_statistics

RNG = np.random.RandomState(42)
from pipeline import Pipeline


class TrainPipeline(Pipeline):
    def __init__(self, goal, settings, directory):
        super().__init__(goal, settings, directory)
        self.pipeline = [
            self.simple_stats('Before Transformation'),
            self.transform_build_year,
            self.transform_build_renovation,
            self.replace_zeros_with_nan,
            self.transform_noise_level,
            self.transform_floor,
            self.transform_num_rooms,
            self.transfrom_description,
            self.transform_living_area,
            self.drop_floor,
            self.simple_stats('After Transformation'),
            self.transform_tags,
            self.transform_onehot,
            self.transform_features,
            #self.train_outlier_detection,
            self.outlier_detection,
            #self.train_living_area,
            self.predict_living_area,
            #self.transform_desc,
            #self.transform_misc_living_area,
            self.extraTreeRegression]

    def train_living_area(self, ads):
        filterd_ads = ads.dropna(subset=['living_area'])
        filterd_ads = filterd_ads.drop(['characteristics', 'price_brutto', 'description'], axis=1)
        logging.debug("Find best estimator for ExtraTreesRegressor")
        X, y = generate_matrix(filterd_ads, 'living_area')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
        best_md = 100
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
            md = mdape(y_test, y_pred)
            if best_md > md:  # Wenn altes md grösser ist als neues md
                best_estimator = estimator
                best_md = md
                logging.info("Better result with estimator: {}".format(estimator))
                train_statistics(y_test, y_pred, 'Living area')
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
            logging.debug("Check feature: {}".format(feature))
            # Only use the this specific feature with our target
            tmp_ad = ads[[feature, self.goal]]
            # Initialize std and std_percent
            std = [np.std(tmp_ad[feature].astype(int))]
            std_percent = [100]
            difference = [0]
            # We are in train phase so best_c should always be 0
            best_c, self.settings['anomaly_detection'][feature] = 0, 0
            last_model, cls_ = None, None

            # Run isolation forest for diffrent contamination
            for c in np.arange(0.01, self.settings['anomaly_detection']['limit'],
                              self.settings['anomaly_detection']['step']):
                last_model, cls_ = cls_ , IsolationForest(max_samples=0.7,
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
                # Calculate diff from last standard derivation to check if we found our contamination
                diff = std_percent[-2] - std_percent[-1]
                # We stop after diff is the first time < 1. So best_c must be 0
                # But we can not stop the calculation because we want to make the whole diagramm
                logging.debug("Diff: {}".format(diff))
                if diff < 2.5 and best_c == 0:
                    # We do not need this c, we need the last c - step
                    best_c = np.around(c - self.settings['anomaly_detection']['step'], 2)
                    joblib.dump(last_model, '{}/isolation_forest_{}.pkl'.format(self.model_folder, feature))
                difference.append(diff)

            # Store best c in our settings to use it later
            self.settings['anomaly_detection'][feature] = best_c if best_c > 0 else 0 + self.settings['anomaly_detection']['step']
            # Plot stuff
            logging.info("Best C for feature {}: {}".format(feature, best_c))
            # Plot the standard derivation for this feature
            fig, ax1 = plt.subplots()
            ax1.set_title('{}'.format(feature))
            ax1.plot(list(np.arange(0.0, self.settings['anomaly_detection']['limit'],
                                    self.settings['anomaly_detection']['step'])),
                    std_percent, c='r')
            ax1.set_ylabel('Reduction of STD deviation in %')
            ax1.set_xlabel('% removed of {}'.format(feature))
            plt.savefig('{}/{}_std.png'.format(self.image_folder, feature))
            plt.close()

            # Plot the difference between the standard derivation
            plt.plot(list(np.arange(0.0, self.settings['anomaly_detection']['limit'],
                                    self.settings['anomaly_detection']['step'])), difference, c='r')
            plt.savefig('{}/{}_diff_of_std.png'.format(self.image_folder, feature))
            plt.close()

        # Save best c for all features
        with open('{}/settings.json'.format(self.directory), 'w') as f:
            f.write(json.dumps(self.settings))
        return ads


    def extraTreeRegression(self, ads):
        filterd_ads = ads.drop(['characteristics', 'description'], axis=1)
        logging.debug("Find best estimator for ExtraTreesRegressor")
        X, y = generate_matrix(filterd_ads, 'price_brutto')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
        model = ExtraTreesRegressor(n_estimators=700)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        train_statistics(y_test, y_pred, title="ExtraTree")
        plot(y_test, y_pred, self.image_folder, show=False, title="ExtraTree")
        return ads