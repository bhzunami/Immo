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
        self.pipeline = [
            self.simple_stats('Before Transformation'),
            self.replace_zeros_with_nan,
            self.transform_build_year,
            self.transform_build_renovation,
            self.transform_noise_level,
            # self.transform_floor,  Floor will be droped
            self.drop_floor,
            self.transform_num_rooms,
            self.transfrom_description,
            self.transform_living_area,
            self.simple_stats('After Transformation'),
            self.transform_tags,
            self.transform_features,
            self.transform_onehot,
            self.train_outlier_detection,
            self.outlier_detection,
            self.train_living_area,
            self.predict_living_area,
            self.transform_misc_living_area,
            self.train_extraReeRegression]


    def transform_description(self, ads):
        pdb.set_trace()

        def stemm_words(row):
            language = detect_language(row.description)
            #letters_only = re.sub("[^a-z0-9üäöèéàêâ]", " ", str(row.desc.lower()))
            letters_only = row.description.split()
            stops = set(json.load(open('{}_stop_words.json'.format(language))))
            #stops = set(stopwords.words(language))
            meaningful_words = [w for w in letters_only if not w in stops]
            stemmer = SnowballStemmer(language)
            stemmed_words = [stemmer.stem(w) for w in meaningful_words]
            return " ".join(stemmed_words)

        ads['desc'] = ads.apply(stemm_words, axis=1)
        return ads


        return ads
    def train_living_area(self, ads):
        filterd_ads = ads.dropna(subset=['living_area'])
        filterd_ads = filterd_ads.drop(['characteristics', 'price_brutto', 'description'], axis=1)
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
            if best_md is None or best_md > md:  # Wenn altes md grösser ist als neues md
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
            chosen_std_percent = 0

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
                    chosen_std_percent = std_percent[-2]
                    joblib.dump(last_model, '{}/isolation_forest_{}.pkl'.format(self.model_folder, feature))
                difference.append(diff)

            # Store best c in our settings to use it later
            self.settings['anomaly_detection'][feature] = best_c if best_c > 0 else 0 + self.settings['anomaly_detection']['step']

            # Plot stuff
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
            plt.savefig('{}/{}_std.png'.format(self.image_folder, feature))
            plt.close()

            # Plot the difference between the standard derivation
            fig, ax1 = plt.subplots()
            ax1.set_title('Difference of σ from previous σ for feature: {}'.format(feature.replace("_", " ")))
            ax1.plot(list(np.arange(0.0, self.settings['anomaly_detection']['limit'],
                                    self.settings['anomaly_detection']['step'])), difference, c='r')

            ax1.set_ylabel('% decline of σ to previous σ')
            ax1.set_xlabel('% removed of {} outliers'.format(feature.replace("_", " ")))
            plt.savefig('{}/{}_diff_of_std.png'.format(self.image_folder, feature))
            plt.close()

        # Save best c for all features
        with open('{}/settings.json'.format(self.directory), 'w') as f:
            f.write(json.dumps(self.settings))
        return ads

    def train_extraReeRegression(self, ads):
        remove = ['characteristics', 'description']
        filterd_ads = ads.drop(remove, axis=1)
        X, y = generate_matrix(filterd_ads, 'price_brutto')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
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
            # Wenn altes md grösser ist als neues md haben wir ein kleiners md somit bessers Ergebnis
            if best_md is None or best_md > md:
                best_estimator = estimator
                best_md = md
                logging.info("Better result with estimator: {}".format(estimator))
                train_statistics(y_test, y_pred, title="ExtraTree_train")
                plot(y_test, y_pred, self.image_folder, show=False, title="ExtraTree_train")
                # Store model
                joblib.dump(model, '{}/extraTree.pkl'.format(self.model_folder))

        self.settings['extraTreeRegression']['estimator'] = best_estimator
        # Save best c for all features
        with open('{}/settings.json'.format(self.directory), 'w') as f:
            f.write(json.dumps(self.settings))
        return ads

