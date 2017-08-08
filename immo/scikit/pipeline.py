import os
import pdb
import logging
import json
import argparse
import datetime
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import ast
# Scikit
# Scikit
from sklearn.ensemble import ExtraTreesRegressor, IsolationForest
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.linear_model import LassoLarsCV, Ridge, RidgeCV, LassoCV, Lasso, LinearRegression, LogisticRegression
from sklearn.metrics.scorer import make_scorer
import xgboost as xgb
import lightgbm as lgb
from sklearn import cross_validation, metrics   #Additional scklearn functions
# NLTK
#import nltk
#from nltk.corpus import stopwords # Import the stop word list
#from nltk.stem import SnowballStemmer

from .a_detection import AnomalyDetection
from .helper import generate_matrix, ape, mape, mdape, gen_subplots, plot, train_statistics
from .combined_ensemble import CombinedEnsemble

RNG = np.random.RandomState(42)

def score_function(y_test, y_pred):
    return mape(y_test, y_pred)

scorer = make_scorer(score_function, greater_is_better=False)
class Pipeline():
    def __init__(self, goal, settings, directory):
        self.goal = goal
        self.settings = settings
        self.directory = directory
        self.image_folder = os.path.abspath(os.path.join(directory, settings['image_folder']))
        self.model_folder = os.path.abspath(os.path.join(directory, settings['model_folder']))
        # Create folder if they do not exist
        if not os.path.exists('{}'.format(self.image_folder)):
            logging.info("Directory for images does not exists. Create one")
            os.makedirs('{}'.format(self.image_folder))

        if not os.path.exists('{}'.format(self.model_folder)):
            logging.info("Directory for models does not exists. Create one")
            os.makedirs('{}'.format(self.model_folder))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Load and Save Methods
    def load_csv(self, filename):
        def load_csv_inner(ads):
            try:
                return pd.read_csv(filename, index_col=0, engine='c')
            except FileNotFoundError:
                logging.info("File {} does not exist. Please run data_analyse.py first".format(filename))
                return None
        return load_csv_inner

    def save_as_df(self, name):
        def inner_save_as_df(ads):
            joblib.dump(ads, name)
            return ads
        return inner_save_as_df

    def load_df(self, name):
        def inner_load_df(ads):
            return joblib.load(name)
        return inner_load_df

    def remove(self, name):
        def inner_remove(ads):
            try:
                os.remove('{}/{}'.format(self.model_folder, name))
            except Exception:
                logging.error("Could not remove pkl")
                pass
        return inner_remove

    def load_pipeline(self, pipeline):
        filename = "{}/ads_prepared.pkl".format(self.model_folder)
        if os.path.isfile(filename):
            return [self.load_df(filename)]
        else:
            return pipeline
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Feature engineering
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    def cleanup(self, remove):
        def inner_cleanup(ads):
            logging.info("Start preparing dataset")
            ads = ads.drop(ads[ads['price'] < 10].index)
            ads = ads.drop(ads[ads['build_year'] < 1200].index)
            ads = ads.drop(ads[ads['build_year'] > 2030].index)
            ads = ads.drop(ads[(ads['ogroup'] == 'raum') | (ads['ogroup'] == 'invalid')].index)
            ads = ads.drop_duplicates(keep='first')

            # Remove empty prices
            dropna = ['price', 'build_year', 'num_rooms', 'living_area']
            ads = ads.dropna(subset=dropna)

            # Remove unwanted cols
            return ads.drop(remove, axis=1)
        return inner_cleanup

    def simple_stats(sefl, title):
        """Show how many NaN has one Feature and how many we can use
        """
        def run(ads):
            total_amount_of_data = ads.shape[0]
            logging.info("{}".format(title))
            logging.info("="*70)
            logging.info("We have total {} values".format(total_amount_of_data))
            logging.info("{:25} | {:17} | {:8}".format("Feature",
                                                       "NaN-Values",
                                                       "usable Values"))
            logging.info("-"*70)
            total_nan = 0
            total_use = total_amount_of_data

            for key in ads.keys():
                if key == 'id' or key == 'Unnamed':  # Keys from pandas we do not want
                    continue
                # for i, key in KEYS:
                nan_values = ads[key].isnull().sum()
                useful_values = total_amount_of_data - nan_values

                # Sum up
                total_nan += nan_values
                total_use = total_use if total_use < useful_values else useful_values

                logging.info("{:25} | {:8} ({:5.2f}%) | {:8} ({:3.0f}%)".format(key,
                        nan_values, (nan_values/total_amount_of_data)*100,
                        useful_values, (useful_values/total_amount_of_data)*100))

            logging.info("-"*70)
            logging.info("{:25} | {:17} | {}".format('Total', total_nan, total_use))
            return ads

        return run

    def show_crawler_stats(self, ads):
        logging.info(ads.groupby('crawler')['price'].count())
        return ads

    def transform_noise_level(self, ads):
        pdb.set_trace()
        """ If we have no nose_level at the address
        we use the municipality noise_level
        """
        def lambdarow(row):
            if np.isnan(row.noise_level):
                return row.m_noise_level
            return row.noise_level

        if 'noise_level' in ads.keys():
            ads['noise_level'] = ads.apply(lambdarow, axis=1)
            return ads
        return ads


    def replace_zeros_with_nan(self, ads):
        """ replace 0 values into np.nan for statistic
        """
        ads.loc[ads.living_area == 0, 'living_area'] = np.nan
        ads.loc[ads.num_rooms == 0, 'num_rooms'] = np.nan
        return ads

    def transform_misc_living_area(self, ads):
        ads['avg_room_area'] = ads['living_area'] / ads['num_rooms']
        return ads


    def transform_build_renovation(self, ads):
        """Set was_renovated to 1 if we have a date in renovation_year
        """
        ads['was_renovated'] = ads.apply(lambda row: not np.isnan(row['last_renovation_year']), axis=1)

        def last_const(row):
            current_year = datetime.date.today().year
            if row['build_year'] >= current_year or row['last_renovation_year'] >= current_year:
                return 0
            elif np.isnan(row['last_renovation_year']):
                return current_year - row['build_year']
            else:
                return current_year - row['last_renovation_year']

        ads['last_construction'] = ads.apply(last_const, axis=1)

        return ads.drop(['last_renovation_year'], axis=1)


    def transform_onehot(self, ads):
        """Build one hot encoding for all columns with string as value
        """
        logging.debug("Features: {}".format(ads.keys()))
        ads = pd.get_dummies(ads, columns=self.settings.get('one_hot_columns'))
        return ads

    def transform_tags(self, ads):
        """Transform tags
        """
        with open(os.path.join(self.directory, '../crawler/taglist.txt')) as f:
            search_words = set(["tags_" + x.split(':')[0] for x in f.read().splitlines()])

        template_dict = dict.fromkeys(search_words, 0)

        def transformer(row):
            the_dict = template_dict.copy()

            for tag in ast.literal_eval(row.tags):
                the_dict["tags_" + tag] = 1

            return pd.Series(the_dict)

        tag_columns = ads.apply(transformer, axis=1)
        return ads.drop(['tags'], axis=1).merge(tag_columns, left_index=True, right_index=True)


    def predict_living_area(self, ads):
        """If living area is missing try to predict one
        and set the predicted flag to 1
        """
        try:
            model = joblib.load('{}/living_area.pkl'.format(self.model_folder))
        except FileNotFoundError:
            logging.error("Could not load living area model. Did you forget to train living area?")
            return ads.dropna(subset=['living_area'])

        tempdf = ads.drop(['price'], axis=1)

        ads.living_area_predicted = 0
        nan_idxs = tempdf.living_area.index[tempdf.living_area.apply(np.isnan)]
        if len(nan_idxs) > 0:
            ads.loc[nan_idxs, 'living_area'] = model.predict(tempdf.drop(['living_area'], axis=1).ix[nan_idxs])
            ads.loc[nan_idxs, 'living_area_predicted'] = 1

        return ads

    def transform_features(self, ads):
        """Transfrom features to more global one
        """
        # Merge some Features:
        ads['bath'] = np.where((ads['tags_badewanne'] == 1) |
                               (ads['tags_badezimmer'] == 1) |
                               (ads['tags_dusche'] == 1) |
                               (ads['tags_lavabo'] == 1), 1, 0)

        ads['interior'] = np.where((ads['tags_anschluss'] == 1) |
                                   (ads['tags_abstellplatz'] == 1) |
                                   (ads['tags_cheminée'] == 1) |
                                    (ads['tags_eingang'] == 1) |
                                    (ads['tags_esszimmer'] == 1) |
                                    (ads['tags_gross'] == 1) |
                                    (ads['tags_heizung'] == 1) |
                                    (ads['tags_lift'] == 1) |
                                    (ads['tags_minergie'] == 1) |
                                    (ads['tags_schlafzimmer'] == 1) |
                                    (ads['tags_wohnzimmer'] == 1) |
                                    (ads['tags_rollstuhlgängig'] == 1) |
                                    (ads['tags_tv'] == 1) |
                                    (ads['tags_küche'] == 1) |
                                    (ads['tags_waschküche'] == 1) |
                                    (ads['tags_waschmaschine'] == 1) |
                                    (ads['tags_wc'] == 1) |
                                    (ads['tags_zimmer'] == 1), 1, 0)

        ads['exterior'] = np.where((ads['tags_aussicht'] == 1) |
                                (ads['tags_balkon'] == 1) |
                                (ads['tags_garten'] == 1) |
                                (ads['tags_garage'] == 1) |
                                (ads['tags_lage'] == 1) |
                                (ads['tags_liegenschaft'] == 1) |
                                (ads['tags_parkplatz'] == 1) |
                                (ads['tags_sitzplatz'] == 1) |
                                (ads['tags_terrasse'] == 1), 1, 0)

        ads['neighbourhood'] = np.where((ads['tags_autobahnanschluss'] == 1) |
                                        (ads['tags_einkaufen'] == 1) |
                                        (ads['tags_kinderfreundlich'] == 1) |
                                        (ads['tags_kindergarten'] == 1) |
                                        (ads['tags_oberstufe'] == 1) |
                                        (ads['tags_primarschule'] == 1) |
                                        (ads['tags_quartier'] == 1) |
                                        (ads['tags_ruhig'] == 1) |
                                        (ads['tags_sommer'] == 1) |
                                        (ads['tags_verkehr'] == 1) |
                                        (ads['tags_zentral'] == 1), 1, 0)

        # Drop the concatenated features
        drop_features = ['tags_badewanne', 'tags_badezimmer', 'tags_dusche', 'tags_lavabo', 'tags_anschluss',
                        'tags_abstellplatz', 'tags_cheminée', 'tags_eingang', 'tags_esszimmer', 'tags_gross',
                        'tags_heizung', 'tags_lift', 'tags_minergie', 'tags_schlafzimmer', 'tags_wohnzimmer',
                        'tags_rollstuhlgängig', 'tags_tv', 'tags_küche', 'tags_waschküche', 'tags_waschmaschine',
                        'tags_wc', 'tags_zimmer', 'tags_aussicht', 'tags_balkon', 'tags_garten', 'tags_garage',
                        'tags_lage', 'tags_liegenschaft', 'tags_parkplatz', 'tags_sitzplatz', 'tags_terrasse',
                        'tags_autobahnanschluss', 'tags_einkaufen', 'tags_kinderfreundlich',
                        'tags_kindergarten', 'tags_oberstufe', 'tags_primarschule', 'tags_quartier',
                        'tags_ruhig', 'tags_sommer', 'tags_verkehr', 'tags_zentral']

        return ads.drop(drop_features, axis=1)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Outlier detection
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    
    def outlier_detection(self, ads):
        """Detect outliers we do not want in our training phase
        The outlier model must be trained first
        """
        if os.path.isfile("{}/ads_clean.pkl".format(self.model_folder)):
            logging.info("Clean file found skipping outlier detection load data from file.")
            return self.load_df("{}/ads_clean.pkl".format(self.model_folder))(ads)

        meshgrid = {
            'build_year': np.meshgrid(np.linspace(0, max(ads['build_year']), 400),
                                      np.linspace(0, max(ads['price']), 1000)),
            'num_rooms': np.meshgrid(np.linspace(-1, max(ads['num_rooms']), 400),
                                     np.linspace(0, max(ads['price']), 1000)),
            'living_area': np.meshgrid(np.linspace(0, max(ads['living_area']), 400),
                                      np.linspace(0, max(ads['price']), 1000)),
            'last_construction': np.meshgrid(np.linspace(0, max(ads['last_construction']), 400),
                                             np.linspace(0, max(ads['price']), 1000)),
            'noise_level': np.meshgrid(np.linspace(0, max(ads['noise_level']), 400),
                                       np.linspace(0, max(ads['price']), 1000))
        }

        ads = self.outlier_detection_light(ads, meshgrid)

        return self.save_as_df("{}/ads_clean.pkl".format(self.model_folder))(ads)


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
            plt.savefig('{}/outlier_detection/{}_std.png'.format(self.image_folder, feature), dpi=250)
            plt.close()

            # Plot the difference between the standard derivation
            fig, ax1 = plt.subplots()
            ax1.set_title('Difference of σ from previous σ for feature: {}'.format(feature.replace("_", " ")))
            ax1.plot(list(np.arange(0.0, self.settings['anomaly_detection']['limit'],
                                    self.settings['anomaly_detection']['step'])), difference, c='r')

            ax1.set_ylabel('% decline of σ to previous σ')
            ax1.set_xlabel('% removed of {} outliers'.format(feature.replace("_", " ")))
            plt.savefig('{}/outlier_detection/{}_diff_of_std.png'.format(self.image_folder, feature), dpi=250)
            plt.close()

        # Save best c for all features
        with open('{}/settings.json'.format(self.directory), 'w') as f:
            f.write(json.dumps(self.settings))
        return ads



    def predict(self, model):
        def inner_predict(ads):
            X, y = generate_matrix(ads, 'price')
            X, y = X.values, y.values
            idx = 0
            model = joblib.load('{}/{}.pkl'.format(self.model_folder, name))
            for train_index, test_index in KFold(n_splits=3, shuffle=True).split(X):
                logging.info('New split')
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                logging.info("Size of training data: {}".format(len(X_train)))
                logging.info("Size of testing data: {}".format(len(X_test)))
                y_pred = model.predict(X_test)
                train_statistics(y_test, y_pred, title="{}_{}".format(name, idx))
                plot(y_test, y_pred, self.image_folder, show=False, title="{}_{}".format(name, idx))
                idx += 1
            return ads

        return inner_predict

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Linear Regression
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -     
    def linear_regression(self, ads):
        X, y = generate_matrix(ads, 'price')
        X, y = X.values, y.values

        for train_index, test_index in KFold(n_splits=3, shuffle=True).split(X):
            logging.info('Linear regression new split')
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            logging.info("Size of training data: {}".format(len(X_train)))
            logging.info("Size of testing data: {}".format(len(X_test)))
            linreg = LinearRegression(normalize=True, n_jobs=-1)
            linreg.fit(X_train, y_train)
            y_pred = linreg.predict(X_test)
            train_statistics(y_test, y_pred)
            plot(y_test, y_pred, self.image_folder, show=False, title="simpel_linear_regresion")
        return ads

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Ridge
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    def train_ridge(self, ads):
        X, y = generate_matrix(ads, 'price')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        ridge = RidgeCV(alphas=[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30])
        ridge.fit(X_train, y_train)
        alpha = ridge.alpha_

        logging.info("Try again for more precision with alphas centered around " + str(alpha))
        ridge = RidgeCV(alphas=[alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, 
                                alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,
                                alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4], 
                        cv=5)
        ridge.fit(X_train, y_train)

        alpha = ridge.alpha_
        logging.info("Best alpha: {}".format(alpha))
        ridgereg = Ridge(alpha=alpha, normalize=True)
        ridgereg.fit(X_train, y_train)
        joblib.dump(ridgereg, '{}/ridge.pkl'.format(self.model_folder))
        y_pred = ridgereg.predict(X_test)
        train_statistics(y_test, y_pred)
        plot(y_test, y_pred, self.image_folder, show=False, title="ridge_regression")
        return ads

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Lasso
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    def train_lasso(self, ads):
        X, y = generate_matrix(ads, 'price')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        lasso = LassoCV(alphas=[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30])
        lasso.fit(X_train, y_train)
        alpha = lasso.alpha_

        logging.info("Try again for more precision with alphas centered around " + str(alpha))
        lasso = LassoCV(alphas=[alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, 
                                alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,
                                alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4], 
                        cv=5)
        lasso.fit(X_train, y_train)

        alpha = lasso.alpha_
        logging.info("Best alpha: {}".format(alpha))

        lassoreg = Lasso(alpha=alpha, normalize=True, max_iter=1e5)
        lassoreg.fit(X_train, y_train)
        joblib.dump(lassoreg, '{}/lasso.pkl'.format(self.model_folder))
        y_pred = lassoreg.predict(X_test)
        train_statistics(y_test, y_pred)
        plot(y_test, y_pred, self.image_folder, show=False, title="lasso_regression")
        return ads

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # K nearest Neighbour
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    def train_kneighbours(self, ads):
        X, y = generate_matrix(ads, 'price')
        X, y = X.values, y.values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        # parameters = {"n_neighbors": [2, 3, 5, 10], "leaf_size":[50, 100, 200]}
        parameters = {"n_neighbors": [2], "leaf_size":[50]}
        neigh = KNeighborsRegressor(weights='distance', n_jobs=-1)
        gd = GridSearchCV(neigh, parameters, verbose=1, scoring=scorer)
        logging.info("Start Fit")
        gd.fit(X_train, y_train)
        logging.info("Best score: {}".format(gd.best_score_))
        logging.info("Best params: {}".format(gd.best_params_))
        logging.info("result: {}".format(gd.cv_results_))
        return ads

    def fit_kneighbours(self, params):
        def fit(ads):
            X, y = generate_matrix(ads, 'price')
            X, y = X.values, y.values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
            
            neigh = KNeighborsRegressor(weights='distance', n_jobs=-1,
                                        leaf_size=params.get('leaf_size', 100),
                                        n_neighbors=params.get('n_neighbors', 2))
            neigh.fit(X_train, y_train)
            joblib.dump(neigh, '{}/kneighbour.pkl'.format(self.model_folder))
            
            y_pred = neigh.predict(X_test)
            train_statistics(y_test, y_pred, title="KNeighbour with leaf size: {}".format(i))
            plot(y_test, y_pred, self.image_folder, show=False, title="k_neigh_{}".format(i))
            return ads
        return fit

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # adaBOOST
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    def train_adaBoost(self, ads):
        X, y = generate_matrix(ads, 'price')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        # learning rate 0.05 - 0.2
        for i in [18, 19, 21, 22, 23]:
            logging.info("Adaboost with estimator: {}".format(i))
            boost = AdaBoostRegressor(DecisionTreeRegressor(),
                                      n_estimators=i, random_state=RNG)
            boost.fit(X_train, y_train)
            y_pred = boost.predict(X_test)
            train_statistics(y_test, y_pred, title="Adaboost with estimator: {}".format(i))
            plot(y_test, y_pred, self.image_folder, show=False, title="adaboost_{}".format(i))

    def fit_adaBoost(self, params):
        def fit(ads):
            X, y = generate_matrix(ads, 'price')
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
            boost = AdaBoostRegressor(DecisionTreeRegressor(),
                                      n_estimators=params.get('n_estimators', 18),
                                      random_state=RNG)
            boost.fit(X_train, y_train)
            joblib.dump(neigh, '{}/adaboost.pkl'.format(self.model_folder))
            y_pred = boost.predict(X_test)
            train_statistics(y_test, y_pred, title="Adaboost")
            plot(y_test, y_pred, self.image_folder, show=False, title="adaboost")
            return ads
        return fit
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Random Forest
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    def train_random_forest(self, ads):
        X, y = generate_matrix(ads, 'price')
        X, y = X.values, y.values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        parameters = {"n_estimators": [100, 500, 700, 1000], "min_samples_leaf":[1, 5]}
        random = RandomForestRegressor(n_jobs=-1)
        gd = GridSearchCV(random, parameters, verbose=1, scoring=scorer)
        logging.info("Start Fit")
        gd.fit(X_train, y_train)
        logging.info("Best score: {}".format(gd.best_score_))
        logging.info("Best score: {}".format(gd.best_params_))
        logging.info("result: {}".format(gd.cv_results_))
        
        return ads

    def fit_random_forest(self, params):
        def fit(ads):
            X, y = generate_matrix(ads, 'price')
            X, y = X.values, y.values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
            model = RandomForestRegressor(n_estimators=params.get('n_estimators', 700), 
                                          max_features="auto",
                                          n_jobs=-1, min_samples_leaf=1)
            model.fit(X_train, y_train)
            joblib.dump(model, '{}/random_forest.pkl'.format(self.model_folder))
            y_pred = model.predict(X_test)
            train_statistics(y_test, y_pred)
            plot(y_test, y_pred, self.image_folder, show=False, title="random_forest")
            return ads
        return fit


    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # XGBoost
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    def train_xgboost(self, ads):
        X, y = generate_matrix(ads, 'price')
        X, y = X.values, y.values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        parameters = {"max_depth": [10, 100, 500],
                      "learning_rate": [0.01, 0.1, 0.3],
                      "n_estimators": [10, 50, 100, 250, 500]}
        xgb_model = xgb.XGBRegressor(silent=False)
        logging.info("Create GridSearch")
        clf = GridSearchCV(xgb_model, parameters, verbose=1, scoring=scorer)
        logging.info("Start Fit")
        clf.fit(X_train, y_train)
        logging.info("Best score: {}".format(gd.best_score_))
        logging.info("Best score: {}".format(gd.best_params_))
        logging.info("result: {}".format(gd.cv_results_))
        
        return ads

    def fit_xgboost(self, params):
        def fit(ads):
            X, y = generate_matrix(ads, 'price')
            X, y = X.values, y.values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
            xgb_model = xgb.XGBRegressor(silent=False,
                                         max_depth=params.get('max_depth', 100),
                                         learning_rate=params.get('learning_rate', 0.1),
                                         n_estimators=params.get('n_estimators', 350),
                                         n_jobs=-1)

            xgb_model.fit(X_train, y_train)
            xgb_model._Booster.save_model('{}/xgbooster.pkl'.format(self.model_folder))
            y_pred_xgb = xgb_model.predict(X_test)
            train_statistics(y_test, y_pred_xgb, title="Stacked xgb")
            return ads
        return fit

    def predict_xgboost(self, ads):
            xgb_model = xgb.XGBRegressor(silent=False, max_depth=100,
                                        learning_rate=0.1, n_estimators=350, n_jobs=-1)
            booster = xgb.Booster()
            booster.load_model('{}/xgbooster.pkl'.format(self.model_folder))
            xgb_model._Booster = booster

            X, y = generate_matrix(ads, 'price')
            X, y = X.values, y.values
            idx = 0
            name = 'xgboost'
            for train_index, test_index in KFold(n_splits=3, shuffle=True).split(X):
                logging.info('New split')
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                logging.info("Size of training data: {}".format(len(X_train)))
                logging.info("Size of testing data: {}".format(len(X_test)))
                y_pred = xgb_model.predict(X_test)
                train_statistics(y_test, y_pred, title="{}_{}".format(name, idx))
                plot(y_test, y_pred, self.image_folder, show=False, title="{}_{}".format(name, idx))
                idx += 1
            return ads

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Extra Tree
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    def train_extra_tree(self, ads):
        X, y = generate_matrix(ads, 'price')
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
            logging.info("Stats for estimator: {}".format(estimator))
            train_statistics(y_test, y_pred, title="ExtraTree_train_{}".format(estimator))
        
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

    def fit_extra_tree(self, params):
        def fit(ads):
            model = ExtraTreesRegressor(n_estimators=params.get('n_estimators', 700),
                                        n_jobs=-1, random_state=RNG)
            model.fit(X_train, y_train)
            joblib.dump(model, '{}/extra_tree_regressior.pkl'.format(self.model_folder))
            y_pred = model.predict(X_test)
            train_statistics(y_test, y_pred, title="ExtraTree_train")
            plot(y_test, y_pred, self.image_folder, show=True, title="ExtraTree_train")
            return ads
        return fit
