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

class Pipeline():

    def __init__(self, goal, settings, directory, ):
        self.goal = goal
        self.settings = settings
        self.directory = directory        
        self.image_folder = os.path.abspath(os.path.join(directory, settings['image_folder']))
        self.model_folder = os.path.abspath(os.path.join(directory, settings['model_folder']))

    def simple_stats(sefl, title):
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

    def transform_noise_level(self, ads):
        """ If we have no nose_level at the address
        we use the municipality noise_level
        """
        def lambdarow(row):
            if np.isnan(row.noise_level):
                return row.m_noise_level
            return row.noise_level

        ads['noise_level'] = ads.apply(lambdarow, axis=1)
        return ads

    def replace_zeros_with_nan(self, ads):
        """ replace 0 values into np.nan for statistic
        """
        ads.loc[ads.living_area == 0, 'living_area'] = np.nan
        ads.loc[ads.living_area == 0, 'num_rooms'] = np.nan
        return ads

    def transform_misc_living_area(self, ads):
        ads['price_brutto_m2'] = ads['price_brutto'] / ads['living_area']
        ads['avg_room_area'] = ads['living_area'] / ads['num_rooms']
        return ads

    def transform_build_year(self, ads):
        return ads.drop(ads.index[np.where(ads['build_year'] > 2030)[0]])

    def transform_build_renovation(self, ads):
        ads['was_renovated'] = ads.apply(lambda row: not np.isnan(row['last_renovation_year']), axis=1)

        def last_const(row):
            if row['build_year'] >= 2017 or row['last_renovation_year'] >= 2017:
                return 0
            elif np.isnan(row['last_renovation_year']):
                return 2017 - row['build_year']
            else:
                return 2017 - row['last_renovation_year']

        ads['last_construction'] = ads.apply(last_const, axis=1)

        return ads.drop(['last_renovation_year'], axis=1)

    def transform_floor(self, ads):
        def lambdarow(row):
            if not np.isnan(row['floor']):
                return row['floor']
            if row['ogroup'] == 'haus':
                return 0

            return float('NaN')

        ads['floor'] = ads.apply(lambdarow, axis=1)

        return ads

    def transform_num_rooms(self, ads):
        return ads.dropna(subset=['num_rooms'])

    def drop_floor(self, ads):
        return ads.drop(['floor'], axis=1)

    def transform_living_area(self, ads):
        return ads.dropna(subset=['living_area'])

    def transfrom_description(self, ads):
        return ads.dropna(subset=['description'])

    def transform_onehot(self, ads):
        ads = pd.get_dummies(ads, columns=['canton_id', 'ogroup',
                                           'otype', 'tourism_region_id', 'district_id',
                                           'mountain_region_id', 'job_market_region_id',
                                           'agglomeration_id', 'metropole_region_id',
                                           'municipality'])
        return ads

    def transform_tags(self, ads):
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
        try:
            model = joblib.load('{}/living_area.pkl'.format(self.model_folder))
        except FileNotFoundError:
            logging.error("Could not load living area model. Did you forget to train living area?")
            return ads.dropna(subset=['living_area'])

        tempdf = ads.drop(['price_brutto', 'description', 'characteristics'], axis=1)

        ads.living_area_predicted = 0
        nan_idxs = tempdf.living_area.index[tempdf.living_area.apply(np.isnan)]
        if len(nan_idxs) > 0:
            ads.loc[nan_idxs, 'living_area'] = model.predict(tempdf.drop(['living_area'], axis=1).ix[nan_idxs])
            ads.loc[nan_idxs, 'living_area_predicted'] = 1

        return ads

    def outlier_detection(self, ads):
        meshgrid = {
            'build_year': np.meshgrid(np.linspace(0, max(ads['build_year']), 400), 
                                      np.linspace(0, max(ads['price_brutto']), 1000)),
            'num_rooms': np.meshgrid(np.linspace(-1, max(ads['num_rooms']), 400),
                                     np.linspace(0, max(ads['price_brutto']), 1000)),
            'living_area': np.meshgrid(np.linspace(0, max(ads['living_area']), 400),
                                      np.linspace(0, max(ads['price_brutto']), 1000)),
            'last_construction': np.meshgrid(np.linspace(0, max(ads['last_construction']), 400),
                                             np.linspace(0, max(ads['price_brutto']), 1000)),
            'noise_level': np.meshgrid(np.linspace(0, max(ads['noise_level']), 400),
                                       np.linspace(0, max(ads['price_brutto']), 1000))
        }
        anomaly_detection = AnomalyDetection(ads, self.image_folder, self.model_folder)
        return anomaly_detection.isolation_forest(self.settings['anomaly_detection'],
                                                  meshgrid, self.goal)


    def transform_desc(self, ads):

        def stemm_words(row):
            language = detect_language(row.description)
            #letters_only = re.sub("[^a-z0-9üäöèéàêâ]", " ", str(row.desc.lower()))
            letters_only = row.description.split()
            stops = set(stopwords.words(language))
            meaningful_words = [w for w in letters_only if not w in stops]
            stemmer = SnowballStemmer(language)
            stemmed_words = [stemmer.stem(w) for w in meaningful_words]
            return " ".join(stemmed_words)

        ads['desc'] = ads.apply(stemm_words, axis=1)
        return ads

    def transform_features(self, ads):
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

        # Drop the concatinated features
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


