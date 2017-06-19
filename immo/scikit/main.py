#!/usr/bin/env python
"""

"""
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
from sklearn import metrics
from sklearn.externals import joblib
# NLTK
import nltk
from nltk.corpus import stopwords # Import the stop word list
from nltk.stem import SnowballStemmer

from a_detection import AnomalyDetection


#from settings import OBJECT_TYPES, MODEL_FOLDER, IMAGE_FOLDER
settings = json.load(open('settings.json'))

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s: %(levelname)s - %(message)s')

parser = argparse.ArgumentParser(description=__doc__)
# Some default config arguments
parser.add_argument('-t', '--train',
                    help='Train algorithmes new (Do not use saved models)',
                    action="store_true")
parser.add_argument('-r', '--run',
                    help='Do not print or store only calc models with the best attributes',
                    action="store_true")
parser.add_argument('-p', '--predict',
                    help='Predict the house price',
                    action="store_true")
parser.add_argument('-g', '--goal',
                    help='What is our target',
                    default="price_brutto")
parser.add_argument('-f', '--file',
                    help='Predict the house price',
                    default='all_2.csv')
                    
args = parser.parse_args()
# Define a global FILE for pipeline
FILE = args.file
GOAL = args.goal

RUN_PIPLINE = []
RNG = np.random.RandomState(42)


def generate_matrix(df, goal):
    X = df.drop([goal], axis=1)
    y = df[goal].astype(int)
    return X, y

def ape(y_true, y_pred):
    return np.abs(y_true - y_pred) / y_true

def mape(y_true, y_pred):
    return ape(y_true, y_pred).mean()

def mdape(y_true, y_pred):
    return np.median(ape(y_true, y_pred))

def gen_subplots(fig, x, y):
    for i in range(x*y):
        ax = fig.add_subplot(x, y, i+1)
        plt.grid()
        yield ax

def plot(y_test, y_pred, show=False, plot_name=""):
    # sort both arrays by y_test
    y_test, y_pred = zip(*sorted(zip(y_test, y_pred)))
    y_test = np.asarray(y_test)
    y_pred = np.asarray(y_pred)
    markersize = 1
    fig = plt.figure(figsize=(10, 10))
    subplots = gen_subplots(fig, 3, 1)

    ax = next(subplots)
    ax.set_xlabel('Actual vs. Predicted values')
    ax.plot(y_test, 'bo', markersize=markersize, label="Actual")
    ax.plot(y_pred, 'ro', markersize=markersize, label="Predicted")

    ax = next(subplots)
    ax.set_xlabel('Predicted value')
    ax.set_ylabel('Residuals')
    ax.plot(y_pred, y_test - y_pred, 'bo', markersize=markersize)
    ax.plot((np.min(y_pred), np.max(y_pred)), (0, 0), 'r-')

    plt.tight_layout()
    plt.savefig("{}/cumulative_prpability_{}.png".format(settings['image_folder'], plot_name))
    if show:
        plt.show()

    fig = plt.figure(figsize=(5, 5))
    subplots = gen_subplots(fig, 1, 1)
    ax = next(subplots)
    ax.set_xlabel('APE')
    my_ape = ape(y_test, y_pred)
    ax.hist(my_ape, 100)
    mape = ape(y_test, y_pred).mean()
    ax.plot((mape, mape), (0, 400), 'r-')

    mdape = np.median(ape(y_test, y_pred))
    ax.plot((mdape, mdape), (0, 400), 'y-')

    ax.set_xticklabels(['{:.2%}'.format(x) for x in ax.get_xticks()])
    plt.tight_layout()
    plt.savefig("{}/verteilung_der_fehler_{}.png".format(settings['image_folder'], plot_name))
    if show:
        plt.show()

def train_statistics(y, pred, title):
    logging.info("Statistics for: {}".format(title))
    diff = np.fabs(y - pred)
    logging.info("             R²-Score: {:10n}".format(metrics.r2_score(y, pred)))
    logging.info("                 MAPE: {:.3%}".format(mape(y, pred)))
    logging.info("                MdAPE: {:.3%}".format(mdape(y, pred)))
    logging.info("            Min error: {:10n}".format(np.amin(diff)))
    logging.info("            Max error: {:10n}".format(np.amax(diff)))
    logging.info("  Mean absolute error: {:10n}".format(metrics.mean_absolute_error(y, pred)))
    logging.info("Median absolute error: {:10n}".format(metrics.median_absolute_error(y, pred)))
    logging.info("   Mean squared error: {:10n}".format(metrics.mean_squared_error(y, pred)))

    num_elements = len(pred)
    apes = ape(y, pred)
    for i in np.arange(5, 100, 5):
        logging.info("I {}: {}".format(i, len(np.where(apes < i/100)[0])/num_elements))


def read_data(file):
    def read(df):
        return pd.read_csv(file, index_col=0, engine='c')

    return read

def simple_stats(title):

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

def transform_noise_level(ads):
    """ If we have no nose_level at the address
    we use the municipality noise_level
    """
    def lambdarow(row):
        if np.isnan(row.noise_level):
            return row.m_noise_level
        return row.noise_level

    ads['noise_level'] = ads.apply(lambdarow, axis=1)
    return ads

def replace_zeros_with_nan(ads):
    """ replace 0 values into np.nan for statistic
    """
    ads.loc[ads.living_area == 0, 'living_area'] = np.nan
    ads.loc[ads.living_area == 0, 'num_rooms'] = np.nan
    return ads

def transform_misc_living_area(ads):
    ads['price_brutto_m2'] = ads['price_brutto'] / ads['living_area']
    ads['avg_room_area'] = ads['living_area'] / ads['num_rooms']
    return ads

def transform_build_year(ads):
    return ads.drop(ads.index[np.where(ads['build_year'] > 2030)[0]])

def transform_build_renovation(ads):
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

def transform_floor(ads):
    def lambdarow(row):
        if not np.isnan(row['floor']):
            return row['floor']
        if row['ogroup'] == 'haus':
            return 0

        return float('NaN')

    ads['floor'] = ads.apply(lambdarow, axis=1)

    return ads

def transform_num_rooms(ads):
    return ads.dropna(subset=['num_rooms'])

def drop_floor(ads):
    return ads.drop(['floor'], axis=1)

def transform_living_area(ads):
    return ads.dropna(subset=['living_area'])

def transfrom_description(ads):
    return ads.dropna(subset=['description'])

def transform_onehot(df):
    df = pd.get_dummies(df, columns=['canton_id', 'ogroup',
                                     'otype', 'tourism_region_id', 'district_id',
                                     'mountain_region_id', 'job_market_region_id',
                                     'agglomeration_id', 'metropole_region_id',
                                     'municipality'])
    return df

def transform_tags(df):
    with open('../crawler/taglist.txt') as f:
        search_words = set(["tags_" + x.split(':')[0] for x in f.read().splitlines()])

    template_dict = dict.fromkeys(search_words, 0)

    def transformer(row):
        the_dict = template_dict.copy()

        for tag in ast.literal_eval(row.tags):
            the_dict["tags_" + tag] = 1

        return pd.Series(the_dict)

    tag_columns = df.apply(transformer, axis=1)
    return df.drop(['tags'], axis=1).merge(tag_columns, left_index=True, right_index=True)

def train_living_area(ads):
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
                           settings['living_area_prediction']['limit'],
                           settings['living_area_prediction']['step']):
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
            joblib.dump(model, '{}/living_area.pkl'.format(settings['model_folder']))

    settings['living_area_prediction']['estimator'] = best_estimator
    # Save best c for all features
    with open('settings.json', 'w') as f:
        f.write(json.dumps(settings))
    return ads

def predict_living_area(ads):    
    try:
        model = joblib.load('{}/living_area.pkl'.format(settings['model_folder']))
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

def train_outlier_detection(ads):
    """Check which contamination is the best for the features
       Run isolation forest with different contamination and check
       difference in the standard derivation.
       If the diff is < 1 we found our c
    """
    for feature in settings['anomaly_detection']['features']:
        logging.debug("Check feature: {}".format(feature))
        # Only use the this specific feature with our target
        tmp_ad = ads[[feature, GOAL]]
        # Initialize std and std_percent
        std = [np.std(tmp_ad[feature].astype(int))]
        std_percent = [100]
        difference = [0]
        # We are in train phase so best_c should always be 0
        best_c, settings['anomaly_detection'][feature] = 0, 0
        last_model, cls_ = None, None

        # Run isolation forest for diffrent contamination
        for c in np.arange(0.01, settings['anomaly_detection']['limit'],
                           settings['anomaly_detection']['step']):
            last_model, cls_ = cls_ , IsolationForest(max_samples=0.7,
                                                      contamination=c,
                                                      n_estimators=settings['anomaly_detection']['estimator'],
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
                logging.info("Found best C: {} for feature: {}".format(c, feature))
                # We do not need this c, we need the last c - step
                best_c = np.around(c - settings['anomaly_detection']['step'], 2)
                joblib.dump(last_model, '{}/isolation_forest_{}.pkl'.format(settings['model_folder'], feature))
            difference.append(diff)

        # Store best c in our settings to use it later
        settings['anomaly_detection'][feature] = best_c if best_c > 0 else 0 + settings['anomaly_detection']['step']
        # Plot stuff
        logging.info("Best C for feature {}: {}".format(feature, best_c))
        # Plot the standard derivation for this feature
        fig, ax1 = plt.subplots()
        ax1.set_title('{}'.format(feature))
        ax1.plot(list(np.arange(0.0, settings['anomaly_detection']['limit'],
                                settings['anomaly_detection']['step'])),
                 std_percent, c='r')
        ax1.set_ylabel('Reduction of STD deviation in %')
        ax1.set_xlabel('% removed of {}'.format(feature))
        plt.savefig('{}/{}_std.png'.format(settings['image_folder'], feature))
        plt.close()

        # Plot the difference between the standard derivation
        plt.plot(list(np.arange(0.0, settings['anomaly_detection']['limit'],
                                settings['anomaly_detection']['step'])), difference, c='r')
        plt.savefig('{}/{}_diff_of_std.png'.format(settings['image_folder'], feature))
        plt.close()

    # Save best c for all features
    with open('settings.json', 'w') as f:
        f.write(json.dumps(settings))
    return ads


def outlier_detection(ads):

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
    anomaly_detection = AnomalyDetection(ads, settings['image_folder'], settings['model_folder'])
    return anomaly_detection.isolation_forest(settings['anomaly_detection'], meshgrid, GOAL)


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

def transform_desc(ads):

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


def transform_features(ads):
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

def extraTreeRegression(ads):
    filterd_ads = ads.drop(['characteristics', 'description'], axis=1)
    logging.debug("Find best estimator for ExtraTreesRegressor")
    X, y = generate_matrix(filterd_ads, 'price_brutto')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    model = ExtraTreesRegressor(n_estimators=700)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    train_statistics(y_test, y_pred)
    plot(y_test, y_pred, show=True, plot_name="ExtraTree")
    return ads

def main(args):
    pipeline = RUN_PIPLINE

    if args.train:
       pipeline = TRAIN_PIPLINE

    logging.info("Start with args {}".format(args))
    ads = pd.read_csv('all_2.csv', index_col=0, engine='c')
    for f in pipeline:
        logging.info("Apply transformation: {}".format(f.__name__))
        ads = f(ads)

TRAIN_PIPLINE = [
    read_data(FILE),
    simple_stats('Before Transformation'),
    transform_build_year,
    transform_build_renovation,
    replace_zeros_with_nan,
    transform_noise_level,
    transform_floor,
    transform_num_rooms,
    transfrom_description,
    transform_living_area,
    drop_floor,
    simple_stats('After Transformation'),
    transform_tags,
    transform_onehot,
    transform_features,
    #train_outlier_detection,
    outlier_detection,
    #train_living_area,
    predict_living_area,
    #transform_desc,
    #transform_misc_living_area,
    extraTreeRegression
]

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # create a file handler
    handler = logging.FileHandler('ml.log')
    handler.setLevel(logging.INFO)
    # create a logging format
    formatter = logging.Formatter('%(asctime)s: %(levelname)s - %(message)s', "%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)

    main(args)
