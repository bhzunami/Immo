#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
import os
import pdb

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
import xgboost as xgb

import logging
import json
import argparse
import numpy as np
import pandas as pd
import sys
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
from collections import defaultdict
import os
from osgeo import gdal

from . import helper
import json
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, load_only, Load
from models import Advertisement, Municipality, ObjectType
from scikit.combined_ensemble import CombinedEnsemble
from scikit import combined_ensemble as combined_ensemble
import sys
import requests
import re
import json

from .pipeline import Pipeline
from models import utils

DIRECTORY = os.path.dirname(os.path.abspath(__file__))
settings = json.load(open('{}/settings.json'.format(DIRECTORY)))

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s: %(levelname)s - %(message)s', filename='train.log')
logging.getLogger().addHandler(logging.StreamHandler())


OPENSTREETMAP_BASE_URL = 'http://nominatim.openstreetmap.org/search/'
GOOGLE_MAP_API_BASE_URL = 'https://maps.googleapis.com/maps/api/geocode/json?address='
ADMIN_GEO = 'http://geodesy.geo.admin.ch/reframe/wgs84tolv03'

engine = create_engine(os.environ.get('DATABASE_URL', None))
Session = sessionmaker(bind=engine)
session = Session()

def load_additional(municipality_id, otype_id):
    m_stmt = session.query(Municipality).filter_by(id=municipality_id).options(
        Load(Municipality).load_only(
            "name",
            "canton_id",
            "district_id",
            "mountain_region_id",
            "language_region_id",
            "job_market_region_id",
            "agglomeration_id",
            "metropole_region_id",
            "tourism_region_id",
            "is_town",
            "noise_level",
            "urban_character_id",
            "steuerfuss_gde",
            "steuerfuss_kanton",
            "degurba_id",
            "planning_region_id",
            "ase",
            "greater_region_id",
            "ms_region_id",
            "municipal_size_class_id",
            "agglomeration_size_class_id",
            "municipal_type22_id",
            "municipal_type9_id")
    ).with_labels().statement

    o_stmt = session.query(ObjectType).filter_by(id=otype_id).options(
        Load(ObjectType).load_only("name", "grouping")
    ).with_labels().statement

    return pd.read_sql_query(m_stmt, session.bind) \
        .join(pd.read_sql_query(o_stmt, session.bind)) \
        .drop(['municipalities_id', 'object_types_id'], axis=1) \
        .rename(columns={'municipalities_name': 'municipality',
                         'municipalities_canton_id': 'canton_id',
                         'municipalities_district_id': 'district_id',
                         'municipalities_planning_region_id': 'planning_region_id',
                         'municipalities_mountain_region_id': 'mountain_region_id',
                         'municipalities_ase': 'ase',
                         'municipalities_greater_region_id': 'greater_region_id',
                         'municipalities_language_region_id': 'language_region_id',
                         'municipalities_ms_region_id': 'ms_region_id',
                         'municipalities_job_market_region_id': 'job_market_region_id',
                         'municipalities_agglomeration_id': 'agglomeration_id',
                         'municipalities_metropole_region_id': 'metropole_region_id',
                         'municipalities_tourism_region_id': 'tourism_region_id',
                         'municipalities_municipal_size_class_id': 'municipal_size_class_id',
                         'municipalities_urban_character_id': 'urban_character_id',
                         'municipalities_agglomeration_size_class_id': 'agglomeration_size_class_id',
                         'municipalities_is_town': 'is_town',
                         'municipalities_degurba_id': 'degurba_id',
                         'municipalities_municipal_type22_id': 'municipal_type22_id',
                         'municipalities_municipal_type9_id': 'municipal_type9_id',
                         'municipalities_noise_level': 'm_noise_level',
                         'municipalities_steuerfuss_gde': 'steuerfuss_gde',
                         'municipalities_steuerfuss_kanton': 'steuerfuss_kanton',
                         'object_types_name': 'otype',
                         'object_types_grouping': 'ogroup'})

def get_long_lat_from_google(street, zipname):
    emptyanswer = (None, None)
    if not os.environ.get('GOOGLE_MAP_API_KEY', None):
        print("Missing Google map api key")
        return emptyanswer

    address = "{}, {}".format(street, zipname)

    url = "{}{}&key={}".format(GOOGLE_MAP_API_BASE_URL, address, os.environ.get('GOOGLE_MAP_API_KEY'))

    try:
        response = requests.get(url)
        if response.status_code != 200:
            return emptyanswer
        res = response.json()
        if res['status'] == "OVER_QUERY_LIMIT" or res['status'] == "ZERO_RESULTS":
            return emptyanswer

        if len(res['results']) > 0:
            long = res['results'][0]['geometry']['location']['lng']
            lat = res['results'][0]['geometry']['location']['lat']
            return long, lat

        print(res['status'])
        return emptyanswer

    except Exception:
        return emptyanswer

def get_lv03(lng, lat):
    emptyanswer = (None, None)

    url = "{}?easting={}&northing={}&format=json".format(ADMIN_GEO, lng, lat)
    try:
        response = requests.get(url)
        if response.status_code != 200:
            print("Could not get X and Y for address {}, {}".format(lng, lat))
            return emptyanswer

        answer = response.json()
        if len(answer) > 0:
            return float(answer['easting']), float(answer['northing'])

        print("Could not get X and Y for address {}, {}".format(lng, lat))

        return emptyanswer
    except Exception:
        print("Could not get X and Y for address {}, {} (exception)".format(lng, lat))
        return emptyanswer


class GeoData():
    def __init__(self, filename):
        driver = gdal.GetDriverByName('GTiff')
        dataset = gdal.Open(filename)
        band = dataset.GetRasterBand(1)

        self.ndval = band.GetNoDataValue()

        transform = dataset.GetGeoTransform()
        self.xOrigin = transform[0]
        self.yOrigin = transform[3]
        self.pixelWidth = transform[1]
        self.pixelHeight = -transform[5]

        cols = dataset.RasterXSize
        rows = dataset.RasterYSize
        self.geodata = band.ReadAsArray(0, 0, cols, rows)

    # lv03 coordinates
    def mean_by_coord(self, easting, northing):
        num_pix = 15

        all_values = []
        for x in range(-num_pix, num_pix+1):
            for y in range(-num_pix, num_pix+1):
                col = int((easting - self.xOrigin) / self.pixelWidth) + x
                row = int((self.yOrigin - northing) / self.pixelHeight) + y
                all_values.append(self.geodata[row][col])
        all_values = np.array(all_values)
        return np.mean(all_values[np.where(all_values != self.ndval)])

def get_noise_level(easting, northing):
    print("Loading tiff file...")
    # this file is not checked in, get it here (GeoTiff): https://opendata.swiss/dataset/larmbelastung-durch-strassenverkehr-tag
    geodata = GeoData("crawler/crawler/scripts/Strassenlaerm_Tag.tif")

    return geodata.mean_by_coord(easting, northing)

def get_municipality_id(zip_municipality):
    # Get zip
    zip_code, *name = utils.get_place(zip_municipality)
    name = utils.extract_municipality(' '.join(name))
    # Search in database
    municipalities = session.query(Municipality).filter(Municipality.zip == utils.extract_number(zip_code)).all()

    for m in municipalities:
        if m.name.startswith(name) or name in m.alternate_names:
            return m.id

    raise Exception("Municipality '{}' not found in database!".format(zip_municipality))

def user_input_to_df(parameters, settings):

    # search coordinates
    lng, lat = get_long_lat_from_google(parameters['street'], parameters['zip_municipality'])
    easting, northing = get_lv03(lng, lat)
    # serch noise_level
    noise_level = get_noise_level(easting, northing)

    last_renovation_year = parameters.get('last_renovation_year')
    if last_renovation_year is None:
        last_renovation_year = np.nan

    df = pd.DataFrame([{
        'living_area': parameters['living_area'],
        'num_rooms': parameters['num_rooms'],
        'build_year': parameters['build_year'],
        'last_renovation_year': last_renovation_year,
        'tags': get_tags_from_text(parameters['description']),
        'noise_level': noise_level,
        'price': parameters['price'],
    }]).join(load_additional(get_municipality_id(parameters['zip_municipality']), parameters['otype_id']))

    return df


with open('crawler/taglist.txt') as f:
    search_words = set([x.split(':')[0] for x in f.read().splitlines()])
remove_tokens = r'[-().,+\':/}{\n\r!?"•;*\[\]%“„ ˋ\t_◦—=«»~><’‘&@…|−]'

def get_tags_from_text(text):
    clean_words = set(re.split(' ', re.sub(remove_tokens, ' ', text.lower())))
    return json.dumps([w for w in clean_words if w in search_words])

def main(params):
    model = joblib.load(params["model_pkl"])
    ads = joblib.load(params["ads_pkl"])

    # if len(list(ads)) - 1 != len(model.feature_importances_):
    #     logging.error("Number of features from given ads dataframe({}) does not match model({})".format(
    #         len(list(ads)), len(model.feature_importances_)))

    tp = Pipeline("price", settings, DIRECTORY)
    data = user_input_to_df(params, settings)

    pipeline = [
        tp.transform_build_renovation,
        tp.transform_noise_level,
        tp.transform_tags,
        tp.transform_features,
        tp.transform_onehot,
        tp.transform_misc_living_area,
    ]

    for p in pipeline:
        data = p(data)

    if len(data) == 0:
        raise Exception("Input data fell through pipeline.")

    if params["detect_outlier"]:
        data = tp.outlier_detection(data)

        if len(data) == 0:
            raise Exception("Input data is an outlier")

    col_new = set(list(data))
    col_exist = set(list(ads))

    if len(col_new - col_exist) > 0:
        raise Exception("Error: there are input columns which are not in the trained model: {}".format(col_new - col_exist))

    data = data.join(pd.DataFrame(columns=list(col_exist - col_new)))
    data[list(col_exist - col_new)] = 0

    data = data[list(ads)]  # ensure correct order for columns
    data = data.drop(['price'], axis=1)

    y_pred = model.predict(data.values)

    y_test = [params['price']]

    logging.info("          Input Price: {}".format(y_test[0]))
    logging.info("           Prediction: {}".format(y_pred[0]))
    logging.info("                  APE: {:.3%}".format(helper.mape(y_test, y_pred)))
    logging.info("           Difference: {:10n}".format((np.fabs(y_test - y_pred))[0]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Usage example: python -m scikit.main_predict -f data.json")
    parser.add_argument('-f', '--file')
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    main(json.load(open(args.file)))
