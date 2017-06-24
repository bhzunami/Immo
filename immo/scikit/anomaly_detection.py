import os
import pandas as pd
import pdb
import math
import time
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import AD_View
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn import metrics
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
import gaussian

rng = np.random.RandomState(42)


DB_TYPES = {'canton_id': object,
            'ogroup': object,
            'district_id': object,
            'mountain_region_id': object,
            'language_region_id': object,
            'job_market_region_id': object,
            'agglomeration_id': object,
            'metropole_region_id': object,
            'tourism_region_id': object,
            'lat': object,
            'long': object}

def get_data(from_db=False):
    if from_db:
        return get_from_db()
    return pd.read_csv('all.csv', index_col=0, engine='c', dtype=DB_TYPES)

def get_from_db():
    engine = create_engine(os.environ.get('DATABASE_URL'))
    Session = sessionmaker(bind=engine)
    session = Session()
    ads = pd.read_sql_query(session.query(AD_View).statement, session.bind)

    ads = transform_tags(ads)

    ads.to_csv('all.csv', header=True, encoding='utf-8')
    return ads


def train_and_evaluate(clf, X_train, y_train, name):
    print("Start Fit")
    clf.fit(X_train, y_train)
    print("END Fit")
    joblib.dump(clf, '{}.pkl'.format(name))
    print("Coefficient of determination on training set: {}".format(clf.score(X_train, y_train)))
    # create a k-fold cross validation iterator of k=5 folds
    #cv = KFold(X_train.shape[0], 5, shuffle=True, random_state=33)
    #scores = cross_val_score(clf, X_train, y_train, cv=cv)
    #print("Average coefficient of determination using 5-fold crossvalidation: {}".format(np.mean(scores)))

def load_or_train_clf(clf, X, y, name):
    if not os.path.exists('{}.pkl'.format(name)):
        clf.fit(X, y)
        joblib.dump(clf, '{}.pkl'.format(name))
    else:
        print("load {}.pkl".format(name))
        clf = joblib.load('{}.pkl'.format(name))

    return clf

def get_outliers_gauss(clf, X, y, name):
    trainingErrors = abs(clf.predict(X) - y)
    outlierIdx = trainingErrors >= np.percentile(trainingErrors, 95)
    print("{} found {} outliers".format(name, outlierIdx[np.where(outlierIdx == True)].shape[0]))
    return (outlierIdx, trainingErrors)

def isolation_forest(clf, X, name):
    outlierIdx = clf.predict(X)
    print("{} found {} outliers".format(name, X[np.where(outlierIdx == -1)].shape[0]))
    return outlierIdx


def plot_outliers(X, y, keys, outlieridx, name):
    fig = plt.figure(figsize=(20, 140))  # Breite, Höhe
    for i, key in enumerate(keys):
        ax = fig.add_subplot(36, 2, i+1)
        try:
            ax.scatter(pd.to_numeric(X[np.where((outlieridx == False) | 
                                                (outlieridx == 1))][:5000, i]), 
                       y[:5000,],
                       c=(0,0,1),
                       s=1)
            ax.scatter(pd.to_numeric(X[np.where((outlieridx == True) | (outlieridx == -1))][:, i]), y[:len(X[np.where((outlieridx == True) | (outlieridx == -1))]),], c=(1,0,0), s=1)
        except Exception as e:
            pdb.set_trace()
        ax.set_ylabel('Price')
        ax.set_xlabel(key)
        ax.set_title('Price - {}'.format(key))

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig('outliers_{}.png'.format(name))


def plot_all(X, y, keys):
    fig = plt.figure(figsize=(8, 20))
    for i, key in enumerate(keys):
        ax = fig.add_subplot(11, 2, i+1)
        ax.scatter(pd.to_numeric(X[:2000, i]), y[:2000,], c=(0,0,1))
        ax.set_ylabel('Price')
        ax.set_xlabel(key)
        ax.set_title('Price - {}'.format(key))

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig('plt.png')


def plot(X, y, outlierIdx):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X.living_area[2000:], y[2000:], X.num_rooms[2000:], c=(0,0,1))
    ax.scatter(X.living_area[outlierIdx], y[outlierIdx], X.num_rooms[outlierIdx],c=(1,0,0))


def transform_tags(dataframe):
    with open('../crawler/taglist.txt') as f:
        search_words = set([x.split(':')[0] for x in f.read().splitlines()])

    template_dict = dict.fromkeys(search_words, 0)

    def transformer(row):
        the_dict = template_dict.copy()

        for tag in row.tags:
            the_dict[tag] = 1

        return pd.Series(the_dict)

    tag_columns = dataframe.apply(transformer, axis=1)

    return dataframe.drop(['tags'], axis=1).merge(tag_columns, left_index=True, right_index=True)


def main():
    ads = get_data(from_db=True)
    ads = ads.drop(['id'], axis=1)
    import sys
    sys.exit(0)
    # Remove some vars
    # If no renovation was found the last renovation was the build year
    ads.last_renovation_year = ads.last_renovation_year.fillna(ads.build_year)
    ads_cleanup = ads.dropna()
    print(ads_cleanup.shape)

    dv = DictVectorizer(sparse=True)
    dv.fit(ads_cleanup.T.to_dict().values())  # Learn a list of feature name Important Price is present here
    print("Len of features: {}".format(len(dv.feature_names_)))
    X = ads_cleanup.drop(['price_brutto', 'long', 'lat'], axis=1)
    y = ads_cleanup['price_brutto']

    train_X = dv.transform(X.T.to_dict().values())  # Transform feature -> value dicts to array or sparse matrix
    train_y = y.values

    plot_data = X.drop(['otype', 'municipality', 'ogroup'], axis=1)
    #plot_all(plot_data.values, y, plot_data.keys())

    classifiers = {
        'Linear regression': LinearRegression(),
        'Isolation forest': IsolationForest(max_samples=100, contamination=0.01, random_state=rng),
        #'KNeighbors': KNeighborsClassifier(n_neighbors=20)
    }

    for name, clf in classifiers.items():
        classifier = load_or_train_clf(clf, train_X, train_y, name)
        if name == 'Isolation forest':
            outIdx = isolation_forest(classifier, train_X, name)
        else:
            outIdx, Error = get_outliers_gauss(classifier, train_X, train_y, name)
        plot_outliers(plot_data.values, y, plot_data.keys(), outIdx, name)

    

if __name__ == "__main__":
    main()
