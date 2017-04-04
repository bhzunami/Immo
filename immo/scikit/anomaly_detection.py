import os
import pandas as pd
import pdb
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
    engine = create_engine(os.environ.get('DATABASE_URL', None))
    Session = sessionmaker(bind=engine)
    session = Session()
    ads = pd.read_sql_query(session.query(AD_View).statement, session.bind)
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


def get_outliers(clf, X, y, name):
    clf.fit(X, y)
    trainingErrors = abs(clf.predict(X) - y)
    outlierIdx = trainingErrors >= np.percentile(trainingErrors, 80)
    print("{} found {} outliers".format(name, outlierIdx[np.where(outlierIdx == True)].shape[0]))
    return (outlierIdx, trainingErrors)

def plot(X, y, outlierIdx):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X.living_area[2000:], y[2000:], X.num_rooms[2000:], c=(0,0,1))
    ax.scatter(X.living_area[outlierIdx], y[outlierIdx], X.num_rooms[outlierIdx],c=(1,0,0))
    plt.show()


def main():
    ads = get_data(from_db=False)
    ads = ads.drop(['id'], axis=1)
    # Remove some vars
    # If no renovation was found the last renovation was the build year
    ads.last_renovation_year = ads.last_renovation_year.fillna(ads.build_year)
    ads_cleanup = ads.dropna()
    print(ads_cleanup.shape)

    dv = DictVectorizer(sparse=True)
    dv.fit(ads_cleanup.T.to_dict().values())  # Learn a list of feature name Important Price is present here
    print("Len of features: {}".format(len(dv.feature_names_)))
    X = ads_cleanup.drop(['price_brutto'], axis=1)
    y = ads_cleanup['price_brutto']

    train_X = dv.transform(X.T.to_dict().values())  # Transform feature -> value dicts to array or sparse matrix
    train_y = y.values


    classifiers = {
        'Linear regression': LinearRegression(),
        'Isolation forest': IsolationForest(max_samples=10000, random_state=rng),
        'KNeighbors': KNeighborsClassifier(n_neighbors=20)
    }

    for name, clf in classifiers.items():
        get_outliers(clf, train_X, train_y, name)

if __name__ == "__main__":
    main()
