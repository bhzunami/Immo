# Import only
import os
import pandas as pd
import pdb
import math
import time
import json
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
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
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns

import sys

rng = np.random.RandomState(42)

OBJECT_TYPES = {'ogroup': object,
                'otype': object,
                'municipality': object,
                'floor': object,
                'canton_id': object,
                'district_id': object,
                'mountain_region_id': object,
                'language_region_id': object,
                'job_market_region_id': object,
                'agglomeration_id': object,
                'metropole_region_id': object,
                'tourism_region_id': object}

CLASSIFICATION_TYPES = {}

def get_data(from_db=False):
    """ Read data from all.csv
    """
    return pd.read_csv('all.csv', index_col=0, engine='c', dtype=OBJECT_TYPES)


def get_col_name(dv, idx):
    return "{} ({})".format(dv.get_feature_names()[idx], idx)

def get_col_name2(X, ohe, ohe_features, dv, idx):
    """
    depricated, because we only use dv
    """
    try:
        # if it's a OHE feature
        # OHE Featrues are left from index 0 to len(ohe.active_features_)-1
        if idx < len(ohe.active_features_):
            active_feature = ohe.active_features_[idx]
            for i, val in enumerate(ohe.feature_indices_):
                if active_feature < val:
                    return "OneHot: " + ohe_features[i-1][1] + " {} ({})".format(active_feature - ohe.feature_indices_[i-1], idx)
        else: # normal feature
            base1 = idx - len(ohe.active_features_)
            for i, val in enumerate(ohe_features):
                # add +i to jump over one hot encoded feature
                if base1 + i < val[0]:
                    return dv.get_feature_names()[base1 + i] + " ({})".format(idx)
            
            return dv.get_feature_names()[base1 + len(ohe_features) - 1] + " ({})".format(idx)
        
        raise Exception("Col idx not found: {}".format(idx))
    except Exception as e:
        pdb.set_trace()

def print_isolation_forest(data, cls, x, y, outlierIdx, elements):
    fig = plt.figure(figsize=(150, 150))

    for i, feature in enumerate(elements):
        Z = cls[feature].decision_function(np.c_[x[feature].ravel(), y[feature].ravel()])
        #Z = cls.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(x[feature].shape)

        ax = fig.add_subplot(2, 2, i+1)
        ax.set_ylabel('Price')
        ax.set_title('Price - {}'.format(feature))
            
        ax.contourf(x[feature],
                    y[feature], Z, 
                    cmap=plt.cm.Blues_r)

        ax.scatter(data[feature][np.where(outlierIdx[feature] == -1)][:, 0],
                   data[feature][np.where(outlierIdx[feature] == -1)][:, 1],
                   c='red', s=2)

        ax.scatter(data[feature][np.where(outlierIdx[feature] == 1)][:, 0],
                   data[feature][np.where(outlierIdx[feature] == 1)][:, 1],
                   c='green', s=2)

        print(feature)

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig("IsolationForest.png")
    plt.close()



if __name__ == "__main__":
    ads = get_data(from_db=False)
    ads = ads.drop(['id', 'long', 'lat', 'price_brutto', 'living_area'], axis=1)
    
    # Insert missing data
    ads.floor.fillna(0, inplace=True)  # Add floor 0 if not present
    ads.num_rooms.fillna(ads.num_rooms.median(), inplace=True)

    # Remove some vars
    # If no renovation was found the last renovation was the build year
    ads_cleanup = ads.dropna()  # Remove all entries with NaN


    # Show some stats
    # Price per sm2 for cantons
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
    sns.barplot(x="canton_id", y="price_brutto_m2", data=ads_cleanup, ax=ax1)
    ax1.set(xlabel='')
    sns.barplot(x="num_rooms", y="price_brutto_m2", data=ads_cleanup, ax=ax2)
    sns.barplot(x="floor", y="price_brutto_m2", data=ads_cleanup, ax=ax3)    
    g = sns.barplot(x="otype", y="price_brutto_m2", data=ads_cleanup, ax=ax4)
    g.set_xticklabels(g.get_xticklabels(), rotation=90)
    #plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()
    #plt.savefig("Analytical_data.png")
    plt.close()
    
    
    sns.lmplot(x="avg_room_area", y="price_brutto_m2", data=ads_cleanup)
    plt.savefig("avg_room_area.png")
    plt.close()
    
    sns.lmplot(x="build_year", y="price_brutto_m2", data=ads_cleanup)
    plt.savefig("build_year.png")
    plt.close()


    #scatterplot
    sns.set()
    cols = ['price_brutto_m2', 'build_year', 'num_rooms', 'avg_room_area', 'last_construction']
    sns.pairplot(ads[cols], size = 2.5)
    plt.savefig("pairplot.png")
    plt.close()
    

    elements = ['floor', 'build_year', 'num_rooms', 'avg_room_area']
    data = {}
    outlierIdx = {}
    xx = {}
    yy = {}
    cls = {
        'floor': IsolationForest(max_samples=0.7, contamination=0.01, random_state=rng),
        'build_year': IsolationForest(max_samples=0.7, contamination=0.01, random_state=rng),
        'num_rooms': IsolationForest(max_samples=0.7, contamination=0.01, random_state=rng),
        'avg_room_area': IsolationForest(max_samples=0.7, contamination=0.01, random_state=rng)
    }
    meshgrids = {
        'floor': np.meshgrid(np.linspace(-1, 40, 400), np.linspace(0, 60000, 1000)),
        'build_year': np.meshgrid(np.linspace(0, 2025, 400), np.linspace(0, 60000, 1000)),
        'num_rooms': np.meshgrid(np.linspace(-1, 100, 400), np.linspace(0, 60000, 1000)),
        'avg_room_area': np.meshgrid(np.linspace(0, 500, 400), np.linspace(0, 60000, 1000))
    }

    for feature in elements:
        # Make a isolation forest with this feature and the price
        ad = ads_cleanup[[feature, 'price_brutto_m2']]
        ad = ad.values
        cls[feature].fit(ad)
        data[feature] = ad
        outlierIdx[feature] = cls[feature].predict(ad)
        try:
            print("found {} outliers".format(ad[np.where(outlierIdx[feature] == -1)].shape[0]))
            xx[feature], yy[feature] = meshgrids[feature]
        except Exception:
            print("Exception in {}".format(feature))

            
    #print_isolation_forest(data, cls, xx, yy, outlierIdx, elements)

    # Remove all outliers in one pice
    o = []
    for feature in elements:
        o = np.unique(np.concatenate((o, np.where(outlierIdx[feature] == -1)[0]), axis=0))
    ads_with_out_anomaly = ads_cleanup.drop(ads_cleanup.index[o.astype(int)])

    print("Removed {} elements from dataset.".format(ads_cleanup.shape[0]-ads_with_out_anomaly.shape[0]))

    # Reindex our dataset
    ads_with_out_anomaly.reindex()

    
    corr = ads_with_out_anomaly.corr()
    labels = corr.keys()
    g = sns.heatmap(corr, vmax=.8, square=True)
    g.set_yticklabels(labels=labels, rotation=0)
    g.set_xticklabels(labels=reversed(labels), rotation=90)
    plt.savefig("corr.png")
    plt.close()
    
    
    # Get the most corelate features
    k = 15 #number of variables for heatmap
    cols = corr.nlargest(k, 'price_brutto_m2')['price_brutto_m2'].index
    cm = np.corrcoef(ads_with_out_anomaly[cols].values.T)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
    hm.set_xticklabels(hm.get_xticklabels(), rotation=90)
    hm.set_yticklabels(reversed(hm.get_xticklabels()), rotation=0)
    plt.savefig("corrDetail.png")
    plt.close()
    
    
    
    # Prepare transformation from pandas dataframe to matrix
    X = ads_with_out_anomaly.drop(['price_brutto_m2'], axis=1)
    y = ads_with_out_anomaly['price_brutto_m2'].astype(int)
    
    # One Hot Encoding for string
    # Otype municiplaiy ogroup
    # - - - - - - - - - - - - - - - - 
    dv = DictVectorizer(sparse=True)  
    dv.fit(X.T.to_dict().values())  # Learn a list of feature name Important Price is present here
    print("Len of features: {}".format(len(dv.feature_names_)))

    # Transform feature -> value dicts to array or sparse matrix
    X_transformed = dv.transform(X.T.to_dict().values())
    y_transformed = y.values


    # ONE HOT ENCODING
    # - - - - - - - - - - - - - - - - - - - 
    # # get features and map to corresponding index in X_transformed, and sort by index
    # classification_features = ['floor', 'canton_id', 'district_id',
    # 'mountain_region_id', 'language_region_id', 'job_market_region_id',
    # 'agglomeration_id', 'metropole_region_id', 'tourism_region_id']

    # # Sort the positions and the name of our classification_features in a list
    # indexs = sorted([(dv.get_feature_names().index(e), e) for e in classification_features], key=lambda x: x[0])

    # rf_enc = OneHotEncoder(categorical_features=[i[0] for i in indexs])
    # rf_enc.fit(X_transformed.todense())
    # X_transformed_oneHot = rf_enc.transform(X_transformed.todense())

    from sklearn import datasets
    from sklearn import metrics
    from sklearn.ensemble import ExtraTreesClassifier
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y_transformed, test_size=0.5)

    model = ExtraTreesClassifier(n_estimators=10,
                                 random_state=rng,
                                 max_depth=500)
    model.fit(X_train, y_train)
    #joblib.dump(model, '{}.pkl'.format('extraTrees'))

    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_],
             axis=0)
    indices = np.argsort(importances)[::-1]  # most important first

    print("Feature ranking:")
    min_percent = 0.01

    for f in range(X_train.shape[1]):
        if importances[indices[f]] > min_percent:
            print("{}. feature {} ({})".format(f+1,
                                            get_col_name(dv, indices[f]), 
                                            importances[indices[f]]))

    # Plot all features which are over the min_percent importance
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(len(np.where(importances > min_percent)[0])), importances[indices[0:len(np.where(importances > min_percent)[0])]], color="r", yerr=std[indices[0:len(np.where(importances > min_percent)[0])]], align="center")
    plt.xticks(range(len(np.where(importances > min_percent)[0])), indices[0:len(np.where(importances > min_percent)[0])])
    plt.xlim([-1, len(np.where(importances > min_percent)[0])])
    plt.show()


    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    from sklearn.feature_selection import f_regression
    skb_chi2 = SelectKBest(chi2, k=40)
    skb_f_reg = SelectKBest(f_regression, k=40)
    X_new = skb_chi2.fit_transform(X_train, y_train)
    X_new = skb_f_reg.fit_transform(X_train.todense(), y_train)

    #indices = np.argsort(skb.scores_)[::-1]
    print("Most 40 important features in chi2")
    for feature in skb_chi2.get_support(True):
        print("Feature: {} {}".format(dv.get_feature_names()[feature], skb_chi2.scores_[feature]))

    print()
    print("Most 40 important features in f_regression")
    for feature in skb_f_reg.get_support(True):
        print("Feature: {} {}".format(dv.get_feature_names()[feature], skb_f_reg.scores_[feature]))

    from sklearn.svm import LinearSVC
    from sklearn.feature_selection import SelectFromModel
    # the smaller C the fewer features selected
    #lsvc = LinearSVC(C=0.02, penalty="l1", dual=False).fit(X_train, y_train)
    #model = SelectFromModel(lsvc, prefit=True)
    model = joblib.load('{}.pkl'.format('linearSVC'))
    print()
    print("Most {} important features in L1".format(len(model.get_support(True))))
    for feature in model.get_support(True):
        print("Feature: {} ".format(dv.get_feature_names()[feature]))





    pdb.set_trace()