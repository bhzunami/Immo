"""
    Data Analysis

    Load data from database or a csv File

    Feature Selection: (http://machinelearningmastery.com/feature-selection-machine-learning-python/)
    Feature selection is a important step to:
      - reduce overfitting
      - imporves accuracy
      - reduces Training Time
"""

import os
import sys
import time
import numpy
from threading import Thread
import numpy as np
import pandas as pd
from sklearn import linear_model
import pdb
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, defer, load_only
from models import Advertisement
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, SelectFromModel, chi2, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC



KEYS = [(0, 'living_area'), (0, 'floor'), (1, 'num_rooms'), (1, 'num_floors'), (2, 'build_year'),
        (2, 'last_renovation_year'), (3, 'cubature'), (3, 'room_height'), (4, 'effective_area'),
        (4, 'plot_area')]

# Set precision to 3
numpy.set_printoptions(precision=3)

class DataAnalysis():

    def __init__(self, from_file=True, file='./homegate.csv'):
        self.from_file = from_file
        if from_file:
            ads = self.load_dataset_from_file(file)
            self.ads = ads.drop(['id'], axis=1)
        else:
            try:
                engine = create_engine(os.environ.get('DATABASE_URL', None))
                Session = sessionmaker(bind=engine)
                self.session = Session()
                self.ads = self.load_dataset_from_database()
                self.ads.to_csv(file, header=True, encoding='utf-8')
            except AttributeError:
                raise Exception("If you want to load data from the database you have to export the DATABASE_URL environment")


    def load_dataset_from_file(self, file):
        """loads data from a csv file
        """
        return pd.read_csv(file)

    def load_dataset_from_database(self):
        """ load data from database
        """

        return pd.read_sql_query(self.session.query(Advertisement).options(
            load_only(
                'num_floors',
                'living_area',
                'floor',
                'price_brutto',
                'num_rooms',
                'object_types_id',
                'build_year',
                'last_renovation_year',
                'cubature',
                'room_height',
                'effective_area',
                'longitude',
                'latitude',
                'noise_level',
                'plot_area',
                'municipalities_id',
                'tags'
                )).statement, self.session.bind)

        # return pd.read_sql_query(self.session.query(Advertisement).options(defer('raw_data')).statement,
        #                          self.session.bind,
        #                          parse_dates=['crawled_at'])


    def simple_stats(self):
        total_amount_of_data = self.ads.shape[0]
        print("We have total {} values".format(total_amount_of_data))
        print("{:25} | {:6} | {:6} | {:6}".format("Feature",
                                                  "0-Values",
                                                  "NaN-Values",
                                                  "usable Values"))
        print("-"*70)
        total_zero = 0
        total_nan = 0
        total_use = 0

        for i, key in enumerate(self.ads.keys()):
            if key == 'id' or key == 'Unnamed':  # Keys from pandas we do not want
                continue
        # for i, key in KEYS:
            zero_values = self.ads.loc[self.ads[key] == 0][key].shape[0]
            nan_values = self.ads[key].isnull().sum()
            useful_values = total_amount_of_data - zero_values - nan_values

            # Sum up
            total_zero += zero_values
            total_nan += nan_values
            total_use += useful_values

            print("{:25} {:6} ({:02.2f}%) | {:6} ({:02.2f}%) | {:6} ({:02.0f}%)".format(key, zero_values, (zero_values/total_amount_of_data)*100,
                                                                                        nan_values, (nan_values/total_amount_of_data)*100,
                                                                                        useful_values, (useful_values/total_amount_of_data)*100))

        print("-"*70)
        print("{:25} {:6} | {:6} | {}".format('Total', total_zero, total_nan, total_use))


    def draw_features(self, show_null_values=False):
        """draw all features assicuiated to the price
        """
        fig, ax = plt.subplots(nrows=5, ncols=2)

        for i, key in enumerate(KEYS):
            current_axis = ax[key[0], i%2]
            zero_values = self.ads.loc[self.ads[key[1]] == 0][key[1]]
            non_zero_values = self.ads.loc[self.ads[key[1]] != 0][key[1]]

            df = pd.concat([zero_values, non_zero_values, self.ads.price_brutto],
                            axis=1, keys=['zero', 'non_zero', 'price_brutto'])

            df.plot(kind='scatter', x='price_brutto',
                    y='non_zero', color='DarkBlue', ax=current_axis, s=2)

            if show_null_values:
                df.plot(kind='scatter', x='price_brutto',
                        y='zero', color='Red', ax=current_axis, s=2)
            # set label and
            current_axis.set_xlabel('Price')
            current_axis.set_ylabel(key[1])
            current_axis.set_title('Price - {}'.format(key[1]))
            current_axis.get_xaxis().get_major_formatter().set_useOffset(False)
            current_axis.ticklabel_format(useOffset=False, style='plain')  # Do not show e^7 as label

        plt.show()

    def prepare_dataset(self):
        print("="*70)
        print("Dataset preparation:")
        print("-"*70)
        # Remove all entries with NaN / None
        self.ads_cleanup = self.ads.dropna()
        print("After remove all NaN the size of our dataset is {}".format(self.ads_cleanup.shape))
        self.y = self.ads_cleanup['price_brutto'].values
        # pdb.set_trace()
        self.X = self.ads_cleanup.drop(['price_brutto', 'id'], axis=1)
        self.keys = list(self.X.keys())[1:]
        self.X = self.X.values

    def transform_tags(self):
        with open('../crawler/taglist.txt') as f:
            search_words = set([x.split(':')[0] for x in f.read().splitlines()])

        template_dict = dict.fromkeys(search_words, 0)

        def transformer(row):
            the_dict = template_dict.copy()

            for tag in row.tags:
                the_dict[tag] = 1

            return pd.Series(the_dict)

        tag_columns = self.ads.apply(transformer, axis=1)

        self.ads = self.ads.drop(['tags'], axis=1).merge(tag_columns, left_index=True, right_index=True)


    # Features Selection
    def select_k_best(self):
        # Use the chi squared statistical test for the best 5 features
        k_best = SelectKBest(score_func=chi2, k=5)

        fit = k_best.fit(self.X, self.y)
        # summarize scores
        print("="*70)
        print("Select K Best fit scores:")
        print("-"*70)
        for key, fit_score in sorted(zip(self.keys, fit.scores_), key=lambda x: x[1], reverse=True):
            print("{:25}: {:6}".format(key, fit_score))

        features = fit.transform(self.X)  # fit.fit_transform(self.X, self.y)

        # summarize selected features
        print("The first 5 features \n{}".format(features[0:6,:]))


    def recursive_feature_elimination(self):
        print("="*70)
        print("Recursive Feature elimination")
        print("="*70)
        model = LogisticRegression()
        rfe = RFE(model, 5)  # Get the top 5 features
        fit = rfe.fit(self.X, self.y)

        for key, f in zip(self.keys, fit.support_):
            print("{:25} {}".format(key, f))

        print("Rankin: {}".format(fit.ranking_))

    def pricipal_component_analysis(self):
        print("="*70)
        print("Principal Component Analysis")
        print("="*70)

        pca = PCA(n_components=5)
        fit = pca.fit(self.X)

        # summarize components
        print("Explained Variance: {}".format(fit.explained_variance_ratio_))
        print(fit.components_)


    def L1(self):
        print("="*70)
        print("L1-Based")
        print("="*70)

        lsvc = LinearSVC(C=0.01, penalty="l1", dual=False)
        fit = lsvc.fit(self.X, self.y)

        model = SelectFromModel(lsvc, prefit=True)
        X_new = model.transform(self.X)
        print("New shape of the matrix: {} old was {}. Removed {} cols".format(X_new.shape,
                                                                               self.X.shape,
                                                                               self.X.shape[1] - X_new.shape[1]))
        print(X_new)

    def plotting(self):
        import seaborn as sns
        # Price distplot
        # Clean up ads
        ads = self.ads[self.ads.price_brutto <= 10000000]
        ax = plt.axes()

        ax.set_title("Vertielung des Kaufpreises")
        sns.distplot(ads['price_brutto'], kde=True, bins=100, hist_kws={'alpha': 0.6}, ax=ax)
        ax.set_xlabel("Kaufpreis")
        ax.get_xaxis().get_major_formatter().set_useOffset(False)
        plt.savefig("images/Verteilung_des_kauf_preises.png")
        print("Distplot - OK")
        plt.clf()
        plt.close()

        # Heatmap of features:
        corr = ads.select_dtypes(include = ['float64', 'int64']).iloc[:, 1:].corr()
        plt.figure(figsize=(12, 12))
        hm = sns.heatmap(corr, vmin=-1, vmax=1, square=True)
        hm.set_xticklabels(hm.get_xticklabels(), rotation=90)
        hm.set_yticklabels(reversed(hm.get_xticklabels()), rotation=0)
        plt.savefig("images/Heatmap.png")
        print("Heatmap - OK")
        plt.clf()
        plt.close()

        cor_dict = corr['price_brutto'].to_dict()
        del cor_dict['price_brutto']
        print("List the numerical features decendingly by their correlation with Sale Price:\n")
        for ele in sorted(cor_dict.items(), key = lambda x: -abs(x[1])):
            print("{0}: \t{1}".format(*ele))

        # Now all features compared to price_brutto
        plt.figure(1)
        f, ax = plt.subplots(3, 2, figsize=(10, 9))
        price = ads.price_brutto.values
        ax[0, 0].scatter(ads.num_rooms.values, price)
        ax[0, 0].set_title('Anzahl Zimmer')
        #ax[0, 0].set_ylabel('Preis')
        ax[0, 1].scatter(ads.living_area.values, price)
        ax[0, 1].set_title('Wohnfläche')
        #ax[0, 1].set_ylabel('Preis')
        ax[1, 0].scatter(ads.effective_area.values, price)
        ax[1, 0].set_title('Effektive Fläche')
        ax[1, 0].set_ylabel('Preis')
        ax[1, 1].scatter(ads.build_year.values, price)
        ax[1, 1].set_title('Baujahr')
        #ax[1, 1].set_ylabel('Preis')
        ax[2, 0].scatter(ads.num_floors.values, price)
        ax[2, 0].set_title('Anzahl Stockwerke')
        #ax[2, 0].set_ylabel('Preis')
        ax[2, 1].scatter(ads.noise_level.values, price)
        ax[2, 1].set_title('Lärmbelastung')
        #ax[2, 1].set_ylabel('Preis')
        #f.text(10.01, 0.5, 'Kaufpreis', va='center', rotation='vertical', fontsize = 32)
        plt.tight_layout()
        plt.savefig("images/Vergleich_zum_preis.png")
        print("Vergleich - OK")
        plt.clf()
        plt.close()



def main():
    data_analysis = DataAnalysis(from_file=True, file='all_noise.csv')
    data_analysis.plotting()
    return
    data_analysis.transform_tags()
    data_analysis.simple_stats()
    data_analysis.draw_features(show_null_values=True)
    data_analysis.prepare_dataset()
    # data_analysis.select_k_best()
    # data_analysis.recursive_feature_elimination()
    data_analysis.pricipal_component_analysis()
    # data_analysis.L1()

    # model.fit(X, y)
    # x = np.array(X[:, 1].A1)
    # f = model.predict(X).flatten()

    # fig, ax = plt.subplots(figsize=(12, 8))
    # ax.plot(x, f, 'r', label='Prediction')
    # ax.scatter(ads.living_area, ads.price_brutto, label='Traning Data')
    # ax.legend(loc=2)
    # ax.set_xlabel('Living Area')
    # ax.set_ylabel('Price')
    # ax.set_title('Predicted Price vs. Living Area')
    # plt.show()


if __name__ == "__main__":
    main()
