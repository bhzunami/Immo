import matplotlib
matplotlib.use('agg')
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
import pdb
import json

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, load_only, Load
from models import Advertisement, Municipality, ObjectType

# Set precision to 3
np.set_printoptions(precision=3)

class DataAnalysis():
    def __init__(self, file='./homegate.csv'):

        self.synopsis = json.load(open('synopsis.json'))

        if os.path.isfile(file):
            print("Use file")
            ads = pd.read_csv(file, index_col=0, engine='c')
        else:
            try:
                engine = create_engine(os.environ.get('DATABASE_URL', None))
                Session = sessionmaker(bind=engine)
                self.session = Session()
                ads = self.load_dataset_from_database()
                ads.to_csv(file, header=True, encoding='utf-8')
            except AttributeError as e:
                raise Exception("If you want to load data from the database you have to export the DATABASE_URL environment")

        self.ads = ads
    def load_dataset_from_database(self):
        """ load data from database
        """
        statement = self.session.query(Advertisement, Municipality, ObjectType).join(Municipality, ObjectType).options(
            Load(Advertisement).load_only(
                "price_brutto",
                "crawler",
                "num_floors",
                "living_area",
                "floor",
                "num_rooms",
                "build_year",
                "last_renovation_year",
                "cubature",
                "room_height",
                "effective_area",
                "longitude",
                "latitude",
                "noise_level",
                "plot_area",
                "tags"),
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
                "municipal_type9_id"),
            Load(ObjectType).load_only("name", "grouping")
        ).with_labels().statement
        data = pd.read_sql_query(statement, self.session.bind)

        data.drop(['advertisements_id', 'municipalities_id', 'object_types_id'], axis=1, inplace=True)
        # Rename
        return data.rename(columns={'advertisements_price_brutto': 'price',
                                    'advertisements_crawler': 'crawler',
                                    'advertisements_living_area': 'living_area',
                                    'advertisements_floor': 'floor',
                                    'advertisements_num_rooms': 'num_rooms',
                                    'advertisements_num_floors': 'num_floors',
                                    'advertisements_build_year': 'build_year',
                                    'advertisements_last_renovation_year': 'last_renovation_year',
                                    'advertisements_cubature': 'cubature',
                                    'advertisements_room_height': 'room_height',
                                    'advertisements_effective_area': 'effective_area',
                                    'advertisements_plot_area': 'plot_area',
                                    'advertisements_longitude': 'longitude',
                                    'advertisements_latitude': 'latitude',
                                    'advertisements_noise_level': 'noise_level',
                                    'advertisements_tags': 'tags',
                                    'municipalities_name': 'municipality',
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

        # Cleanup the datakeys

    def simple_stats(self):
        print("We have total {} values".format(len(self.ads)))
        print("{:25} | {:6} | {:6}".format("Feature",
                                           "NaN-Values",
                                           "usable Values"))
        print("-"*70)
        for i, key in enumerate(self.ads.keys()):
            if key == 'id' or key == 'Unnamed':  # Keys from pandas we do not want
                continue
            nan_values = self.ads[key].isnull().sum()
            useful_values = len(self.ads) - nan_values

            print("{:25} {:6} ({:02.2f}%) | {:6} ({:02.0f}%)".format(key,
                                                                    nan_values,
                                                                    (nan_values/len(self.ads))*100,
                                                                    useful_values,
                                                                    (useful_values/len(self.ads))*100))
        # Missing data
        # Calculate percent of missing data
        missing_data = (self.ads.isnull().sum() / len(self.ads)) * 100
        # Remove itmes we have 100% and sort
        missing_data = missing_data.drop(missing_data[missing_data == 0].index).sort_values(ascending=False)
        b = sns.barplot(x=missing_data.index, y=missing_data)
        plt.xlabel('Features')
        plt.ylabel('% von fehlenden Werten')
        plt.title('Fehlende Features in %')
        plt.xticks(rotation='90')
        plt.tight_layout()
        for text in b.get_xticklabels():
            text.set_text(text.get_text().replace("_", " ").title())
        plt.savefig("images/analysis/missing_values.png", dpi=250)
        plt.clf()
        plt.close()

    def clean_dataset(self):
        print("="*70)
        print("Dataset preparation:")
        print("-"*70)
        # Remove elements with no price
        ads = self.ads.dropna(subset=['price'])
        removed_ads_with_missing_price = len(self.ads) - len(ads)
        print("Removed {} ads because we do not have a price.".format(removed_ads_with_missing_price))
        # Cleanup some outliers
        ads = ads.drop(ads[ads['num_floors'] > 20].index)
        ads = ads.drop(ads[ads['price'] > 20000000].index)
        ads = ads.drop(ads[ads['price'] < 10].index)
        ads = ads.drop(ads[ads['living_area'] > 5000].index)
        ads = ads.drop(ads[ads['num_rooms'] > 20].index)
        ads = ads.drop(ads[ads['build_year'] < 1200].index)
        ads = ads.drop(ads[ads['build_year'] > 2050].index)
        ads = ads.drop(ads[ads['last_renovation_year'] < 1200].index)
        ads = ads.drop(ads[ads['last_renovation_year'] > 2050].index)
        ads = ads.drop(ads[ads['cubature'] > 20000].index)
        ads = ads.drop(ads[ads['floor'] > 30].index)
        # Remove to lower values
        # ads = ads.drop(ads[ads['living_area'] < 20].index)
        # ads = ads.drop(ads[ads['cubature'] < 20].index)
        # ads = ads.drop(ads[ads['num_rooms'] < 1].index)
        print("Removed {} outliers. Dataset size: {}".format(len(self.ads) - len(ads) - removed_ads_with_missing_price, len(ads)))

        #print("Describe: \n{}".format(ads.describe()))
        
        print("Nummerical features:")
        print(ads.num_rooms.describe())
        print(ads.living_area.describe())
        print(ads.build_year.describe())
        print(ads.num_floors.describe())
        print(ads.cubature.describe())
        print(ads.floor.describe())
        print(ads.noise_level.describe())
        print(ads.last_renovation_year.describe())        
        self.ads = ads

    def plot_numerical_values(self):
        ax = plt.axes()
        ax.set_title("Verteilung des Kaufpreises")
        sns.distplot(self.ads['price'], kde=True, bins=100, hist_kws={'alpha': 0.6}, ax=ax)
        ax.set_xlabel("Kaufpreis CHF")
        plt.savefig("images/analysis/Verteilung_des_kauf_preises.png", dpi=250)
        print("Distplot - OK")
        plt.clf()
        plt.close()
        
        ax = plt.axes()
        ax.set_title("Verteilung des Kaufpreises mit log")
        sns.distplot(np.log1p(self.ads['price']), kde=True, bins=100, hist_kws={'alpha': 0.6}, ax=ax)
        ax.set_xlabel("Kaufpreis CHF (log)")
        plt.savefig("images/analysis/Verteilung_des_kauf_preises_log.png", dpi=250)
        print("Distplot - OK")
        plt.clf()
        plt.close()

        for f, name in [('num_rooms', 'Anzahl Zimmer'),
                        ('living_area', 'Fläche [m^2]'),
                        ('noise_level', 'Lärmbelastung')]:
            ax = plt.axes()
            ax.set_title("Verteilung der {}".format(name))
            sns.distplot(self.ads[f].dropna(), kde=False, bins=100, hist_kws={'alpha': 0.6}, ax=ax)
            ax.set_xlabel("{}".format(name))
            plt.savefig("images/analysis/Verteilung_{}.png".format(f), dpi=250)
            print("Distplot - OK")
            plt.clf()
            plt.close()

        # Heatmap of features:
        corr = self.ads.select_dtypes(include = ['float64', 'int64']).corr()
        plt.figure(figsize=(12, 12))
        hm = sns.heatmap(corr, vmin=-1, vmax=1, square=True)
        for text in hm.get_xticklabels():
            text.set_text(text.get_text().replace("_", " ").replace("id", "").title())
        hm.set_xticklabels(hm.get_xticklabels(), rotation=90)
        hm.set_yticklabels(reversed(hm.get_xticklabels()), rotation=0)
        hm.set_title("Heatmap aller Features", fontsize=20)
        plt.savefig("images/analysis/Heatmap_all.png", dpi=250)
        print("Heatmap all - OK")
        plt.clf()
        plt.close()

        corr = self.ads.select_dtypes(include = ['float64']).corr()
        plt.figure(figsize=(12, 12))
        hm = sns.heatmap(corr, vmin=-1, vmax=1, square=True)
        for text in hm.get_xticklabels():
            text.set_text(text.get_text().replace("_", " ").replace("id", "").title())
        hm.set_xticklabels(hm.get_xticklabels(), rotation=90)
        hm.set_yticklabels(reversed(hm.get_xticklabels()), rotation=0)
        hm.set_title("Heatmap numerischer Features", fontsize=20)
        plt.savefig("images/analysis/Heatmap_numerical.png", dpi=250)
        print("Heatmap Numerical - OK")
        plt.clf()
        plt.close()

        cor_dict = corr['price'].to_dict()
        del cor_dict['price']
        print("List the numerical features decendingly by their correlation with Sale Price:\n")
        for ele in sorted(cor_dict.items(), key = lambda x: -abs(x[1])):
            print("{0}: \t{1}".format(*ele))

        # Now all features compared to price
        plt.figure(1)
        f, ax = plt.subplots(4, 2, figsize=(10, 9))
        price = self.ads.price.values
        ax[0, 0].scatter(self.ads.num_rooms.values, price)
        ax[0, 0].set_title('Anzahl Zimmer')
        ax[0, 1].scatter(self.ads.living_area.values, price)
        ax[0, 1].set_title('Wohnfläche [m²]')
        ax[1, 0].scatter(self.ads.build_year.values, price)
        ax[1, 0].set_title('Baujahr')
        ax[1, 0].set_ylabel('Preis')
        ax[1, 1].scatter(self.ads.num_floors.values, price)
        ax[1, 1].set_title('Anzahl Stockwerke')
        ax[2, 0].scatter(self.ads.cubature.values, price)
        ax[2, 0].set_title('Cubature')
        ax[2, 1].scatter(self.ads.floor.values, price)
        ax[2, 1].set_title('Stockwerk')
        ax[3, 0].scatter(self.ads.noise_level.values, price)
        ax[3, 0].set_title('Lärmbelastung')
        ax[3, 1].scatter(self.ads.last_renovation_year.values, price)
        ax[3, 1].set_title('Letzte Renovaton')
        plt.tight_layout()
        plt.savefig("images/analysis/Vergleich_zum_preis.png", dpi=250)
        print("Vergleich - OK")
        plt.clf()
        plt.close()

        fig = plt.figure()
        from scipy import stats
        res = stats.probplot(self.ads['price'], plot=plt)
        plt.savefig("images/analysis/skewness.png", dpi=250)
        print("skewness - OK")
        plt.clf()
        plt.close()
        fig = plt.figure()
        res = stats.probplot(np.log1p(self.ads['price']), plot=plt)
        plt.savefig("images/analysis/log_skewness.png", dpi=250)
        print("Log skewness - OK")
        plt.clf()
        plt.close()

    def plot_categorical_features(self):

        ax = plt.axes()
        b = sns.boxplot(x='canton_id', y='price', data=self.ads, ax=ax)
        b.set_xticklabels(self.synopsis['CANTON_ID'], rotation=90)
        ax.set_xlabel("")
        ax.set_ylabel("Kaufpreis CHF")
        ax.set_title("Kaufpreise auf Kantone")
        plt.tight_layout()
        plt.savefig("images/analysis/boxPlot_cantons.png", dpi=250)
        print("boxplot cantons - OK")
        plt.clf()
        plt.close()

        ax = plt.axes()
        b = sns.barplot(x='canton_id', y='price', data=self.ads, ax=ax)
        b.set_xticklabels(self.synopsis['CANTON_ID'], rotation=90)
        ax.set_xlabel("")
        ax.set_ylabel("Kaufpreis CHF (Durchschnitt)")
        ax.set_title("Kaufpreise auf Kantone")
        plt.tight_layout()
        plt.savefig("images/analysis/barplot_canton.png", dpi=250)
        print("barplot canton - OK")
        plt.clf()
        plt.close()

        ax = plt.axes()
        b = sns.barplot(x='otype', y='price', data=self.ads, ax=ax)
        b.set_xticklabels(b.get_xticklabels(), rotation=90)
        plt.tight_layout()
        ax.set_xlabel("")
        ax.set_ylabel("Kaufpreis CHF (Durchschnitt)")
        plt.savefig("images/analysis/barplot_gruppen.png", dpi=250)
        print("barplot Gruppen - OK")
        plt.clf()
        plt.close()

        for key in ['TOURISM_REGION_ID', 'METROPOLE_REGION_ID', 'JOB_MARKET_REGION_ID',
                    'MOUNTAIN_REGION_ID', 'LANGUAGE_REGION_ID', 'MUNICIPAL_SIZE_CLASS_ID',
                    'GREATER_REGION_ID', 'AGGLOMERATION_SIZE_CLASS_ID',
                    'IS_TOWN', 'DEGURBA_ID']:
            ax = plt.axes()
            b = sns.barplot(x=key.lower(), y='price', data=self.ads, ax=ax)
            b.set_xticklabels(self.synopsis[key], rotation=90)
            ax.set_xlabel("")
            ax.set_ylabel("Kaufpreis CHF (Durchschnitt)")
            ax.set_title(key.replace('_', ' ').replace('ID', '').title())
            plt.tight_layout()
            plt.savefig("images/analysis/barplot_{}.png".format(key.lower()), dpi=250)
            print("barplot {} - OK".format(key.lower()))
            plt.clf()
            plt.close()

            # Boxplot only have data where price is lower 5 millions (Graphical better)
            ax = plt.axes()
            b = sns.boxplot(x=key.lower(), y='price',
                            data=self.ads[self.ads.price < 5000000],
                            ax=ax)
            b.set_xticklabels(self.synopsis[key], rotation=90)
            ax.set_xlabel("")
            ax.set_ylabel("Kaufpreis CHF (Durchschnitt)")
            ax.set_title(key.replace('_', ' ').replace('ID', '').title())
            plt.tight_layout()
            plt.savefig("images/analysis/boxplot_{}.png".format(key.lower()), dpi=250)
            print("boxplot {} - OK".format(key.lower()))
            plt.clf()
            plt.close()




def main():
    data_analysis = DataAnalysis(file='advertisements.csv')
    data_analysis.simple_stats()
    data_analysis.clean_dataset()
    data_analysis.plot_numerical_values()
    data_analysis.plot_categorical_features()


if __name__ == "__main__":
    main()
