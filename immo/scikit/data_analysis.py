import os
import sys
import time
from threading import Thread
import numpy as np
import pandas as pd
from sklearn import linear_model
import pdb
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, defer
from models import Advertisement
from sklearn.metrics import accuracy_score

KEYS = [(0, 'living_area'), (0, 'floor'), (1, 'num_rooms'), (1, 'num_floors'), (2, 'build_year'),
            (2, 'last_renovation_year'), (3, 'cubature'), (3, 'room_height'), (4, 'effective_area'),
            (4, 'plot_area')]

def draw_plt(ax, ads, name, pos):
    zero_values = ads.loc[ads[name] == 0][name]
    non_zero_values = ads.loc[ads[name] != 0][name]

    df = pd.concat([zero_values, non_zero_values, ads.price_brutto],
                   axis=1, keys=['zero', 'non_zero', 'price_brutto'])
    df.plot(kind='scatter', x='price_brutto',
            y='non_zero', color='DarkBlue', ax=pos, s=2)
    # df.plot(kind='scatter', x='price_brutto', y='zero', color='Red', ax=pos, s=2)
    pos.set_xlabel('Price')
    pos.set_ylabel(name)
    pos.set_title('Price - {}'.format(name))
    pos.get_xaxis().get_major_formatter().set_useOffset(False)
    pos.ticklabel_format(useOffset=False, style='plain')


def plot_data(advertisements):
    fig, ax = plt.subplots(nrows=5, ncols=2)
    
    for i, key in enumerate(KEYS):
        draw_plt(ax, advertisements, key[1], ax[key[0], i % 2])

    plt.show()

def show_stats(ads):
    total_amount_of_data = ads.shape[0]
    print("We have total {} values".format(total_amount_of_data))
    print("Feature                  0-Values   NaN-Values   usable-Values")
    print("-"*70)

    total_zero = 0
    total_nan = 0
    total_use = 0
    for i, key in KEYS:
        zero_values = ads.loc[ads[key] == 0][key].shape[0]
        nan_values = ads[key].isnull().sum()
        useful_values = total_amount_of_data - zero_values - nan_values

        # Sum up
        total_zero += zero_values
        total_nan += nan_values
        total_use += useful_values

        print("{:25} {:6} ({:02.2f}%) | {:6} ({:02.2f}%) | {:6} ({:02.0f}%)".format(key, zero_values, (zero_values/total_amount_of_data)*100, nan_values, (nan_values/total_amount_of_data)*100, useful_values, (useful_values/total_amount_of_data)*100))

    print("-"*70)
    print("{:25} {:6}  |  {:6}   |  {}".format('Total', total_zero, total_nan, total_use))


class DataAnalysis(Thread):

    def __init__(self, from_file=False):
        self.advertisement = None
        self.from_file = from_file
        super(DataAnalysis, self).__init__()

    def run(self):
        self.get_data()

    def get_data(self):
        if self.from_file:
            self.advertisements = pd.read_csv('./homegate.csv')
            return

        self.advertisements = pd.read_sql_query(session.query(Advertisement).options(defer('raw_data')).filter(Advertisement.crawler == 'homegate').statement,
                                                session.bind,
                                                parse_dates=['crawled_at'])

        self.advertisements.to_csv(
            './homegate.csv', header=True, encoding='utf-8')


def main():

    thread = DataAnalysis(from_file=True)
    thread.start()
    sys.stdout.write("Get data ")
    while thread.is_alive():
        sys.stdout.write(".")
        sys.stdout.flush()
        time.sleep(5)
    print()  # New line
    advertisements = thread.advertisements
    # plot_data(advertisements)
    show_stats(advertisements)

    advertisements.insert(0, 'ones', 1)

    # Remove these houses where price is NaN
    print("Missing price for {} ads".format(
        len(advertisements.price_brutto.loc[advertisements.price_brutto == 0])))

    # ads = advertisements[advertisements.price_brutto.notnull()]
    ads = advertisements.dropna()
    # pdb.set_trace()
    print("SIZE: {}".format(len(ads)))
    X = ads[['ones', 'living_area']]
    y = ads[['price_brutto']]

    X = np.matrix(X.values)
    y = np.matrix(y.values)

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
    engine = create_engine(os.environ.get('DATABASE_URL', None))
    Session = sessionmaker(bind=engine)
    session = Session()
    main()
