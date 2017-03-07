import os
import sys
import time
from threading import Thread
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, defer
from models import Advertisement


class DataAnalysis(Thread):

    def __init__(self, from_file=False):
        self.advertisement = None
        self.from_file = from_file
        super(DataAnalysis, self).__init__()

    def run(self):
        if self.from_file:
            self.advertisement = pd.read_csv('./homegate.csv')
            return

        self.advertisements = pd.read_sql_query(session.query(Advertisement).options(defer('raw_data')).filter(Advertisement.crawler == 'homegate').statement,
                                                session.bind,
                                                parse_dates=['crawled_at'])

        self.advertisements.to_csv('./homegate.csv', header=True, sep='\t', encoding='utf-8')

def main():

    thread = DataAnalysis()
    thread.start()
    sys.stdout.write("Get data ")
    while thread.is_alive():
        sys.stdout.write(".")
        sys.stdout.flush()
        time.sleep(5)
    advertisements = thread.advertisements

    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(15, 10))

    advertisements.plot(kind='scatter', x='price_brutto', y ='living_area', ax=ax[0, 0])
    ax[0, 0].set_xlabel('Price')
    ax[0, 0].set_ylabel('Living Area')
    ax[0, 0].set_title('Price - Living Area')
    ax[0, 0].get_xaxis().get_major_formatter().set_useOffset(False)


    advertisements.plot(kind='scatter', x='price_brutto', y ='num_rooms', ax=ax[0, 1])
    ax[0, 1].set_xlabel('Price')
    ax[0, 1].set_ylabel('Nr. Rooms')
    ax[0, 1].set_title('Price - Nr. rooms')
    ax[0, 1].get_xaxis().get_major_formatter().set_useOffset(False)


    advertisements.plot(kind='scatter', x='price_brutto', y ='floor', ax=ax[1, 0])
    ax[1, 0].set_xlabel('Price')
    ax[1, 0].set_ylabel('Floor')
    ax[1, 0].set_title('Price - Floor')
    ax[1, 0].get_xaxis().get_major_formatter().set_useOffset(False)


    advertisements.plot(kind='scatter', x='price_brutto', y ='effective_area', ax=ax[1, 1])
    ax[1, 1].set_xlabel('Price')
    ax[1, 1].set_ylabel('Effective Area')
    ax[1, 1].set_title('Price - Effective Area')
    ax[1, 1].get_xaxis().get_major_formatter().set_useOffset(False)

    advertisements.plot(kind='scatter', x='price_brutto', y ='floors_house', ax=ax[2, 0])
    ax[2, 0].set_xlabel('Price')
    ax[2, 0].set_ylabel('Nr of floors in the house')
    ax[2, 0].set_title('Price - Numbers of floors')
    ax[2, 0].get_xaxis().get_major_formatter().set_useOffset(False)

    plt.show()

if __name__ == "__main__":
    engine = create_engine(os.environ.get('DATABASE_URL', None))
    Session = sessionmaker(bind=engine)
    session = Session()
    main()