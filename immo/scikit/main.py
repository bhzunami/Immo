#!/usr/bin/env python
"""
http://bbengfort.github.io/tutorials/2016/05/19/text-classification-nltk-sckit-learn.html
https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words
"""
import matplotlib
matplotlib.use('Agg')
import os
import pdb
import logging
import json
import argparse
import numpy as np
import pandas as pd

from train_pipeline import TrainPipeline

DIRECTORY = os.path.dirname(os.path.abspath(__file__))
settings = json.load(open('{}/settings.json'.format(DIRECTORY)))

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
                    default='{}/all_2.csv'.format(DIRECTORY))

args = parser.parse_args()

def main(args):
    logging.info("Start with args {}".format(args))    
    ads = pd.read_csv(args.file, index_col=0, engine='c')
    
    if args.train:
        tp = TrainPipeline(args.goal, settings, DIRECTORY)
        for f in tp.pipeline:
            logging.info("Apply transformation: {}".format(f.__name__))
            ads = f(ads)

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
