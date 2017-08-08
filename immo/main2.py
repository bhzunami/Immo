#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
import os
import pdb
import logging
import json
import argparse
import numpy as np
import pandas as pd
import sys

from .train_pipeline import TrainPipeline

DIRECTORY = os.path.dirname(os.path.abspath(__file__))
settings = json.load(open('{}/settings.json'.format(DIRECTORY)))

logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(levelname)s - %(message)s', filename='train.log')
logging.getLogger().addHandler(logging.StreamHandler())

def main(commands):

    tp = TrainPipeline('price', settings, DIRECTORY)
    ads = None

    for line in commands:
        logging.info("Apply transformation: {}".format(line[0]))
        func = getattr(tp, line[0])
        if len(line) > 1:
            line.pop(0)
            func = func(*line)
        ads = func(ads)

    logging.info("Pipeline finished.")

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    commands = json.load(open("commands.json"))
    main(commands)
