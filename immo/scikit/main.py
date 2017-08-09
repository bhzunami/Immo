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

from .pipeline import Pipeline
DIRECTORY = os.path.dirname(os.path.abspath(__file__))
settings = json.load(open('{}/settings.json'.format(DIRECTORY)))

logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(levelname)s - %(message)s', filename='train.log')
logging.getLogger().addHandler(logging.StreamHandler())

def main(commands):

    p = Pipeline('price', settings, DIRECTORY)
    ads = None

    for line in commands:
        if line[0] != "echo":
            logging.info("Apply transformation: {}".format(line[0]))
        func = getattr(p, line[0])
        if len(line) > 1:
            line.pop(0)
            func = func(*line)
        try:
            ads = func(ads)
        except Exception:
            logging.error("Could not run method {}".format(line[0]))
            continue

    logging.info("Pipeline finished.")

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    commands = json.load(open("allCommands.json"))
    main(commands)
