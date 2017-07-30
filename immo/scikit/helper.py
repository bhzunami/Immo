import matplotlib
import logging
from sklearn import metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdb

def generate_matrix(df, goal):
    X = df.drop([goal], axis=1)
    y = df[goal].astype(float)
    return X, y

def ape(y_test, y_pred):
    return np.abs((y_test - y_pred) / y_test)

def mape(y_test, y_pred):
    return ape(y_test, y_pred).mean()

def mdape(y_test, y_pred):
    return np.median(ape(y_test, y_pred))

def gen_subplots(fig, x, y):
    for i in range(x*y):
        ax = fig.add_subplot(x, y, i+1)
        plt.grid()
        yield ax

def plot(y_test, y_pred, image_folder, show=False, title="dummy"):
    # sort both arrays by y_test
    y_test, y_pred = zip(*sorted(zip(y_test, y_pred)))
    y_test = np.asarray(y_test)
    y_pred = np.asarray(y_pred)
    markersize = 1
    fig = plt.figure(figsize=(10, 10))
    subplots = gen_subplots(fig, 3, 1)

    ax = next(subplots)
    ax.set_xlabel('Actual vs. Predicted values')
    ax.plot(y_test, 'bo', markersize=markersize, label="Actual")
    ax.plot(y_pred, 'ro', markersize=markersize, label="Predicted")

    ax = next(subplots)
    ax.set_xlabel('Predicted value')
    ax.set_ylabel('Residuals')
    ax.plot(y_pred, y_test - y_pred, 'bo', markersize=markersize)
    ax.plot((np.min(y_pred), np.max(y_pred)), (0, 0), 'r-')

    plt.tight_layout()
    plt.savefig("{}/{}_cumulative_probability.png".format(image_folder, title))
    if show:
        plt.show()

    fig = plt.figure(figsize=(5, 5))
    subplots = gen_subplots(fig, 1, 1)
    ax = next(subplots)
    ax.set_xlabel('APE')
    my_ape = ape(y_test, y_pred)
    ax.hist(my_ape, 100)
    mape = ape(y_test, y_pred).mean()
    ax.plot((mape, mape), (0, 400), 'r-')

    mdape = np.median(ape(y_test, y_pred))
    ax.plot((mdape, mdape), (0, 400), 'y-')

    ax.set_xticklabels(['{:.2%}'.format(x) for x in ax.get_xticks()])
    plt.tight_layout()
    plt.savefig("{}/{}_verteilung_der_fehler.png".format(image_folder, title))
    if show:
        plt.show()

def train_statistics(y_test, y_pred, title="dummy"):
    logging.info("Statistics for: {}".format(title))
    diff = np.fabs(y_test - y_pred)
    logging.info("             R²-Score: {:10n}".format(metrics.r2_score(y_test, y_pred)))
    logging.info("                 MAPE: {:.3%}".format(mape(y_test, y_pred)))
    logging.info("                MdAPE: {:.3%}".format(mdape(y_test, y_pred)))
    logging.info("            Min error: {:10n}".format(np.amin(diff)))
    logging.info("            Max error: {:10n}".format(np.amax(diff)))
    logging.info("  Mean absolute error: {:10n}".format(metrics.mean_absolute_error(y_test, y_pred)))
    logging.info("Median absolute error: {:10n}".format(metrics.median_absolute_error(y_test, y_pred)))
    logging.info("   Mean squared error: {:10n}".format(metrics.mean_squared_error(y_test, y_pred)))

    num_elements = len(y_pred)
    apes = ape(y_test, y_pred)
    for i in np.arange(5, 100, 5):
        logging.info("I {}: {}".format(i, len(np.where(apes < i/100)[0])/num_elements))

