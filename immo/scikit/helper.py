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

    ax = plt.axes()
    ax.plot(y_test, 'bo', markersize=markersize, label="Actual")
    ax.plot(y_pred, 'ro', markersize=markersize, label="Predicted")
    ax.set_title("Preisverteilung")
    ax.set_xlabel("Inserat Nr.")
    ax.set_ylabel("Preis")
    plt.tight_layout()
    plt.savefig("{}/{}_verteilung_preis.png".format(image_folder, title), dpi=250)
    plt.clf()
    plt.close()

    ax = plt.axes()
    ax.plot(y_pred, y_test - y_pred, 'bo', markersize=markersize)
    ax.plot((np.min(y_pred), np.max(y_pred)), (0, 0), 'r-')
    ax.set_title("Tukey Anscombe")
    ax.set_xlabel("Geschätzte Werte")
    ax.set_ylabel("Residuals")
    plt.tight_layout()
    plt.savefig("{}/{}_tukey_anscombe.png".format(image_folder, title), dpi=250)
    plt.clf()
    plt.close()

    # Title Absolute Abweichung in %
    ax = plt.axes()
    ax.hist(np.nan_to_num(np.log1p(y_test) - np.log1p(y_pred)), 100)
    ax.set_xticklabels(['{:.2}'.format(x) for x in ax.get_xticks()])
    ax.set_title("Absolute Abweichung")
    ax.set_xlabel("Residuen loge(y) − loge(yˆ)")
    ax.set_ylabel("Anzahl Inserate")
    plt.tight_layout()
    plt.savefig("{}/{}_verteilung_residuals_log.png".format(image_folder, title), dpi=250)
    plt.clf()
    plt.close()


def train_statistics(y_test, y_pred, title="dummy"):
    logging.info("Statistics for: {}".format(title))
    diff = np.fabs(y_test - y_pred)
    my_ape = ape(y_test, y_pred)
    logging.info("             R²-Score: {:10n}".format(metrics.r2_score(y_test, y_pred)))
    logging.info("                 MAPE: {:.3%}".format(mape(y_test, y_pred)))
    logging.info("                MdAPE: {:.3%}".format(mdape(y_test, y_pred)))
    logging.info("            Min error: {:10n}".format(np.amin(diff)))
    logging.info("            Max error: {:10n}".format(np.amax(diff)))
    logging.info("          Max error %: {:10n}".format(np.amax(my_ape)))
    logging.info("  Mean absolute error: {:10n}".format(metrics.mean_absolute_error(y_test, y_pred)))
    logging.info("Median absolute error: {:10n}".format(metrics.median_absolute_error(y_test, y_pred)))
    logging.info("   Mean squared error: {:10n}".format(metrics.mean_squared_error(y_test, y_pred)))

    num_elements = len(y_pred)
    apes = ape(y_test, y_pred)
    for i in np.arange(5, 100, 5):
        logging.info("I {:2n}: {:.3%}".format(i, len(np.where(apes < i/100)[0])/num_elements))

