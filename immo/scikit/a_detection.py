from sklearn.ensemble import IsolationForest
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import pdb
import logging

RNG = np.random.RandomState(42)

class AnomalyDetection(object):

    def __init__(self, data, image_folder, model_folder, goal='price'):
        self.data = data
        self.goal = goal
        self.image_folder = image_folder
        self.model_folder = model_folder

    def plot_isolation_forest(self, data, cls_, x, y, outlierIdx, elements):
        if not os.path.exists('{}/outlier_detection'.format(self.image_folder)):
            logging.info("Directory for outliers does not exists. Create one")
            os.makedirs('{}/outlier_detection'.format(self.image_folder))
        for i, feature in enumerate(elements):
            Z = cls_[feature].decision_function(np.c_[x[feature].ravel(), y[feature].ravel()])
            Z = Z.reshape(x[feature].shape)
            plt.contourf(x[feature],
                         y[feature], Z,
                         cmap=plt.cm.Blues_r)

            b1 = plt.scatter(data[feature][np.where(outlierIdx[feature] == -1)][:, 0].tolist(),
                             data[feature][np.where(outlierIdx[feature] == -1)][:, 1].tolist(),
                             c='red', s=2)

            b2 = plt.scatter(data[feature][np.where(outlierIdx[feature] == 1)][:, 0].tolist(),
                             data[feature][np.where(outlierIdx[feature] == 1)][:, 1].tolist(),
                             c='green', s=2)

            plt.title('Price - {}'.format(feature.replace("_", " ")))
            plt.ylabel('Price')
            plt.xlabel('{}'.format(feature.replace("_", " ")))
            plt.axis('tight')

            plt.xlim((min(x[feature][0]), max(x[feature][0])))
            plt.ylim(0, max(self.data[self.goal]))
            plt.legend([b1, b2], ["Outlier", "Accepted values"], loc="upper left", frameon=True)
            plt.savefig("{}/outlier_detection/{}_IsolationForest.png".format(self.image_folder, feature), transparent=True, dpi=250)
            plt.close()

    def isolation_forest(self,
                         settings,
                         meshgrid,
                         goal='price',
                         load=False):
        """ get a dataframe and features to check for anomaly detection
        returns a new dataframe with removed anomalies
        """
        input_size = self.data.shape
        data = {}
        outlierIdx = {}
        x = {}
        y = {}
        cls_ = {}
        features = settings['features']
        for feature in features:
            ad = self.data[[feature, goal]].values
            try:
                cls_[feature] = joblib.load('{}/isolation_forest_{}.pkl'.format(self.model_folder, feature))
            except FileNotFoundError:
                logging.error("Could not read isolation forest model for feature {}. Did you forget to train isolation forest".format(feature))
                raise Exception("Missing Model")

            data[feature] = ad
            outlierIdx[feature] = cls_[feature].predict(ad)
            logging.info("found {} outliers for feature {}".format(ad[np.where(outlierIdx[feature] == -1)].shape[0], feature))
            x[feature], y[feature] = meshgrid[feature]

        self.plot_isolation_forest(data, cls_, x, y, outlierIdx, features)

        # Remove all outliers in one pice
        o = []
        for feature in features:
            o = np.unique(np.concatenate((o, np.where(outlierIdx[feature] == -1)[0]), axis=0))
        new_data = self.data.drop(self.data.index[o.astype(int)])

        logging.info("Removed {} elements from dataset.".format(input_size[0]-new_data.shape[0]))

        # Reindex our dataset
        return new_data.reindex()
