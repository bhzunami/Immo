from sklearn.ensemble import IsolationForest
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import pdb
RNG = np.random.RandomState(42)
import logging

class AnomalyDetection(object):

    def __init__(self, data, image_folder, model_folder):
        self.data = data
        self.image_folder = image_folder
        self.model_folder = model_folder

    def plot_isolation_forest(self, data, cls_, x, y, outlierIdx, elements):
        for i, feature in enumerate(elements):
            fig, ax = plt.subplots()
            Z = cls_[feature].decision_function(np.c_[x[feature].ravel(), y[feature].ravel()])
            Z = Z.reshape(x[feature].shape)
            ax.set_title('Price - {}'.format(feature.replace("_", " ")))
            ax.set_ylabel('Price')
            ax.set_xlabel('{}'.format(feature.replace("_", " ")))

            ax.contourf(x[feature],
                        y[feature], Z,
                        cmap=plt.cm.Blues_r)

            a1 = ax.scatter(data[feature][np.where(outlierIdx[feature] == -1)][:, 0].tolist(),
                            data[feature][np.where(outlierIdx[feature] == -1)][:, 1].tolist(),
                            c='red', s=2)

            a2 = ax.scatter(data[feature][np.where(outlierIdx[feature] == 1)][:, 0].tolist(),
                            data[feature][np.where(outlierIdx[feature] == 1)][:, 1].tolist(),
                            c='green', s=2)

            ax.set_xlim((min(x[feature][0]), max(x[feature][0])))
            ax.set_ylim(0, 35e6)
            ax.legend([a1, a2],
                      ["Outlier", "Accepted values"],
                      loc="upper left")

            #plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            plt.savefig("{}/{}_IsolationForest.png".format(self.image_folder, feature))
            plt.close()

    def isolation_forest(self,
                         settings,
                         meshgrid,
                         goal='price_brutto_m2',
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
