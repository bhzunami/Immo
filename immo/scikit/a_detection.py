from sklearn.ensemble import IsolationForest
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from settings import RNG, IMAGE_FOLDER, MODEL_FOLDER
import pdb

class AnomalyDetection(object):

    def __init__(self, data):
        self.data = data

    def plot_isolation_forest(self, data, cls, x, y, outlierIdx, elements):
        fig = plt.figure(figsize=(40, 30))
        for i, feature in enumerate(elements):
            name = feature[0] # feature is a tupple (name, C)
            Z = cls[name].decision_function(np.c_[x[name].ravel(), y[name].ravel()])
            #Z = cls.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(x[name].shape)

            ax = fig.add_subplot(3, 2, i+1)
            ax.set_ylabel('Price')
            ax.set_title('Price - {}'.format(name))

            ax.contourf(x[name],
                        y[name], Z, 
                        cmap=plt.cm.Blues_r)

            ax.scatter(data[name][np.where(outlierIdx[name] == -1)][:, 0].tolist(),
                       data[name][np.where(outlierIdx[name] == -1)][:, 1].tolist(),
                       c='red', s=2)

            ax.scatter(data[name][np.where(outlierIdx[name] == 1)][:, 0].tolist(),
                       data[name][np.where(outlierIdx[name] == 1)][:, 1].tolist(),
                       c='green', s=2)

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig("{}/IsolationForest.png".format(IMAGE_FOLDER))
        plt.close()

    def isolation_forest(self,
                         features,
                         meshgrid,
                         goal='price_brutto_m2',
                         show_plot=False,
                         load=False):
        """ get a dataframe and features to check for anomaly detection
        returns a new dataframe with removed anomalies
        """
        input_size = self.data.shape
        data = {}
        outlierIdx = {}
        x = {}
        y = {}
        cls = {}
        for feature in features:
            name = feature[0]
            c = feature[1]
            ad = self.data[[name, goal]].values
            if load:
                cls[name] = joblib.load('{}/isolation_forest_{}.pkl'.format(MODEL_FOLDER, name))
            else:
                cls[name] = IsolationForest(max_samples=0.7, contamination=c, random_state=RNG)
                cls[name].fit(ad)

                # Store cls
                joblib.dump(cls[name], '{}/isolation_forest_{}.pkl'.format(MODEL_FOLDER, name))

            data[name] = ad
            outlierIdx[name] = cls[name].predict(ad)
            print("found {} outliers".format(ad[np.where(outlierIdx[name] == -1)].shape[0]))
            x[name], y[name] = meshgrid[name]

        if show_plot:
            self.plot_isolation_forest(data, cls, x, y, outlierIdx, features)

        # Remove all outliers in one pice
        o = []
        for feature in features:
            name = feature[0]
            o = np.unique(np.concatenate((o, np.where(outlierIdx[name] == -1)[0]), axis=0))
        new_data = self.data.drop(self.data.index[o.astype(int)])

        print("Removed {} elements from dataset.".format(input_size[0]-new_data.shape[0]))

        # Reindex our dataset
        return new_data.reindex()
