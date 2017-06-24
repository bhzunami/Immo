import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EmpiricalCovariance, MinCovDet
import pandas as pd
from sklearn.model_selection import train_test_split
import pdb
from sklearn.externals import joblib
from sklearn.feature_extraction import DictVectorizer

def one_hot_dataframe(data, cols, replace=False):
    """ Takes a dataframe and a list of columns that need to be encoded.
        Returns a 3-tuple comprising the data, the vectorized data,
        and the fitted vectorizor.
    """
    vec = DictVectorizer()
    mkdict = lambda row: dict((col, row[col]) for col in cols)
    vecData = pd.DataFrame(vec.fit_transform(data[cols].apply(mkdict, axis=1)).toarray())
    vecData.columns = vec.get_feature_names()
    vecData.index = data.index
    if replace is True:
        data = data.drop(cols, axis=1)
        data = data.join(vecData)
    return (data, vecData, vec)


def get_outliners(dataset, outliers_fraction=0.25):
    clf = svm.OneClassSVM(nu=0.95 * outliers_fraction + 0.05, kernel="rbf", gamma=0.1)
    clf.fit(dataset)
    result = clf.predict(dataset)
    return result

def main():
    ads = pd.read_csv('all.csv', engine='c')  # Use the c engine to speed up performance
    # Remove all entries with NaN / None
    ads = ads.dropna()
    print("After remove all NaN the size of our dataset is {}".format(ads.shape))

    # One-class SVM with non-linear kernel (RBF)
    # http://scikit-learn.org/stable/auto_examples/svm/plot_oneclass.html#sphx-glr-auto-examples-svm-plot-oneclass-py
    if not os.path.exists('OneClassSVM.pkl'):
        clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
        print("Start Fit")
        clf.fit(ads)
        print("STore fit")
        joblib.dump(clf, 'OneClassSVM.pkl')
    else:
        clf = joblib.load('OneClassSVM.pkl')

    print("Fit is done start predict")
    # prediction = clf.predict(ads)
    # training_dataset = ads[prediction == 1]
    # removed_dataset = ads[prediction == -1]

    rng = np.random.RandomState(42)
    if not os.path.exists('IsolationForest2.pkl'):
        clf = IsolationForest(max_samples=10000, random_state=rng)
        clf.fit(ads)
        joblib.dump(clf, 'IsolationForest.pkl')
    else:
        clf = joblib.load('IsolationForest.pkl')
    print("predict randomForest")
    prediction = clf.predict(ads)

    training_dataset = ads[prediction == 1]
    removed_dataset = ads[prediction == -1]

    brutto = training_dataset['price_brutto']
    area = training_dataset['living_area']

    xx, yy = np.meshgrid(np.linspace(min(brutto), max(brutto), 100), np.linspace(min(area), max(area), 100))
    Z = clf.decision_function(ads)
    Z = Z.reshape(xx.shape)

    plt.title("IsolationForest")
    plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)

    plt.scatter(training_dataset['price_brutto'], training_dataset['living_area'], c='blue', s=2)
    plt.scatter(removed_dataset['price_brutto'], removed_dataset['living_area'], c='red', s=2)


    pdb.set_trace()

    return

    y = training_dataset['price_brutto'].values
    # pdb.set_trace()
    X = training_dataset.drop(['price_brutto', 'id'], axis=1)
    keys = list(X.keys())[1:]
    X = X.values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)

    pdb.set_trace()
    
if __name__ == "__main__":
    main()