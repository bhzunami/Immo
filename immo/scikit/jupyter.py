# Import only
import pandas as pd
import pdb
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import seaborn as sns
from a_detection import AnomalyDetection
from feature_analysis import FeatureAnalysis
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import chi2, f_regression
from sklearn.linear_model import LassoLarsCV, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn import metrics


from settings import OBJECT_TYPES, MODEL_FOLDER, IMAGE_FOLDER


def gen_subplots(fig, x, y):
    for i in range(x*y):
        ax = fig.add_subplot(x, y, i+1)
        plt.grid()
        yield ax

class SupervisedLearning(object):

    def __init__(self, dataset, cleanup=True, drop_cols=['id', 'long', 'lat', 'price_brutto']):
        if cleanup:
            self.data = self.prepare_dataset(dataset, drop_cols)
        else:
            self.data = dataset
        self.dv = DictVectorizer(sparse=True)

    def prepare_dataset(self, dataset, drop_cols):
        """ remove unwanted columns and clean up NaN values or insert
            data in NaN values
        """
        # Insert missing values
        dataset.floor.fillna(0, inplace=True)  # Add floor 0 if not present
        dataset.num_rooms.fillna(dataset.num_rooms.median(), inplace=True)

        # Drop colums we are not interessted
        dataset = dataset.drop(drop_cols, axis=1)
        return dataset.dropna()   # Drop NaN Values


    def plot_info(self, ads, save=False):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
        # We have to convert conaton_id and floor to integer to get the right order.
        data = ads.copy()
        data['canton_id'] = ads['canton_id'].astype(int)
        data['floor'] = ads['floor'].astype(int)

        # Canton_id
        sns.barplot(x="canton_id", y="price_brutto_m2", data=data, ax=ax1)

        # Num rooms
        sns.barplot(x="num_rooms", y="price_brutto_m2", data=data, ax=ax2)

        # Floor
        sns.barplot(x="floor", y="price_brutto_m2", data=data, ax=ax3)

        # Object type
        g = sns.barplot(x="otype", y="price_brutto_m2", data=data, ax=ax4)
        g.set_xticklabels(g.get_xticklabels(), rotation=90)
        
        if save:
            plt.savefig("{}/Analytical_data.png".format(IMAGE_FOLDER))
        plt.close()

        sns.lmplot(x="avg_room_area", y="price_brutto_m2", data=data)
        if save:
            plt.savefig("{}/avg_room_area.png".format(IMAGE_FOLDER))            
        plt.close()

        sns.lmplot(x="build_year", y="price_brutto_m2", data=data)
        if save:
            plt.savefig("{}/build_year.png".format(IMAGE_FOLDER))
        plt.close()

        #scatterplot
        sns.set()
        cols = ['price_brutto_m2', 'build_year', 'num_rooms', 'avg_room_area', 'last_construction']
        sns.pairplot(data[cols], size = 2.5)
        if save:
            plt.savefig("{}/pairplot.png".format(IMAGE_FOLDER))            
        plt.close()

        # Price distplot
        sns.distplot(data['price_brutto_m2'], kde=True, bins=100, hist_kws={'alpha': 0.6})
        if save:
            plt.savefig("{}/Dist_Price_plot.png".format(IMAGE_FOLDER))
        plt.close()


    def one_hot(self, X, y):
        # Prepare transformation from pandas dataframe to matrix
        # One Hot Encoding for string
        # Otype municiplaiy ogroup
        # - - - - - - - - - - - - - - - -
        print("Size of data {}".format(X.shape))
        self.dv.fit(X.T.to_dict().values())  # Learn a list of feature name Important Price is present here
        print("Len of features after one hot: {}".format(len(self.dv.feature_names_)))

        # Transform feature -> value dicts to array or sparse matrix
        X = self.dv.transform(X.T.to_dict().values())
        y = y.values
        return X, y

    def generate_matrix(self, df, goal):
        X = df.drop([goal], axis=1)
        y = df[goal].astype(int)
        return X, y
        
    def generate_test_train(self, X, y):
        return train_test_split(X, y, test_size=0.5)

    def get_col_name(self, idx, show_id=False):
        """ return the name of the feature from the dv
        """
        if show_id:
            return "{} ({})".format(self.dv.get_feature_names()[idx], idx)
        return "{}".format(self.dv.get_feature_names()[idx])

    def convert_to_df(self, X):   
        # Drop some features
        # ads_with_out_anomaly
        return pd.DataFrame(X.todense(), columns=self.dv.feature_names_)

    def fit(self, X_train, y_train, X_test, y_test, models, store=True):
        for name, model in models.items():
            print("Fit model: {}".format(name))
            # if name == 'lasso':
            #     pdb.set_trace()
            #     model.fit(X_train.todense(), y_train)
            # else:
            if store:
                model.fit(X_train, y_train)
                joblib.dump(model, '{}/{}.pkl'.format(MODEL_FOLDER, name))
            else:
                model = joblib.load('{}/{}.pkl'.format(MODEL_FOLDER, name))

            print("MODEL: {} scores:".format(name))
            print(model.score(X_train, y_train), model.score(X_test, y_test))


def main():
    ads = pd.read_csv('all.csv', index_col=0, engine='c', dtype=OBJECT_TYPES)
    sl = SupervisedLearning(ads)
    sl.plot_info(ads, save=True)

    # Anomaly detection - Remove wired ads
    # - - - - - - - - - - - - - - - - - - 
    anomaly_detection = AnomalyDetection(sl.data)

    # Data for plot of anomaly detection 'floor', 
    features = [('floor', 0.03),
                ('build_year', 0.055),
                ('num_rooms', 0.03),
                ('avg_room_area', 0.03),
                ('last_construction', 0.045)]

    meshgrid = {
        'floor': np.meshgrid(np.linspace(-1, 40, 400), np.linspace(0, 60000, 1000)),
        'build_year': np.meshgrid(np.linspace(0, 2025, 400), np.linspace(0, 60000, 1000)),
        'num_rooms': np.meshgrid(np.linspace(-1, 100, 400), np.linspace(0, 60000, 1000)),
        'avg_room_area': np.meshgrid(np.linspace(0, 500, 400), np.linspace(0, 60000, 1000)),
        'last_construction': np.meshgrid(np.linspace(0, 800, 400), np.linspace(0, 60000, 1000))
    }

    # for feature in features:
    #     print("Analyse feature {}".format(feature))
    #     # Remove wired ads
    #     # fig, *ax = plt.subplots(11)
    #     # ax = ax[0]
    #     # ax[0].hist(np.sort(sl.data.avg_room_area))
    #     data_frames = []
    #     std = []
    #     std_percent = []
    #     data_frames.append(pd.DataFrame({feature: sl.data[feature].astype(int), 'percent': 0}))
    #     std.append(np.std(sl.data[feature].astype(int)))
    #     std_percent.append(0)

    data = anomaly_detection.isolation_forest(features, meshgrid, 'price_brutto_m2', show_plot=True)

    fig = plt.figure(figsize=(10, 15))  # länge x, länge y
    subplots = gen_subplots(fig, 10, 2)  # num y num x
    for feature in features:
        name = feature[0]
        ax = next(subplots)
        sns.distplot(sl.data[name].astype(int), hist=False, kde_kws={'cumulative': True }, ax=ax)
        ax = next(subplots)
        sns.distplot(sl.data[name].astype(int), ax=ax)

        ax = next(subplots)
        sns.distplot(data[name].astype(int), hist=False, kde_kws={'cumulative': True }, ax=ax)
        ax = next(subplots)
        sns.distplot(data[name].astype(int), ax=ax)

    plt.savefig("{}/anomaly_detection_cleanup.png".format(IMAGE_FOLDER))
    plt.close()
    # for i in np.arange(0.5, 11, 0.5):
    #     data_frames.append(pd.DataFrame({feature: data[feature].astype(int), 'percent': i}))
    #     std.append(np.std(data[feature].astype(int)))
    #     std_percent.append((1-std[-1]/std[0])*100)

    #     # sl.data has the new values
    #     p_df = pd.concat(data_frames)
    #     g = sns.FacetGrid(p_df, row="percent")
    #     g.map(sns.distplot, feature)
    #     plt.savefig('{}/{}_dist.png'.format(IMAGE_FOLDER, feature))
    #     plt.close()

        # sns.distplot(sl.data[feature].astype(int), hist=False, kde_kws={'cumulative': True })
        # plt.savefig('{}/{}_cum.png'.format(IMAGE_FOLDER, feature))
        # plt.close()

        # fig, ax1 = plt.subplots()
        # #ax1.plot(std)
        # #ax1.set_ylabel('STD deviation')
        # ax1.set_title('{}'.format(feature))
        # ax1.plot(std_percent, list(np.arange(0, 11, 0.5)), c='r')
        # ax1.set_ylabel('Reduction of STD deviation in %')
        # ax1.set_xlabel('% removed of {}'.format(feature))
        # plt.savefig('{}/{}_std.png'.format(IMAGE_FOLDER, feature))
        # plt.close()

    # Analyse features
    #- - - - - - - - -
    selected_features = set(['price_brutto_m2']) # Add y value
    feature_analysis = FeatureAnalysis(data)

    # set data
    sl.data = data

    #  Make one hot coding
    X, y = sl.generate_matrix(feature_analysis.data, 'price_brutto_m2')
    X, y = sl.one_hot(X, y)
    X_train, X_test, y_train, y_test = sl.generate_test_train(X, y)

    df = sl.convert_to_df(X)
    df['price_brutto_m2'] = y
    feature_analysis.show_correlation(df, goal='price_brutto_m2', save=True)

    # Print important features
    # min_percent how many percent a feature should support
    features = feature_analysis.tree_classifier(X_train, y_train,sl,
                                                min_percent=0.01, store_model=True,
                                                plot_feature=True)
    for f in features:
        selected_features.add(f[0])
        print("{}".format(f))

    # the smaller C the fewer features selected
    features = feature_analysis.linear_svc(X_train, y_train, C=0.1, penalty='l1', dual=False, store=True)
    print("Most {} important features in L1".format(len(features)))
    for feature in features:
        selected_features.add(sl.dv.get_feature_names()[feature])
        print("Feature: {} ".format(sl.dv.get_feature_names()[feature]))

    #model, features_chi2 = feature_analysis.select_KBest(sl, chi2, 40, name='chi2')
    #for feature in features_chi2:
    #    print("Feature: {} {}".format(sl.dv.get_feature_names()[feature], model.scores_[feature]))

    #model, features_chi2 = feature_analysis.select_KBest(sl, f_regression, 40, name='f_regression', True)
    #for feature in features_chi2:
    #    print("Feature: {} {}".format(sl.dv.get_feature_names()[feature], model.scores_[feature]))

    df = df[list(selected_features)]
    df.to_csv('clean_all.csv', header=True, encoding='utf-8')
    return
    X, y = sl.generate_matrix(df, 'price_brutto_m2')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    models = {'random forest': RandomForestRegressor(),
              'ridge': Ridge(),
              'lasso': LassoLarsCV(),
              #'linearSVC_fit': LinearSVC(),
              #'svc': SVC(),
              'logistic': LogisticRegression(),
              'gauss': GaussianNB()}
    pdb.set_trace()
    sl.fit(X_train, y_train, X_test, y_test, models)

def ape(y_true, y_pred):
    return np.abs(y_true - y_pred) / y_true

def mape(y_true, y_pred): 
    return ape(y_true, y_pred).mean() 

def mdape(y_true, y_pred):
    return np.median(ape(y_true, y_pred))

def show_results(y_true, y_pred):
    num_elements = len(y_pred)
    apes = ape(y_true, y_pred)
    for i in np.arange(5, 100, 5):
        print("I {}: {}".format(i, len(np.where(apes < i/100)[0])/num_elements))


def calculate():
    ads = pd.read_csv('clean_all.csv', index_col=0, engine='c', dtype=OBJECT_TYPES)
    sl = SupervisedLearning(ads, cleanup=False)
    X, y = sl.generate_matrix(ads, 'price_brutto_m2')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    models = {'random forest': RandomForestRegressor(),
              'ridge': Ridge(),
              'lasso': LassoLarsCV()}
              #'linearSVC_fit': LinearSVC(),
              #'svc': SVC(),
              #'logistic': LogisticRegression(),
              #'gauss': GaussianNB()}
    sl.fit(X_train, y_train, X_test, y_test, models, True)

    for name, model in models.items():
        #model = joblib.load('{}/{}.pkl'.format(MODEL_FOLDER, name))
        y_predicted = model.predict(X_test)
        print("{}: MAPE: {:.3%}, MDAPE: {:.3%}".format(name, mape(y_test, y_predicted), mdape(y_test, y_predicted)))

scorer = make_scorer(mean_squared_error, greater_is_better = False)

def rmse_cv_train(model, X_train, y_train):
    rmse= np.sqrt(-cross_val_score(model, X_train, y_train, scoring = scorer, cv = 10))
    return(rmse)

def rmse_cv_test(model, X_test, y_test):
    rmse= np.sqrt(-cross_val_score(model, X_test, y_test, scoring = scorer, cv = 10))
    return(rmse)

def calc2():
    ads = pd.read_csv('clean_all.csv', index_col=0, engine='c', dtype=OBJECT_TYPES)
    sl = SupervisedLearning(ads, cleanup=False)
    X, y = sl.generate_matrix(ads, 'price_brutto_m2')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
    #ridge = RidgeCV(alphas = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30])
    #ridge.fit(X_train, y_train)
    #alpha = ridge.alpha_
    #print("Best alpha :", alpha)

    #print("Try again for more precision with alphas centered around " + str(alpha))
    # ridge = RidgeCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, 
    #                           alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,
    #                           alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4], 
    #                 cv = 10)

    ridge = RidgeCV(alphas=[0.54])
    ridge.fit(X_train, y_train)
    alpha = ridge.alpha_
    print("Best alpha :", alpha)

    #print("Ridge RMSE on Training set :", rmse_cv_train(ridge, X_train, y_train).mean())
    #print("Ridge RMSE on Test set :", rmse_cv_test(ridge, X_test, y_test).mean())
    y_test_rdg = ridge.predict(X_test)
    print("MAPE: {:.3%}, MDAPE: {:.3%}".format(mape(y_test, y_test_rdg), mdape(y_test, y_test_rdg)))

    fig = plt.figure(figsize=(10, 6))
    subplots = gen_subplots(fig, 1, 1)
    
    ax = next(subplots)
    ax.set_xlabel('APE')
    my_ape = ape(y_test, y_test_rdg)
    ax.hist(my_ape, 100)
    m = ape(y_test, y_test_rdg).mean()
    ax.plot((m, m), (0,400), 'r-')
    
    mm = np.median(ape(y_test, y_test_rdg))
    ax.plot((mm, mm), (0,400), 'y-')
    
    ax.set_xticklabels(['{:.2%}'.format(x) for x in ax.get_xticks()])

    plt.show()
    values = y_test - y_test_rdg
    print(values)

    
def plot(y_test, y_pred):
     # sort both array by y_test
    y_test, y_pred = zip(*sorted(zip([x for x in y_test], [x for x in y_pred])))
    y_test = np.asarray(y_test)
    y_pred = np.asarray(y_pred)
    
    fig = plt.figure(figsize=(10, 10))
    subplots = gen_subplots(fig, 3, 1)

    ax = next(subplots)    
    ax.set_xlabel('Actual vs. Predicted values')
    ax.plot(y_test, 'bo', markersize=1, label="Actual")
    ax.plot(y_pred, 'ro', markersize=1, label="Predicted")
    
    ax = next(subplots)
    ax.set_xlabel('Residuals')
    ax.plot(y_test - y_pred, 'bo', markersize=1)
    ax.plot((0,len(y_test)), (0,0), 'r-')
    

    plt.show()
    fig = plt.figure(figsize=(5, 5))
    subplots = gen_subplots(fig, 1, 1)
    ax = next(subplots)
    ax.set_xlabel('APE')
    my_ape = ape(y_test, y_pred)
    ax.hist(my_ape, 100)
    m = ape(y_test, y_pred).mean()
    ax.plot((m, m), (0,400), 'r-')
    
    mm = np.median(ape(y_test, y_pred))
    ax.plot((mm, mm), (0,400), 'y-')
    ax.set_xticklabels(['{:.2%}'.format(x) for x in ax.get_xticks()])

    plt.show()


def calc3():
    ads = pd.read_csv('clean_all.csv', index_col=0, engine='c', dtype=OBJECT_TYPES)

    ads = ads[(ads['price_brutto_m2'] < 14000) & (ads['price_brutto_m2'] > 2500)]
    sl = SupervisedLearning(ads, cleanup=False)
    X, y = sl.generate_matrix(ads, 'price_brutto_m2')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    y_predicted = model.predict(X_test)
    statistics(y_test, y_predicted)
    plot(y_test, y_predicted)
    show_results(y_test, y_predicted)


def feature():
    ads = pd.read_csv('clean_all.csv', index_col=0, engine='c', dtype=OBJECT_TYPES)
    best_mape = {'feature0': 'NONE', 'm0': 1000}
    keys = list(ads.keys())
    keys.remove('price_brutto_m2')
    choosen_features = set(['price_brutto_m2'])
    for i in range(0, len(keys)):
        print("ROUND: {} with features: {}".format(i, choosen_features))
        best_mape['feature{}'.format(i)] =  'NONE'
        best_mape['m{}'.format(i)] = 1000
        for feature in keys:
            features = list(choosen_features) + [feature]
            ad = ads[features]
            sl = SupervisedLearning(ad, cleanup=False)
            X, y = sl.generate_matrix(ad, 'price_brutto_m2')
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
            model = RandomForestRegressor()
            model.fit(X_train, y_train)
            y_predicted = model.predict(X_test)
            m = mape(y_test, y_predicted)
            if best_mape['m{}'.format(i)] > m:
                print("Found new best: {} with feature {}".format(m, feature))
                best_mape['m{}'.format(i)] = m
                best_mape['feature{}'.format(i)] = feature
        print("BEST Feature in i{} : {} wiht MAPE: {}".format(i,
                                                              best_mape['feature{}'.format(i)],
                                                              best_mape['m{}'.format(i)]))
        choosen_features.add(best_mape['feature{}'.format(i)])
        keys.remove(best_mape['feature{}'.format(i)])
    print(best_mape)


    # Select 2 best
    best2_mape = {'feature': 'NONE', 'm': 1000}
    for feature in ads.keys():
        if feature == 'price_brutto_m2' or feature == best_mape['feature']:
            continue
        print("Run for feature {}".format(feature))
        ad = ads[[feature, 'price_brutto_m2', best_mape['feature']]]
        sl = SupervisedLearning(ad, cleanup=False)
        X, y = sl.generate_matrix(ad, 'price_brutto_m2')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        y_predicted = model.predict(X_test)
        m = mape(y_test, y_predicted)
        if best2_mape['m'] > m:
            print("Found new best: {} with feature {}".format(m, feature))
            best2_mape['m'] = m
            best2_mape['feature'] = feature
    print("2nd BEST Feature: {} wiht MAPE: {}".format(best2_mape['feature'], best2_mape['m']))



def statistics(y, pred):
    diff = np.fabs(y - pred)
    print("             R²-Score: {:10n}".format(metrics.r2_score(y, pred)))
    print("                 MAPE: {:.3%}".format(mape(y, pred)))
    print("                MdAPE: {:.3%}".format(mdape(y, pred)))
    print("            Min error: {:10n}".format(np.amin(diff)))
    print("            Max error: {:10n}".format(np.amax(diff)))
    print("  Mean absolute error: {:10n}".format(metrics.mean_absolute_error(y, pred)))
    print("Median absolute error: {:10n}".format(metrics.median_absolute_error(y, pred)))
    print("   Mean squared error: {:10n}".format(metrics.mean_squared_error(y, pred)))

def calc4():
    from sklearn.datasets import make_gaussian_quantiles
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.tree import DecisionTreeClassifier
    ads = pd.read_csv('clean_all.csv', index_col=0, engine='c', dtype=OBJECT_TYPES)
    sl = SupervisedLearning(ads, cleanup=False)
    X, y = sl.generate_matrix(ads, 'price_brutto_m2')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    bdt_real = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=2),
    n_estimators=600,
    learning_rate=1)

    bdt_discrete = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=2),
        n_estimators=600,
        learning_rate=1.5,
        algorithm="SAMME")

    bdt_real.fit(X_train, y_train)
    #bdt_discrete.fit(X_train, y_train)

    y_real_predict = bdt_real.predict(X_test)
    # y_discrete_predict = bdt_discrete.predict(X_test)

    pdb.set_trace()
    print("REAL MAPE: {:.3%}, MDAPE: {:.3%}".format(mape(y_test, y_real_predict), mdape(y_test, y_real_predict)))
    plot(y_test, y_real_predict)
    
    #print("DISCRETE: MAPE: {:.3%}, MDAPE: {:.3%}".format(mape(y_test, y_discrete_predict), mdape(y_test, #y_discrete_predict)))
    #plot(y_test, y_discrete_predict)
    
    

if __name__ == "__main__":
    #main()
    #calculate()
    #calc3()
    feature()
    