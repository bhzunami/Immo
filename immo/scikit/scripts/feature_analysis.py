import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.externals import joblib
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from settings import RNG, IMAGE_FOLDER, MODEL_FOLDER
import pdb

class FeatureAnalysis(object):

    def __init__(self, data):
        self.data = self.cleanup_features(data)

    def cleanup_features(self, data):
        # Merge some Features:
        data['bath'] = np.where((data['badewanne'] == 1) |
                                (data['badezimmer'] == 1) |
                                (data['dusche'] == 1) |
                                (data['lavabo'] == 1), 1, 0)

        data['interior'] = np.where((data['anschluss'] == 1) |
                                    (data['abstellplatz'] == 1) |
                                    (data['cheminée'] == 1) |
                                    (data['eingang'] == 1) |
                                    (data['esszimmer'] == 1) |
                                    (data['gross'] == 1) |
                                    (data['heizung'] == 1) |
                                    (data['lift'] == 1) |
                                    (data['minergie'] == 1) |
                                    (data['schlafzimmer'] == 1) |
                                    (data['wohnzimmer'] == 1) |
                                    (data['rollstuhlgängig'] == 1) |
                                    (data['tv'] == 1) |
                                    (data['küche'] == 1) |
                                    (data['waschküche'] == 1) |
                                    (data['waschmaschine'] == 1) |
                                    (data['wc'] == 1) |
                                    (data['zimmer'] == 1), 1, 0)

        data['exterior'] = np.where((data['aussicht'] == 1) |
                                    (data['balkon'] == 1) |
                                    (data['garten'] == 1) |
                                    (data['garage'] == 1) |
                                    (data['lage'] == 1) |
                                    (data['liegenschaft'] == 1) |
                                    (data['parkplatz'] == 1) |
                                    (data['sitzplatz'] == 1) |
                                    (data['terrasse'] == 1), 1, 0)

        data['neighbourhood'] = np.where((data['autobahnanschluss'] == 1) |
                                         (data['einkaufen'] == 1) |
                                         (data['kinderfreundlich'] == 1) |
                                         (data['kindergarten'] == 1) |
                                         (data['oberstufe'] == 1) |
                                         (data['primarschule'] == 1) |
                                         (data['quartier'] == 1) |
                                         (data['ruhig'] == 1) |
                                         (data['sommer'] == 1) |
                                         (data['verkehr'] == 1) |
                                         (data['zentral'] == 1), 1, 0)

        # Drop the concatinated features
        drop_features = ['badewanne', 'badezimmer', 'dusche', 'lavabo', 'anschluss',
                         'abstellplatz', 'cheminée', 'eingang', 'esszimmer', 'gross',
                         'heizung', 'lift', 'minergie', 'schlafzimmer', 'wohnzimmer',
                         'rollstuhlgängig', 'tv', 'küche', 'waschküche', 'waschmaschine',
                         'wc', 'zimmer', 'aussicht', 'balkon', 'garten', 'garage',
                         'lage', 'liegenschaft', 'parkplatz', 'sitzplatz', 'terrasse',
                         'autobahnanschluss', 'einkaufen', 'kinderfreundlich',
                         'kindergarten', 'oberstufe', 'primarschule', 'quartier',
                         'ruhig', 'sommer', 'verkehr', 'zentral']

        return data.drop(drop_features, axis=1)

    def show_correlation(self, df, goal='price_brutto_m2', save=True):
        if not save:
            corr = joblib.load('{}/correlation.pkl'.format(MODEL_FOLDER))
        else:
            corr = df.corr()
            joblib.dump(corr, '{}/correlation.pkl'.format(MODEL_FOLDER))

        # labels = corr.keys()
        # g = sns.heatmap(corr, vmax=.8, square=True)
        # g.set_yticklabels(labels=labels, rotation=0)
        # g.set_xticklabels(labels=reversed(labels), rotation=90)
        # if save:
        #     plt.savefig("{}/correlation.png".format(IMAGE_FOLDER))
        # plt.close()

        # Get the corelate features which coreolate with price_brutto_m2 at the most
        k = 30 #number of variables for heatmap
        cols = corr.nlargest(k, goal)[goal].index
        cm = np.corrcoef(df[cols].values.T)
        sns.set(font_scale=1.25)
        #hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},
        #                 yticklabels=cols.values, xticklabels=cols.values)

        hm = sns.heatmap(cm, square=True,
                         yticklabels=cols.values, xticklabels=cols.values)
        hm.set_xticklabels(hm.get_xticklabels(), rotation=90)
        hm.set_yticklabels(reversed(hm.get_xticklabels()), rotation=0)
        plt.savefig("{}/corrDetail.png".format(IMAGE_FOLDER))
        #plt.show()
        plt.close()

        if not save:
            corr = joblib.load('{}/correlation_to_goal.pkl'.format(MODEL_FOLDER))
        else:
            corr = df.corr()[goal]
            joblib.dump(corr, '{}/correlation_to_goal.pkl'.format(MODEL_FOLDER))
        # convert series to dataframe so it can be sorted
        corr = pd.DataFrame(corr)
        # correct column label from SalePrice to correlation
        corr.columns = ["Correlation"]
        # sort correlation
        sorted_corr = corr.sort_values(by=['Correlation'], ascending=False)
        print("Correlation to {}".format(goal))
        print(sorted_corr.head(50))


    def tree_classifier(self, X, y, sl, min_percent=0.01, store_model=True, plot_feature=True):
        model = ExtraTreesClassifier(n_estimators=10,
                                     random_state=RNG,
                                     max_depth=500)
        if not store_model:
            model = joblib.load('{}/extraTreesClassifier.pkl'.format(MODEL_FOLDER))
        else:
            model.fit(X, y)
            joblib.dump(model, '{}/{}.pkl'.format(MODEL_FOLDER, 'extraTreesClassifier'))

        importances = model.feature_importances_
        std = np.std([tree.feature_importances_ for tree in model.estimators_],
                     axis=0)
        indices = np.argsort(importances)[::-1]  # most important first

        selected_features = []
        for f in range(X.shape[1]):
            if importances[indices[f]] > min_percent:
                feature_name = sl.get_col_name(indices[f]).split()[0]
                selected_features.append((feature_name, importances[indices[f]]))

        if plot_feature:
            self.plot_feature_importance(importances, indices, min_percent, std, sl)

        return selected_features

    def plot_feature_importance(self, importances, indices, min_percent, std, sl):
        # Plot all features which are over the min_percent importance
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(len(np.where(importances > min_percent)[0])),
                importances[indices[0:len(np.where(importances > min_percent)[0])]],
                color="r",
                yerr=std[indices[0:len(np.where(importances > min_percent)[0])]],
                align="center")

        plt.xticks(range(len(np.where(importances > min_percent)[0])),
                   [sl.get_col_name(e) for e in indices[0:len(np.where(importances > min_percent)[0])]])
        plt.xticks(rotation=90)
        plt.xlim([-1, len(np.where(importances > min_percent)[0])])
        plt.savefig("{}/Feature_importance.png".format(IMAGE_FOLDER))

    def select_KBest(self, X, y, score_func, k=40, name='', dense=False, store=True):
        model = SelectKBest(score_func, k=k)
        if dense:
            model.fit_transform(X.todense(), y)
        else:
            model.fit_transform(X, y)

        if store:
            joblib.dump(model, '{}/{}.pkl'.format(MODEL_FOLDER, name))
        return (model, model.get_support(True))


    def linear_svc(self, X, y, C=0.02, penalty='l1', dual=False, store=True):
        if store:
            lsvc = LinearSVC(C=C, penalty=penalty, dual=dual).fit(X, y)
            joblib.dump(lsvc, '{}/{}.pkl'.format(MODEL_FOLDER, 'linearSVC_1'))
        else:
            lsvc = joblib.load('{}/{}.pkl'.format(MODEL_FOLDER, 'linearSVC_1'))
        model = SelectFromModel(lsvc, prefit=True)
        return model.get_support(True)



def main():
    import json
    data = json.load(open("{}/Features_random_forest.json".format(MODEL_FOLDER)))
    # Sould be added as list
    feature_dict = {}
    value_dict = {}
    for key, value in data.items():
        if key.startswith("feature"):
            feature_dict[int(key[7:])] = value

    features = []
    for key, value in sorted(feature_dict.items()):
        features.append(value)

    for key, value in data.items():
        if key.startswith("m"):
            value_dict[int(key[1:])] = value

    values = []
    for key, value in sorted(value_dict.items()):
        values.append(value)

    df = pd.DataFrame({'features': pd.Series(features), 'value': pd.Series(values)})
    plt.plot(range(1, 155), values)
    plt.savefig("{}/Important_features_random_forest".format(IMAGE_FOLDER))
    for i, feature in enumerate(features):
        print("{}. {}".format(i, feature))
    
if __name__ == "__main__":
    main()