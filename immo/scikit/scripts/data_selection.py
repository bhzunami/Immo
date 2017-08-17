import os
from models import Advertisement, Municipality, ObjectType
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LassoLarsCV, Ridge, RidgeCV, LassoCV, Lasso, LinearRegression, LogisticRegression

from sklearn.naive_bayes import MultinomialNB
from sqlalchemy.orm import sessionmaker, defer, load_only, aliased
from sqlalchemy.sql import select
from sqlalchemy.sql.expression import join
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier, SGDRegressor
import re
import ast
import json
import nltk
import sys
from nltk.corpus import stopwords # Import the stop word list
from nltk.stem import SnowballStemmer
from jupyter import statistics, plot


OBJECT_TYPES = {'description': str, 'desc': str, 'groupint': str, 'characteristics': object}
import pdb


def select_from_database(url=os.environ.get('DATABASE_URL', None)):
    engine = create_engine(os.environ.get('DATABASE_URL', None))
    Session = sessionmaker(bind=engine)
    session = Session()
    ot = aliased(ObjectType)
    ad = aliased(Advertisement)
    s = select([ad.price_brutto.label('price'),
                ad.description.label('description'),
                ad.living_area.label('living_area'),
                ad.characteristics.label('characteristics'),
                # ad.additional_data.label('additional_data'),  # Not much important stuff some distances
                ot.grouping.label('grouping'),
                #Municipality.canton_id.label('canton')
                # ended.quarter.label('equarter'),
               ]
              ).select_from(
                  join(ad, Municipality, isouter=False)
                  .join(ot, ad.object_types_id==ot.id, isouter=False)
                  #.join(ended, Ad.ended_id == ended.id, isouter=False)
              ).where(ad.build_year >= 1000)
    return pd.read_sql_query(s, session.bind)

def select_from_file(file):
    return pd.read_csv(file, index_col=0, engine='c', dtype=OBJECT_TYPES)

def detect_language(text):
    """ detect the language by the text where 
    the most stopwords for a language are found
    """
    languages_ratios = {}
    tokens = nltk.wordpunct_tokenize(text)
    words_set = set([word.lower() for word in tokens])
    for language in ['german', 'french', 'italian']:
        stopwords_set = set(stopwords.words(language))
        languages_ratios[language] = len(words_set.intersection(stopwords_set)) # language "score"

    return max(languages_ratios, key=languages_ratios.get)


def stemm_words(row):
    language = detect_language(row.description)
    #letters_only = re.sub("[^a-z0-9üäöèéàêâ]", " ", str(row.desc.lower()))
    letters_only = row.description.split()
    stops = set(stopwords.words(language))
    meaningful_words = [w for w in letters_only if not w in stops]
    stemmer = SnowballStemmer(language)
    stemmed_words = [stemmer.stem(w) for w in meaningful_words]
    return " ".join(stemmed_words)

def category_ads(low, high):
    def return_func(row):
        if row.price > high:
            return 'high'

        if row.price <= high and row.price > low:
            return 'normal'

        return 'budget'
    return return_func

    # def to_words(self, desc):
    #     letters_only = re.sub("[^a-zA-Z]", " ", str(desc))
    #     words = letters_only.lower().split()
    #     stops = set(stopwords.words("german"))
    #     meaningful_words = [w for w in words if not w in stops]
    #     return( " ".join( meaningful_words ))


def ape(y_true, y_pred):
    return np.abs(y_true - y_pred) / y_true

def mape(y_true, y_pred):
    return ape(y_true, y_pred).mean()

def mdape(y_true, y_pred):
    return np.median(ape(y_true, y_pred))

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

    num_elements = len(pred)
    apes = ape(y, pred)
    for i in np.arange(5, 100, 5):
        print("I {}: {}".format(i, len(np.where(apes < i/100)[0])/num_elements))


if __name__ == "__main__":
    # From database
    #advertisements = select_from_database(url=os.environ.get('DATABASE_URL', None))
    #advertisements.to_csv('description_analyse.csv', header=True, encoding='utf-8')
    # From file
    #advertisements = select_from_file('all.csv')
    #advertisements.drop(['noise_level', 'm_noise_level', 'floor', 'last_renovation_year'], axis=1, inplace=True)

    #advertisements = advertisements.dropna()  # Remove empty entries
    #advertisements.reindex()

    # Convert decription to unicode, otherwise NaN is inserted
    #advertisements['description'] = advertisements['description'].astype('U')

    #advertisements['desc'] = advertisements.apply(stemm_words, axis=1)
    #advertisements.to_csv('category_ads.csv', header=True, encoding='utf-8')
    # From file
    advertisements = select_from_file('desc_analyse.csv')
    advertisements['characteristics'] = advertisements['characteristics'].apply(ast.literal_eval)
    series = []
    bath_keys = ['Badezimmer', 'Badewannen', 'WCs', 'Duschen', 'Toiletten']
    for index, ad in advertisements.iterrows():
        if type(ad['characteristics']) == list:
            continue
        bath = {}
        for key, value in ad['characteristics'].items():
            if key in bath_keys:
                bath[key] = int(value)
                bath['living_area'] = ad['living_area']
                bath['desc'] = ad['desc']
                #bath['num_room'] = ad['num_room']
                series.append(bath)
   
    df = pd.DataFrame(series)

    df['bath_max'] = df[['Badezimmer', 'Badewannen', 'WCs', 'Duschen', 'Toiletten']].apply(lambda x: np.mean(x), axis=1)
    sns.lmplot('bath_max', 'living_area', data=df)
    plt.show()

    X_desc = df['desc'].astype('U').values
    y = df['bath_max'].values
    count_vect = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_features=5000)
    X_train_counts = count_vect.fit_transform(X_desc)

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    temp_df = pd.DataFrame(X_train_counts.todense())
    temp_df['living_area'] = df['living_area'].values
    
    X = temp_df.values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    alpha = 0.006
    print("Best alpha: {}".format(alpha))

    lassoreg = Lasso(alpha=alpha, normalize=True, max_iter=1e5)
    lassoreg.fit(X_train, y_train)
    y_pred = lassoreg.predict(X_test)
    statistics(y_test, y_pred)
    plot(y_test, y_pred, show=True, plot_name="lasso")

    #  plt.figure(figsize=(10, 8))
    # g = sns.barplot(y=list(chars.keys()), x=list(chars.values()))
    # g.tick_params(axis='y', labelsize=5)

    sys.exit(0)
    for group in [('haus',(700000, 1500000)), ('wohnung', (500000, 1000000))]:

        ads = pd.DataFrame(advertisements[advertisements['grouping'] == group[0]])

        print("Read {} {} data".format(len(ads), group[0]))
        # sns.distplot(ads['price'], kde=True, bins=300, hist_kws={'alpha': 0.6})
        # plt.show()
        # print(ads.price.describe())
        # Make categories low, high
        ads['category'] = ads.apply(category_ads(group[1][0], group[1][1]), axis=1)
        
        X = ads['desc'].astype('U').values
        y = ads['category'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
        count_vect = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_features=5000)
        X_train_counts = count_vect.fit_transform(X_train)

        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

        #clf = RandomForestClassifier(n_estimators = 100)
        #clf = clf.fit(X_train_tfidf, y_train)

        for alpha in [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 3e-4, 1e-3]:  # 3e-3, 0.01, 0.03 ,0.1, 0.3, 1, 3, 10, 30
            clf = SGDClassifier(loss="log", penalty='l2', alpha=alpha, n_iter=100, random_state=42).fit(X_train_tfidf, y_train)
            #  clf = MultinomialNB().fit(X_train_tfidf, y_train)

            X_new_counts = count_vect.transform(X_test)
            X_new_tfidf = tfidf_transformer.transform(X_new_counts)
            predicted = clf.predict(X_new_tfidf)

            # for doc, category in zip(y_test, predicted):
            #     print('%r => %s' % (doc, category))
            print("RESULTS for {} with alpha: {}".format(group, alpha))
            print(np.mean(predicted == y_test))
            print(metrics.classification_report(y_test, predicted))

    # words = []
    # for i in range(0, len(ds.ads) ):
    #     # If the index is evenly divisible by 1000, print a message
    #     if (i+1)%1000 == 0:
    #         print("Review {} of {}".format(i+1, len(ds.ads)))
    #     words.append(ds.to_words(ds.ads['description'][i]))

    # ds.ads['desc'] = words
    # ds.ads.to_csv('description_analyse.csv', header=True, encoding='utf-8')