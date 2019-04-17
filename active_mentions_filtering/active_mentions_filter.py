import os

from nltk import TweetTokenizer, PorterStemmer
from sklearn import svm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
import util
import mlutils
import config
import active_mentions_filtering.db as db
import brownclustering.brownclusters as bc
import numpy as np

tknzr = TweetTokenizer(preserve_case=False, reduce_len=True)
translate_table = dict((ord(char), ' ') for char in '\"\'!#$%&()*+,-./:;<=>?@[\\]^_{|}~')
stemmer = PorterStemmer()


def tokenize_and_stem(text):
    text = text.translate(translate_table)
    return [stemmer.stem(t) for t in tknzr.tokenize(text)]

def get_classifier_pipeline():
    bc.load()
    bc_tfidf = TfidfVectorizer(tokenizer=bc.tokenize_and_tag, min_df=10, max_df=0.9)
    tranfsormer = TfidfVectorizer(tokenizer=tokenize_and_stem, stop_words=util.get_stopwords(),
                                  min_df=30, max_df=0.9)
    wtfrmr1 = mlutils.CustomWordClusterTransformer(['anti-depressant', 'mentalillness', 'mental illness',
                                                    'brain disease', 'mental health', 'depressive disorder',
                                                    'mental disorder', 'suicidal', 'suicide', 'anxiety',
                                                    'depression', 'bipolar', 'schizophrenia'])
    wtfrmr2 = mlutils.CustomWordClusterTransformer(['suicidal', 'suicide', 'self-hate'])
    wtfrmr3 = mlutils.CustomWordClusterTransformer(['self-harm', 'self harm'])
    wtfrmr4 = mlutils.CustomWordClusterTransformer(['ptsd', 'p.t.s.d', 'post-traumatic', 'post traumatic', 'stress disorder'])
    pipeline = Pipeline([('features', FeatureUnion([
            ('custom_words1', wtfrmr1),
            ('custom_words2', wtfrmr2),
            ('custom_words3', wtfrmr3),
            ('custom_words4', wtfrmr4),
            ('tfidf_bow', tranfsormer),
            ('bctfidf', bc_tfidf),
        ])),
        # ('todense', FunctionTransformer(lambda x: x.todense(), accept_sparse=True)),
        # ('pca', PCA(n_components=0.75)),
        # ('clf', mlutils.get_thresholded_estimator(RandomForestClassifier, 0.3)(n_estimators=500)),
        ('clf', RandomForestClassifier(n_estimators=500)),
        # ('clf', mlutils.get_thresholded_estimator(CalibratedClassifierCV, 0.3)(svm.LinearSVC()))
        # ('clf', mlutils.get_thresholded_estimator(CalibratedClassifierCV(svm.LinearSVC()), 0.3))
    ])

    return pipeline


def get_valid_labelled_tweets(labelled_tweets, valid_users):
    t = []
    for lt in labelled_tweets:
        if lt[1] in valid_users:
            t.append(lt)
    return t


def get_labelled_data_for_depr(valid_users):
    labelled_tweets = db.get_depr_labelled_tweets(valid_labels=[1, 0])
    if valid_users is not None:
        labelled_tweets = get_valid_labelled_tweets(labelled_tweets, valid_users)
    tweets = np.array([tr[2] for tr in labelled_tweets])
    labels = np.array([tr[3] for tr in labelled_tweets])
    p_tweets = tweets[labels == 1]
    n_tweets = tweets[labels == 0]
    tweets = np.concatenate([p_tweets, n_tweets])
    labels = np.concatenate([np.ones(len(p_tweets)), np.zeros(len(n_tweets))])
    return tweets, labels


def get_labelled_data_for_depr_mh(valid_users):
    labelled_tweets = db.get_depr_labelled_tweets(valid_labels=[1, 2, 0])
    if valid_users is not None:
        labelled_tweets = get_valid_labelled_tweets(labelled_tweets, valid_users)
    tweets = np.array([tr[2] for tr in labelled_tweets])
    labels = np.array([tr[3] for tr in labelled_tweets])
    p_tweets = tweets[np.logical_or(labels == 1, labels == 2)]
    n_tweets = tweets[labels == 0]
    tweets = np.concatenate([p_tweets, n_tweets])
    labels = np.concatenate([np.ones(len(p_tweets)), np.zeros(len(n_tweets))])
    return tweets, labels


def get_labelled_data_for_ptsd(valid_users):
    labelled_tweets = db.get_ptsd_labelled_tweets(valid_labels=[1, 0])
    if valid_users is not None:
        labelled_tweets = get_valid_labelled_tweets(labelled_tweets, valid_users)
    tweets = np.array([tr[2] for tr in labelled_tweets])
    labels = np.array([tr[5] for tr in labelled_tweets])
    p_tweets = tweets[labels == 1]
    n_tweets = tweets[labels == 0]
    tweets = np.concatenate([p_tweets, n_tweets])
    labels = np.concatenate([np.ones(len(p_tweets)), np.zeros(len(n_tweets))])
    return tweets, labels


def get_labelled_data_for_ptsd_mh(valid_users):
    labelled_tweets = db.get_ptsd_labelled_tweets(valid_labels=[1, 2, 0])
    if valid_users is not None:
        labelled_tweets = get_valid_labelled_tweets(labelled_tweets, valid_users)
    tweets = np.array([tr[2] for tr in labelled_tweets])
    labels = np.array([tr[5] for tr in labelled_tweets])
    p_tweets = tweets[np.logical_or(labels == 1, labels == 2)]
    n_tweets = tweets[labels == 0]
    tweets = np.concatenate([p_tweets, n_tweets])
    labels = np.concatenate([np.ones(len(p_tweets)), np.zeros(len(n_tweets))])
    return tweets, labels


def get_depression_model(valid_users=None, forced_retrain=False):
    if not forced_retrain and os.path.isfile(config.ACTIVE_MENTIONS_DEPR_MODEL_PATH):
        return util.load_picke_file(config.ACTIVE_MENTIONS_DEPR_MODEL_PATH)

    tweets, labels = get_labelled_data_for_depr(valid_users)
    pipeline = get_classifier_pipeline()
    pipeline.fit(tweets, labels)
    util.dump_picke_file(pipeline, config.ACTIVE_MENTIONS_DEPR_MODEL_PATH)
    return pipeline


def get_depr_mental_helth_model(valid_users=None, forced_retrain=False):
    if not forced_retrain and os.path.isfile(config.ACTIVE_MENTIONS_DEPR_MH_MODEL_PATH):
        return util.load_picke_file(config.ACTIVE_MENTIONS_DEPR_MH_MODEL_PATH)

    tweets, labels = get_labelled_data_for_depr_mh(valid_users)
    pipeline = get_classifier_pipeline()
    pipeline.fit(tweets, labels)
    util.dump_picke_file(pipeline, config.ACTIVE_MENTIONS_DEPR_MH_MODEL_PATH)
    return pipeline



def get_ptsd_model(valid_users=None, forced_retrain=False):
    if not forced_retrain and os.path.isfile(config.ACTIVE_MENTIONS_PTSD_MODEL_PATH):
        return util.load_picke_file(config.ACTIVE_MENTIONS_PTSD_MODEL_PATH)

    tweets, labels = get_labelled_data_for_ptsd(valid_users)
    pipeline = get_classifier_pipeline()
    pipeline.fit(tweets, labels)
    util.dump_picke_file(pipeline, config.ACTIVE_MENTIONS_PTSD_MODEL_PATH)
    return pipeline


def get_ptsd_mh_model(valid_users=None, forced_retrain=False):
    if not forced_retrain and os.path.isfile(config.ACTIVE_MENTIONS_PTSD_MH_MODEL_PATH):
        return util.load_picke_file(config.ACTIVE_MENTIONS_PTSD_MH_MODEL_PATH)

    tweets, labels = get_labelled_data_for_ptsd(valid_users)
    pipeline = get_classifier_pipeline()
    pipeline.fit(tweets, labels)
    util.dump_picke_file(pipeline, config.ACTIVE_MENTIONS_PTSD_MH_MODEL_PATH)
    return pipeline


def evaluate_pipeline(tweets, labels, pipeline, cv_folds=10):
    scoring = ['f1', 'precision', 'recall', 'roc_auc']
    scores = cross_validate(pipeline, tweets, labels, scoring=scoring, cv=cv_folds, return_train_score=False)

    for t in scores.keys():
        s = scores[t]
        print(t, str(round(s.mean(), 3)) + '(' + str(round(s.std(), 3)) + '), ')
    return scores