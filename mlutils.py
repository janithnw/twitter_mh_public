import uuid

import numpy as np
import scipy.sparse
from nltk import TweetTokenizer
from sklearn.base import BaseEstimator, TransformerMixin
import datetime
import re
import pandas as pd
import os
import subprocess
import config
import util
import math

from sklearn.feature_extraction.text import CountVectorizer

tknzr = TweetTokenizer(reduce_len=True)

def tokenize_only_alphanumeric_tokens(text):
    return [t for t in tknzr.tokenize(text) if re.match(r"[a-zA-Z\s'#]|(_e_(\w)+_e_)", t) is not None]


def selector_fn_noop(df_rec):
    return {
        'text': df_rec['tweets'],
        'times': df_rec['created_at'],
        'cmu_pos_tags': df_rec['cmu_pos_tags'],
        'nltk_pos_tags': df_rec['nltk_pos_tags']
    }
#
# def selector_fn_all_tweets(df_rec):
#     return {
#         'text': df_rec['tweets'],
#         'times': df_rec['created_at'],
#         'cmu_pos_tags': df_rec['cmu_pos_tags'],
#         'nltk_pos_tags': df_rec['nltk_pos_tags']
#     }
#
# def selector_fn_filtered_tweets(r):
#     return {
#         'text': [r['tweets'][i] for i in range(len(r['mental_health_probs'])) if r['mental_health_probs'][i] <= 0.5],
#         'times': [r['created_at'][i] for i in range(len(r['mental_health_probs'])) if r['mental_health_probs'][i] <= 0.5],
#         'cmu_pos_tags': r['cmu_pos_tags'],
#         'nltk_pos_tags': r['nltk_pos_tags']
#     }
#
# def selector_fn_all_cmu_pos_tags(df_rec):
#     return {
#         'text': [' '.join(t[1] for t in tagged_tweet)  for tagged_tweet in df_rec['cmu_pos_tags']],
#         'times': df_rec['created_at']
#     }
#
# def selector_fn_all_nltk_pos_tags(df_rec):
#     return {
#         'text': df_rec['nltk_pos_tags'],
#         'times': df_rec['created_at']
#     }


def iterate(iter):
    if isinstance(iter, pd.DataFrame):
        return iter.iterrows()
    elif isinstance(iter, list):
        return enumerate(iter)

class ColumnSelector(BaseEstimator, TransformerMixin):

    def __init__(self, column):
        self.column = column

    def fit(self, x, y=None):
        return self


    def concatenate(self, record):
        d = '\n~~\n'.join(record)
        d = d.replace('USER ', '').replace(' USER', '')
        return d

    def transform(self, df):
        if isinstance(df, pd.DataFrame):
            if callable(self.column):
                selector_fn = self.column
                return [self.concatenate(selector_fn(row)['text']) for i, row in df.iterrows()]
            else:
                return df[self.column].apply(self.concatenate)
        elif isinstance(df, list) and (isinstance(df[0], dict) or isinstance(df[0], pd.Series)):
            if callable(self.column):
                selector_fn = self.column
                return [self.concatenate(selector_fn(row)['text']) for row in df]
            else:
                return [self.concatenate(row[self.column]) for row in df]
        else:
            return [self.concatenate(tweets) for tweets in df]


class POSTagColumnSelector(BaseEstimator, TransformerMixin):

    def __init__(self, column):
        self.column = column

    def fit(self, x, y=None):
        return self

    def concatenate(self, record):
        p = ''
        for t in record:
            p = p + ' '.join(tag[1] for tag in t) + '\n'
        return p

    def transform(self, df):
        if isinstance(df, pd.DataFrame):
            return df[self.column].apply(self.concatenate)
        elif isinstance(df, list) and (isinstance(df[0], dict) or isinstance(df[0], pd.Series)):
            return [self.concatenate(row[self.column]) for row in df]
        else:
            return [self.concatenate(tweets) for tweets in df]


class CustomWordClusterTransformer(BaseEstimator, TransformerMixin):
    """
    This transformer returns [1] if the input instance contains any of the
    words given
    """
    def __init__(self, words):
        self.words = words

    def fit(self, x, y=None):
        return self

    def __check(self, text):
        for w in self.words:
            if w in text:
                return [1]
        return [0]

    def transform(self, texts):
        res = np.array(list(map(self.__check, texts)))
        return res


def chunk_tweets_by_week(tweets, times):
    start_date = times[-1]
    end_date = start_date + datetime.timedelta(days=7)
    chunks = []
    chunk = []
    i = len(tweets) - 1
    while i >= 0:
        if times[i] <= end_date:
            chunk.append(tweets[i])
            i -= 1
        else:
            end_date = end_date + datetime.timedelta(days=7)
            if len(chunk) > 0:
                chunks.append(chunk)
            chunk = []
    chunks.append(chunk)
    return chunks


class TweetChunker(BaseEstimator, TransformerMixin):

    def __init__(self, transformer, selector_fn):
        self.transformer = transformer
        self.selector_fn = selector_fn

    def fit(self, df, labels=None):
        X = []
        Y = []
        for i in range(len(labels)):
            r = self.selector_fn(df.iloc[i])
            chunks = chunk_tweets_by_week(r['text'], r['times'])
            for chunk in chunks:
                X.append('\n'.join(chunk))
                Y.append(labels[i])
        self.transformer.fit(X, Y)
        return self

    def transform(self, df):
        chunks = []
        user_indexes = []
        chunk_lengths = []
        for i, r in iterate(df):
            r = self.selector_fn(r)
            c = chunk_tweets_by_week(r['text'], r['times'])
            user_indexes.extend([i] * len(c))
            chunks.extend(['\n'.join(cc) for cc in c])
            chunk_lengths.extend([len(cc) for cc in c])

        x = self.transformer.transform(chunks)
        if scipy.sparse.issparse(x):
            x = x.todense()
        x = np.array(x)
        user_indexes = np.array(user_indexes)
        chunk_lengths = np.array(chunk_lengths)

        X = []
        for i, r in iterate(df):
            idxs = np.where(user_indexes == i)[0]
            if len(idxs) > 0:
                x_tr = x[idxs, :]
                cl = chunk_lengths[idxs]
                cl = cl / sum(cl)
                # print('Before: ', cl[:, None].shape, x_tr.shape)
                if cl[:, None].shape[0] == x_tr.shape[0]:
                    x_tr = np.dot(cl[:, None].T, x_tr)
                else:
                    x_tr = np.dot(cl[:, None], x_tr)
                # x_tr = np.sum(np.dot(cl[:, None].T, x_tr), axis=0)
                x_tr = x_tr[0]
                # print('After: ', x_tr.shape)
            else:
                x_tr = np.zeros(len(x[0]))
            X.append(x_tr)
        X = np.array(X)
        # print(X.shape)
        return X


class POSTagChunker(BaseEstimator, TransformerMixin):

    def __init__(self, transformer, selector_fn, tags_column):
        self.transformer = transformer
        self.selector_fn = selector_fn
        self.tags_column = tags_column

    def __flatten_tagged_tweets(self, tweets):
        return '\n'.join([' '.join([t[1] for t in tweet]) for tweet in tweets])

    def fit(self, df, labels=None):
        X = []
        Y = []
        for i in range(len(labels)):
            r = self.selector_fn(df.iloc[i])
            chunks = chunk_tweets_by_week(r[self.tags_column], r['times'])
            for chunk in chunks:
                X.append(self.__flatten_tagged_tweets(chunk))
                # X.append('\n'.join(chunk))
                Y.append(labels[i])
        self.transformer.fit(X, Y)
        return self

    def transform(self, df):
        chunks = []
        user_indexes = []
        chunk_lengths = []
        for i, r in iterate(df):
            r = self.selector_fn(r)
            c = chunk_tweets_by_week(r[self.tags_column], r['times'])
            user_indexes.extend([i] * len(c))
            chunks.extend([self.__flatten_tagged_tweets(cc) for cc in c])
            chunk_lengths.extend([len(cc) for cc in c])

        x = self.transformer.transform(chunks)
        if scipy.sparse.issparse(x):
            x = x.todense()
        x = np.array(x)
        user_indexes = np.array(user_indexes)
        chunk_lengths = np.array(chunk_lengths)

        X = []
        for i, r in iterate(df):
            idxs = np.where(user_indexes == i)[0]
            if len(idxs) > 0:
                x_tr = x[idxs, :]
                cl = chunk_lengths[idxs]
                cl = cl / sum(cl)
                # print('Before: ', cl[:, None].shape, x_tr.shape)
                if cl[:, None].shape[0] == x_tr.shape[0]:
                    x_tr = np.dot(cl[:, None].T, x_tr)
                else:
                    x_tr = np.dot(cl[:, None], x_tr)
                # x_tr = np.sum(np.dot(cl[:, None].T, x_tr), axis=0)
                x_tr = x_tr[0]
                # print('After: ', x_tr.shape)
            else:
                x_tr = np.zeros(len(x[0]))
            X.append(x_tr)
        X = np.array(X)
        # print(X.shape)
        return X


def run(*popenargs, input=None, check=False, **kwargs):
    if input is not None:
        if 'stdin' in kwargs:
            raise ValueError('stdin and input arguments may not both be used.')
        kwargs['stdin'] = subprocess.PIPE

    process = subprocess.Popen(*popenargs, **kwargs)
    try:
        stdout, stderr = process.communicate(input)
    except:
        process.kill()
        process.wait()
        raise
    retcode = process.poll()
    if check and retcode:
        raise subprocess.CalledProcessError(
            retcode, process.args, output=stdout, stderr=stderr)
    return retcode, stdout, stderr


class SLDA(BaseEstimator, TransformerMixin):

    def __init__(self, cv_vocab, dir=None, force_retrain=False, iters=100, e_step_iters=100, priors_model=config.CTRL_DEPR_TOPIC_PRIORS_MODEL):
        self.cv_vocab = cv_vocab
        self.cv = CountVectorizer(vocabulary=util.load_picke_file(cv_vocab), tokenizer=tokenize_only_alphanumeric_tokens)
        self.cv._validate_vocabulary()
        self.iters = iters
        self.e_step_iters = e_step_iters
        self.already_trained = False
        self.priors_model = priors_model
        self.dir = dir
        if self.dir is None:
            self.model_dir = config.SLDA_MODELS_DIR + uuid.uuid4().hex + '/'
            os.makedirs(self.model_dir)
        else:
            self.model_dir = config.SLDA_MODELS_DIR + self.dir + '/'
            if os.path.isdir(self.model_dir) and not force_retrain:
                self.already_trained = True
            else:
                os.makedirs(self.model_dir)

        self.train_input_file = self.model_dir + 'input.npy'
        self.transform_input_file = self.model_dir + 'input.npy'
        self.model_file = self.model_dir + 'model.npy'
        self.transformed_file = self.model_dir + 'transformed.npy'

    def __call_train(self):
        print('Calling train')
        cmds = ['fslda', 'train',
                '--topics', '50',
                '--iterations', str(self.iters),
                '--e_step_iterations', str(self.e_step_iters),
                '--e_step_tolerance', '0.1',
                '--snapshot_every', '10',
                '--workers', '4',
                '--continue_from_unsupervised', self.priors_model,
                self.train_input_file,
                self.model_file]
        o = run(cmds, stdout=subprocess.PIPE)
        return o

    def __call_transform(self):
        cmds = ['fslda', 'transform',
                '--workers', '4',
                self.model_file,
                self.transform_input_file,
                self.transformed_file]
        # print(' '.join(cmds))
        o = run(cmds, stdout=subprocess.PIPE)
        return o

    def fit(self, x, y=None):
        if self.already_trained:
            print('Already trained, skipping training')
            return self

        self.cv._validate_vocabulary()
        xx = np.array(self.cv.transform(x).todense())
        yy = np.array(y)
        with open(self.train_input_file, "wb") as f:
            np.save(f, xx.astype(np.int32).T)
            np.save(f, yy.astype(np.int32))

        o = self.__call_train()
        #         print(o.stdout.decode('ascii'))
        return self


    def transform(self, x):
        self.cv._validate_vocabulary()
        xx = np.array(self.cv.transform(x).todense())
        with open(self.transform_input_file, "wb") as f:
            np.save(f, xx.astype(np.int32).T)

        o = self.__call_transform()
        #         print(o.stdout.decode('ascii'))

        with open(self.transformed_file, "rb") as f:
            xx_tr = np.load(f).T

        xx_tr = xx_tr / xx_tr.sum(axis=1)[:, None]
        return xx_tr


def get_thresholded_estimator(class_name, threshold):

    class Thresholded(class_name):
        def predict(self, X):
            return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    return Thresholded


class DenseTransformer(TransformerMixin):

    def transform(self, X, y=None, **fit_params):
        return X.todense()

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self


def H(p, n):
    if p == 0 or n == 0:
        return 0
    return -p / (p + n) * math.log2(p / (p + n)) - n / (p + n) * math.log2(n / (p + n))


def t_infogain(X, Y, t, i):
    Y1 = Y[X[:, i] < t]
    p1_i = Y1.sum()
    n1_i = len(Y1) - p1_i

    Y2 = Y[X[:, i] >= t]
    p2_i = Y2.sum()
    n2_i = len(Y2) - p2_i

    return H(Y.sum(), len(Y) - Y.sum()) - (H(p1_i, n1_i) * len(Y1) / len(Y) + H(p2_i, n2_i) * len(Y2) / len(Y))


def infogain(X, Y, i):
    m = X[:, i].mean()
    s = X[:, i].std()
    start = m - 3 * s
    stop = m + 3 * s
    step = (stop - start) / 100
    t_range = np.arange(start, stop, step)
    infogains = [t_infogain(X, Y, t, i) for t in t_range]
    if len(infogains) == 0:
        return 0
    return max(infogains)

'''
X - Input Matrix
Y - labels (binary - 0,1)
'''
def compute_infogain(X, Y):
    return [infogain(X, Y, i) for i in range(X.shape[1])]

def compute_cohensd(X, Y):
    pos_mean = X[Y==1].mean(axis=0)
    neg_mean = X[Y==0].mean(axis=0)
    # pos_std = X[Y==1].std(axis=0)
    # neg_std = X[Y==0].std(axis=0)
    # cd1 = 2 * (pos_mean - neg_mean) / (pos_std + neg_std)
    std = X.std(axis=0)
    # cd2 = (pos_mean - neg_mean) / std
    # print(cd1, cd2)
    return (pos_mean - neg_mean)/std


def compute_pvals(X, Y):
    return [scipy.stats.ttest_ind(X[Y == 0, i], X[Y == 1, i]) for i in range(X.shape[1])]