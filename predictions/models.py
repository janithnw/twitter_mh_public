import os.path
import sys

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from sklearn import svm, metrics
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline, FeatureUnion

import config
import util
from brownclustering import brownclusters as bc
import mlutils
bc.load()



def get_transformers(col_selector_fn, slda_dir, slda_priors_model, vocab_path):
    vocab = util.load_picke_file(vocab_path)
    return {
        'aggr_bow': [
            ('union', FeatureUnion(transformer_list=[
                ('aggr', mlutils.TweetChunker(FeatureUnion([
                    ('bow', TfidfVectorizer(vocabulary=vocab, tokenizer=mlutils.tokenize_only_alphanumeric_tokens)),
                ]), col_selector_fn))
            ]))],

        'aggr_clusters': [
            ('union', FeatureUnion(transformer_list=[
                ('aggr', mlutils.TweetChunker(FeatureUnion([
                    ('clusters', TfidfVectorizer(tokenizer=bc.tokenize_and_tag, min_df=0.01)),
                ]), col_selector_fn))
            ]))],

        'bow': [
            ('union', FeatureUnion(transformer_list=[
                ('bagofwords', Pipeline([
                    ('selector', mlutils.ColumnSelector(col_selector_fn)),
                    ('tfidf', TfidfVectorizer(vocabulary=vocab, tokenizer=mlutils.tokenize_only_alphanumeric_tokens))
                ]))
            ]))],

        'clusters': [
            ('union', FeatureUnion(transformer_list=[
                ('brown_clusters', Pipeline([
                    ('selector', mlutils.ColumnSelector(col_selector_fn)),
                    ('tfidf', TfidfVectorizer(tokenizer=bc.tokenize_and_tag, min_df=0.01))
                ]))
            ]))],

        'slda': [
            ('union', FeatureUnion(transformer_list=[
                ('agg', mlutils.TweetChunker(FeatureUnion([
                    ('slda', mlutils.SLDA(vocab_path, dir=slda_dir, iters=100, e_step_iters=100, priors_model=slda_priors_model))
                ]), col_selector_fn))
            ]))],

        'aggslda_bow': [
            ('union', FeatureUnion(transformer_list=[
                ('agg', mlutils.TweetChunker(FeatureUnion([
                    ('slda', mlutils.SLDA(vocab_path, dir=slda_dir, iters=100, e_step_iters=100, priors_model=slda_priors_model)),
                    ('bow', TfidfVectorizer(vocabulary=vocab, tokenizer=mlutils.tokenize_only_alphanumeric_tokens)),
                ]), col_selector_fn))
            ]))],

        'aggslda_aggbow_clusters': [
            ('union', FeatureUnion(transformer_list=[
                ('agg', mlutils.TweetChunker(FeatureUnion([
                    ('slda', mlutils.SLDA(vocab_path, dir=slda_dir, iters=100, e_step_iters=100, priors_model=slda_priors_model)),
                    ('bow', TfidfVectorizer(vocabulary=vocab, tokenizer=mlutils.tokenize_only_alphanumeric_tokens)),
                ]), col_selector_fn)),
                ('brown_clusters', Pipeline([
                    ('selector', mlutils.ColumnSelector(col_selector_fn)),
                    ('tfidf', TfidfVectorizer(tokenizer=bc.tokenize_and_tag, min_df=0.01))
                ]))
            ]))],


        'aggslda_aggbow_clusters_nltk': [
            ('union', FeatureUnion(transformer_list=[
                ('agg', mlutils.TweetChunker(FeatureUnion([
                    ('slda', mlutils.SLDA(vocab_path, dir=slda_dir, iters=100, e_step_iters=100, priors_model=slda_priors_model)),
                    ('bow', TfidfVectorizer(vocabulary=vocab, tokenizer=mlutils.tokenize_only_alphanumeric_tokens)),
                ]), col_selector_fn)),
                ('brown_clusters', Pipeline([
                    ('selector', mlutils.ColumnSelector(col_selector_fn)),
                    ('tfidf', TfidfVectorizer(tokenizer=bc.tokenize_and_tag, min_df=0.01))
                ])),
                ('pos_tags', Pipeline([
                    ('selector', mlutils.POSTagColumnSelector('nltk_pos_tags')),
                    ('tfidf', TfidfVectorizer(tokenizer=mlutils.tknzr.tokenize, min_df=0.01, ngram_range=(1, 3)))
                ]))
            ]))],


        'aggslda_aggbow_clusters_cmu': [
            ('union', FeatureUnion(transformer_list=[
                ('agg', mlutils.TweetChunker(FeatureUnion([
                    ('slda', mlutils.SLDA(vocab_path, dir=slda_dir, iters=100, e_step_iters=100, priors_model=slda_priors_model)),
                    ('bow', TfidfVectorizer(vocabulary=vocab, tokenizer=mlutils.tokenize_only_alphanumeric_tokens)),
                ]), col_selector_fn)),
                ('brown_clusters', Pipeline([
                    ('selector', mlutils.ColumnSelector(col_selector_fn)),
                    ('tfidf', TfidfVectorizer(tokenizer=bc.tokenize_and_tag, min_df=0.01))
                ])),
                ('pos_tags', Pipeline([
                    ('selector', mlutils.POSTagColumnSelector('cmu_pos_tags')),
                    ('tfidf', TfidfVectorizer(tokenizer=mlutils.tknzr.tokenize, min_df=0.01, ngram_range=(1, 3)))
                ]))
            ]))],

        'bow_clusters_cmu': [
            ('union', FeatureUnion(transformer_list=[
                ('bagofwords', Pipeline([
                    ('selector', mlutils.ColumnSelector(col_selector_fn)),
                    ('tfidf', TfidfVectorizer(vocabulary=vocab, tokenizer=mlutils.tokenize_only_alphanumeric_tokens))
                ])),
                ('brown_clusters', Pipeline([
                    ('selector', mlutils.ColumnSelector(col_selector_fn)),
                    ('tfidf', TfidfVectorizer(tokenizer=bc.tokenize_and_tag, min_df=0.01))
                ])),
                ('pos_tags', Pipeline([
                    ('selector', mlutils.POSTagColumnSelector('cmu_pos_tags')),
                    ('tfidf', TfidfVectorizer(tokenizer=mlutils.tknzr.tokenize, min_df=0.01, ngram_range=(1, 3)))
                ]))
            ]))],

        'bow_clusters_nltk': [
            ('union', FeatureUnion(transformer_list=[
                ('bagofwords', Pipeline([
                    ('selector', mlutils.ColumnSelector(col_selector_fn)),
                    ('tfidf', TfidfVectorizer(vocabulary=vocab, tokenizer=mlutils.tokenize_only_alphanumeric_tokens))
                ])),
                ('brown_clusters', Pipeline([
                    ('selector', mlutils.ColumnSelector(col_selector_fn)),
                    ('tfidf', TfidfVectorizer(tokenizer=bc.tokenize_and_tag, min_df=0.01))
                ])),
                ('pos_tags', Pipeline([
                    ('selector', mlutils.POSTagColumnSelector('nltk_pos_tags')),
                    ('tfidf', TfidfVectorizer(tokenizer=mlutils.tknzr.tokenize, min_df=0.01, ngram_range=(1, 3)))
                ]))
            ]))]
    }

def get_pipelines(col_selector_fn, slda_dir, clf, slda_priors_model, vocab):
    transformers = get_transformers(col_selector_fn, slda_dir, slda_priors_model, vocab)
    pipelines = {}
    for k in transformers.keys():
        pipelines[k] = Pipeline(transformers[k] + [('clf', clf)])
    return pipelines


def evaluate(pipeline, train, test):
    pipeline.fit(train, train['labels'].astype(int).values)
    preds = pipeline.predict_proba(test)[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(test['labels'].astype(int).values, preds, pos_label=1)
    print('AUC: ', metrics.auc(fpr, tpr))
    print('AP: ', metrics.average_precision_score(test['labels'].astype(int).values, preds))
    print(classification_report(test['labels'].astype(int).values, preds >= 0.5))
    return preds

