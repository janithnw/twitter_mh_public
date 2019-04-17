import sys, os
sys.path.insert(1, '..')
import util
import pickle
import mlutils
import numpy as np
import pandas as pd
import config
import models
from sklearn.model_selection import cross_validate
from sklearn import svm
from sklearn.calibration import CalibratedClassifierCV


cv_folds = 10
FILTERED_ALL = 'filtered'
vocab_path = config.CTRL_DEPR_VOCAB_PATH
slda_priors_model = config.CTRL_DEPR_TOPIC_PRIORS_MODEL
pipelines_to_eval = [
    'bow',
    'bow_clusters_cmu',
    'bow_clusters_nltk',
    # 'slda',
    'aggslda_bow',
    'aggslda_aggbow_clusters',
    'aggslda_aggbow_clusters_nltk',
    'aggslda_aggbow_clusters_cmu'
]


def cross_val(pipeline, df, labels):
    scoring = ['f1', 'precision', 'recall', 'average_precision', 'roc_auc']
    scores = cross_validate(pipeline, df, labels, scoring=scoring, cv=cv_folds, return_train_score=False, verbose=3)

    for t in scoring:
        s = scores['test_' + t]
        print(str(round(s.mean(), 3)) + '(' + str(round(s.std(), 3)) + '), ', end=' ')
    print()
    return scores


if __name__ == "__main__":
    df = pd.concat([util.load_picke_file(config.CTRL_DEPR_FILTERED_DF), util.load_picke_file(config.CTRL_DEPR_HELD_OUT_FILTERED_DF)])
    # df = pd.concat([util.load_picke_file(config.CTRL_DEPR_DF), util.load_picke_file(config.CTRL_DEPR_HELD_OUT_DF)])
    # df = pd.concat([util.load_picke_file(config.CTRL_PTSD_DF), util.load_picke_file(config.CTRL_PTSD_HELD_OUT_DF)])
    # df = pd.concat([util.load_picke_file(config.CTRL_PTSD_FILTERED_DF), util.load_picke_file(config.CTRL_PTSD_HELD_OUT_FILTERED_DF)])
    # df = util.load_picke_file(config.CTRL_PTSD_HELD_OUT_DF)
    labels = df['labels'].astype(int).values

    pipelines = models.get_pipelines(mlutils.selector_fn_noop, 'split_filtered_tweets', CalibratedClassifierCV(svm.LinearSVC(), cv=3),
                                     slda_priors_model, vocab_path)


    # cross_val_results = {}
    cross_val_results = util.load_picke_file('crossval_dvc_filtered_results.p')
    for p in pipelines_to_eval:
        if p in cross_val_results:
            continue
        print(p + '\t', end=' ', flush=True)
        cross_val_results[p] = cross_val(pipelines[p], df, labels)
        util.dump_picke_file(cross_val_results, 'crossval_dvc_filtered_results.p')