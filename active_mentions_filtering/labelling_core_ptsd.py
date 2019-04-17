import os.path
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import sqlite3
import numpy as np
import util
from labelling.classifier_core import get_tweets_to_label, evaluate_classifier_cv


def get_connection():
    return sqlite3.connect(util.get_file_path('labelling/tweets.db'))


def get_unlabelled_tweets_ptsd(annotator=1):
    conn = get_connection()
    c = conn.cursor()
    query = 'SELECT * FROM tweets WHERE tweets.is_about_ptsd is null AND tweets.username IN (SELECT username FROM users WHERE condition=2) ORDER BY RANDOM() LIMIT 10000'
    if annotator == 2:
        query = 'SELECT * FROM tweets WHERE tweets.is_about_ptsd2 is null AND tweets.is_about_ptsd is not null ORDER BY RANDOM()'
    print(query)
    res = c.execute(query).fetchall()
    conn.close()
    return np.array(res)


def label_tweets(tweet_tuples):
    res = []
    for t in tweet_tuples:
        print(t[1] + ': ' + t[2])
        resp = input('Response [0:Control, 1:PTSD, 2:Other mental health, -1:Unsure, c: Saves this session]: ')
        if resp == 'c':
            return res
        else:
            try:
                res.append((t[0], int(resp)))
            except:
                print("Error with response: " + resp)
    return res


def get_labelled_tweets(valid_labels=[1, 0]):
    conn = get_connection()
    c = conn.cursor()
    res = c.execute("SELECT * FROM tweets WHERE tweets.is_about_ptsd IN (" + ",".join([str(i) for i in valid_labels]) + ")").fetchall()
    conn.close()
    return res


def update_labels(labels, annotator=1):
    conn = get_connection()
    c = conn.cursor()
    for label in labels:
        if annotator == 1:
            c.execute('UPDATE tweets SET is_about_ptsd = ? WHERE id=?', (label[1], label[0]))
        elif annotator == 2:
            c.execute('UPDATE tweets SET is_about_ptsd2 = ? WHERE id=?', (label[1], label[0]))
    conn.commit()
    conn.close()


if __name__ == "__main__":

    # u = get_unlabelled_tweets_ptsd(annotator=1)
    # l = get_labelled_tweets(valid_labels=[0, 1])
    # t = get_tweets_to_label(l, u, labels_col_idx=5)
    # labels = label_tweets(t)
    # update_labels(labels, annotator=1)
    # print(len(labels))

    l = get_labelled_tweets(valid_labels=[0, 1, 2])
    evaluate_classifier_cv(l)




