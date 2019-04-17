import os.path
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import sqlite3
import glob
import os
import csv
import json
from common import LABEL_IDS, valid_tweet, preprocess
import pickle
# from random import shuffle
from labelling.classifier_core import get_tweets_to_label, evaluate_classifier_split, evaluate_classifier_loo, \
    evaluate_classifier_cv
import numpy as np
import util

"""
Code related to labelling tweets
"""

def get_connection():
    return sqlite3.connect(util.get_file_path('labelling/tweets.db'))


def get_labelled_tweets(valid_labels=[1, 0]):
    conn = get_connection()
    c = conn.cursor()
    res = c.execute("SELECT * FROM tweets WHERE tweets.is_about_depression IN (" + ",".join([str(i) for i in valid_labels]) + ")").fetchall()
    conn.close()
    return res


def get_labelled_tweets_to_reannotate(valid_labels=[1, 0]):
    conn = get_connection()
    c = conn.cursor()
    res = c.execute("SELECT * FROM tweets WHERE tweets.is_about_depression IN (" + ",".join([str(i) for i in valid_labels]) + ") AND tweets.is_about_depression3 IS NULL AND tweets.is_about_depression2 IS NOT NULL").fetchall()
    conn.close()
    return res


def get_unlabelled_tweets(user_type=1):
    conn = get_connection()
    c = conn.cursor()
    #res = c.execute('SELECT * FROM tweets WHERE tweets.is_about_depression is null AND tweets.username IN (SELECT username FROM users WHERE mentions_depr=1)').fetchall()
    # res = c.execute('SELECT * FROM tweets WHERE tweets.is_about_depression is null').fetchall()
    res = c.execute('SELECT * FROM tweets WHERE tweets.is_about_depression is null and username in (SELECT username FROM users WHERE condition=' + str(user_type) + ')').fetchall()
    conn.close()
    return np.array(res)

def get_unlabelled_tweets_reannotation():
    """
    Returns the tweets that were annotated by the first annotator to be annotated by the second annotator
    :return:
    """
    conn = get_connection()
    c = conn.cursor()
    #res = c.execute('SELECT * FROM tweets WHERE tweets.is_about_depression is null AND tweets.username IN (SELECT username FROM users WHERE mentions_depr=1)').fetchall()
    res = c.execute('SELECT * FROM tweets WHERE tweets.is_about_depression IN (0, 1, 2) AND tweets.is_about_depression2 IS NULL ORDER BY random()').fetchall()
    conn.close()
    return np.array(res)


def label_tweets(tweet_tuples):
    res = []
    for t in tweet_tuples:
        print(t[1] + ': ' + t[2])
        resp = input('Response [0:Control, 1:Depression, 2:Other mental health, -1:Unsure, c: Saves this session]: ')
        if resp == 'c':
            return res
        else:
            try:
                res.append((t[0], int(resp)))
            except:
                print("Error with response: " + resp)
    return res


def update_labels(labels, annotator=1):
    conn = get_connection()
    c = conn.cursor()
    for label in labels:
        if annotator == 1:
            c.execute('UPDATE tweets SET is_about_depression = ? WHERE id=?', (label[1], label[0]))
        elif annotator == 2:
            c.execute('UPDATE tweets SET is_about_depression2 = ? WHERE id=?', (label[1], label[0]))
        else:
            c.execute('UPDATE tweets SET is_about_depression3 = ? WHERE id=?', (label[1], label[0]))
    conn.commit()
    conn.close()

if __name__ == "__main__":
    # l = get_labelled_tweets_to_reannotate(valid_labels=[2])
    # u = get_unlabelled_tweets_reannotation()
    # t = get_tweets_to_label(l, u)
    # print(len(l))
    # labels = label_tweets(l)
    # update_labels(labels, annotator=3)
    # print(len(labels))
    l = get_labelled_tweets(valid_labels=[0, 1, 2])
    evaluate_classifier_cv(l)





"""
Initial code used to load data to the databse from the Data directory
"""
def load_data_to_db(dir_prefix, db):
    tweet_files = glob.glob(os.path.join(dir_prefix, '*.*'))

    with open('../../anonymized_user_info_by_chunk_training.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader, None)
        users = {row[0]: {'age': row[1], 'num_tweets': row[2], 'gender': row[3], 'condition': row[4]} for row in reader}

        # user_records = [(un, float(users[un]['age']), int(users[un]['num_tweets']), users[un]['gender'], LABEL_IDS[users[un]['condition']]) for un in users.keys()]
        #
        # c = db.cursor()
        # c.executemany('INSERT INTO users VALUES (?,?,?,?,?)', user_records)
        # db.commit()

        tweet_records = []
        for file in tweet_files:
            username = os.path.splitext(os.path.basename(file))[0]
            print(username)

            label = LABEL_IDS[users[username]['condition']]
            tweet_file = open(file, 'r')
            i = 0
            for line in tweet_file:
                tweet = json.loads(line)
                i += 1
                if valid_tweet(tweet):
                    t = preprocess(tweet['text'])
                    tweet_records.append((username, t))
            tweet_file.close()

        # c = db.cursor()
        # c.executemany('INSERT INTO tweets (username, text) VALUES (?,?)', tweet_records)
        # db.commit()


def update_users_mentioning_depr():
    [users_using_tk, indexes] = pickle.load(open("../temp_data/users_using_depr.p", "rb"))
    conn = get_connection()
    c = conn.cursor()
    for user in users_using_tk.keys():
        c.execute('UPDATE users SET mentions_depr = 1 WHERE username=\"'+user+'\"')

    conn.commit()
    conn.close()
