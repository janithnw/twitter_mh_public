import csv
from datetime import datetime
import glob
import os
import html

import nltk
import pandas as pd
import numpy as np
import json
import emoji
import re

from sklearn.model_selection import train_test_split
from unidecode import unidecode
from external_scripts import CMUTweetTagger
import config
import util

def is_retweeted_tweet(tweet):
    return 'http' in tweet['text'] or 'retweeted_status' in tweet or tweet['text'][:2] == 'RT'


regex_mentions = re.compile(r'@\w+:?')


def preprocess(tweet_text, preserve_case=False):
    tweet_text = emoji.demojize(tweet_text, delimiters=(" _E_", "_E_ "))
    tweet_text = unidecode(tweet_text)
    if not preserve_case:
        tweet_text = tweet_text.lower()
    tweet_text = html.unescape(regex_mentions.sub("USER", tweet_text))
    return tweet_text


def correct_emoji_pos_tag(tag):
    if tag[0] in emoji.UNICODE_EMOJI:
        if len(tag) == 3:
            return (tag[0], 'EMJ', tag[2])
        else:
            return (tag[0], 'EMJ')
    return tag


def correct_emoji_pos_tags(tags):
    return [correct_emoji_pos_tag(t) for t in tags]


def clean_for_cmu_tagging(text):
    return re.sub('\n|\r|\t', '', text)


def load_user_matchings():
    with open(config.TRAINING_LABELS_PATH, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)  # skip the headers
        records = [row for row in reader]
        return {records[i][0]: records[i+1][0] for i in range(0, len(records), 2)}


def get_user_pairs_for(possitive_label):
    matchings = load_user_matchings()
    df = pd.read_csv(config.TRAINING_LABELS_PATH)
    pos_users = df.where(df['condition'] == possitive_label).dropna()['anonymized_screen_name'].values
    users = np.concatenate([pos_users, [matchings[u] for u in pos_users]])
    return users


def load_tweets_to_df(valid_labels=[0, 1], tweets_dir=config.TRAINING_TWEETS_PATH, labels_dir=config.TRAINING_LABELS_PATH, valid_users=None):
    tknzr_pos_tagging = nltk.TweetTokenizer(preserve_case=True, reduce_len=True)

    tweet_files = glob.glob(os.path.join(tweets_dir, '*.*'))

    with open(labels_dir, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)
        users = {row[0]: {'age': row[1], 'num_tweets': row[2], 'gender': row[3], 'condition': row[4]} for row in reader}

        X = []
        D = []
        Y = []
        Users = []
        NLTK_PTAGS = []
        CMU_PTAGS = []
        tweets_to_tag = []
        i = 0
        for file in tweet_files:
            username = os.path.splitext(os.path.basename(file))[0]
            if valid_users is not None and username not in valid_users:
                continue
            print(username)

            label = config.LABEL_IDS[users[username]['condition']]
            if label not in valid_labels:
                continue
            tweet_file = open(file, 'r')
            tweets = []
            dates = []
            nltk_pos_tags = []
            cmu_pos_tags = []
            for line in tweet_file:
                tweet = json.loads(line)
                if not is_retweeted_tweet(tweet):
                    t = preprocess(tweet['text'], preserve_case=True)
                    tweets.append(t)
                    dates.append(datetime.strptime(tweet['created_at'], '%a %b %d %H:%M:%S +0000 %Y'))
                    nltk_pos_tags.append(correct_emoji_pos_tags(nltk.pos_tag(tknzr_pos_tagging.tokenize(tweet['text']))))
                    cmu_pos_tags.append(i)
                    tweets_to_tag.append(clean_for_cmu_tagging(tweet['text']))
                    i += 1
            tweet_file.close()
            D.append(np.array(dates))
            X.append(np.array(tweets))
            Y.append(label)
            Users.append(username)
            NLTK_PTAGS.append(nltk_pos_tags)
            CMU_PTAGS.append(cmu_pos_tags)

        PTAGS_NEW = []
        tagged = CMUTweetTagger.runtagger_parse(tweets_to_tag)
        for record in CMU_PTAGS:
            PTAGS_NEW.append([correct_emoji_pos_tags(tagged[r]) for r in record])
        CMU_PTAGS = PTAGS_NEW
        df = pd.DataFrame(data=np.vstack([X, CMU_PTAGS, NLTK_PTAGS, D, Y]).transpose(), index=Users, columns=['tweets', 'cmu_pos_tags', 'nltk_pos_tags', 'created_at', 'labels'])
        return df



def create_data_sets():
    """
    Loads the files from CLPsych data directory, and create the following data frames:
        * ctrl_depr
        * ctrl_depr_held_out
        * ctrl_ptsd
        * ctrl_ptsd_held_out
    Each dataset contains valid tweets, nltk and cmu pos tags
    :return: ctrl_depr, ctrl_depr_held_out
    """
    valid_users = get_user_pairs_for('depression')
    ctrl_depr_full = load_tweets_to_df(valid_labels=[0, 1], valid_users=valid_users)
    print(len(ctrl_depr_full))
    print(len(ctrl_depr_full.where(ctrl_depr_full['labels']==1).dropna()))
    ctrl_depr, ctrl_depr_held_out = train_test_split(ctrl_depr_full, test_size=0.3, random_state=1)
    util.dump_picke_file(ctrl_depr, 'paper_computed/ctrl_depr.p')
    util.dump_picke_file(ctrl_depr_held_out, 'paper_computed/ctrl_depr_held_out.p')
    ctrl_depr_full = None

    valid_users = get_user_pairs_for('ptsd')
    ctrl_ptsd_full = load_tweets_to_df(valid_labels=[0, 2], valid_users=valid_users)
    ctrl_ptsd_full['labels'] = (ctrl_ptsd_full['labels']/2).astype(int) #Convert label with 2 to 1
    ctrl_ptsd, ctrl_ptsd_held_out = train_test_split(ctrl_ptsd_full, test_size=0.3, random_state=1)
    util.dump_picke_file(ctrl_ptsd, 'paper_computed/ctrl_ptsd.p')
    util.dump_picke_file(ctrl_ptsd_held_out, 'paper_computed/ctrl_ptsd_held_out.p')

    return ctrl_depr, ctrl_depr_held_out, ctrl_ptsd, ctrl_ptsd_held_out


if __name__ == '__main__':
    create_data_sets()
