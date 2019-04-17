import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import active_mentions_filtering.active_mentions_filter as amf
import numpy as np
import util
import config

def add_depr_and_mental_health_tweet_probs(df, valid_users):

    df['tweet_count'] = [len(tweets) for tweets in df.tweets]
    print('Users with zero tweets removed: ', (df['tweet_count'] == 0).sum())
    df = df.where(df['tweet_count'] > 0).dropna()

    model = amf.get_depression_model(valid_users=valid_users)
    probs = []
    for index, row in df.iterrows():
        print(index)
        probs.append(model.predict_proba(row['tweets'])[:, 1])
    df['depr_probs'] = probs
    model = amf.get_depr_mental_helth_model(valid_users=valid_users)
    probs = []
    for index, row in df.iterrows():
        print(index)
        probs.append(model.predict_proba(row['tweets'])[:, 1])
    df['mental_health_probs'] = probs

    # df['depr_tweet_perc'] = [(row['depr_probs'] > 0.5).sum()/len(row['tweets']) for index, row in df.iterrows()]
    # df['non_depr_tweets'] = [row['tweets'][row['depr_probs'] < 0.5] for index, row in df.iterrows()]
    # df['non_depr_cmu_pos_tags'] = [np.array(row['cmu_pos_tags'])[row['depr_probs'] < 0.5] for index, row in df.iterrows()]
    # df['non_depr_nltk_pos_tags'] = [np.array(row['nltk_pos_tags'])[row['depr_probs'] < 0.5] for index, row in df.iterrows()]
    # df['mh_tweet_perc'] = [(row['mental_health_probs'] > 0.5).sum()/len(row['tweets']) for index, row in df.iterrows()]
    # df['non_mh_tweets'] = [row['tweets'][row['mental_health_probs'] < 0.5] for index, row in df.iterrows()]
    # df['non_mh_cmu_pos_tags'] = [np.array(row['cmu_pos_tags'])[row['mental_health_probs'] < 0.5] for index, row in df.iterrows()]
    # df['non_mh_nltk_pos_tags'] = [np.array(row['nltk_pos_tags'])[row['mental_health_probs'] < 0.5] for index, row in df.iterrows()]

    return df



def add_ptsd_and_mental_health_tweet_probs(df, valid_users):
    df['tweet_count'] = [len(tweets) for tweets in df.tweets]
    print('Users with zero tweets removed: ', (df['tweet_count'] == 0).sum())
    df = df.where(df['tweet_count'] > 0).dropna()

    model = amf.get_ptsd_model(valid_users=valid_users)
    probs = []
    for index, row in df.iterrows():
        print(index)
        probs.append(model.predict_proba(row['tweets'])[:, 1])
    df['ptsd_probs'] = probs

    model = amf.get_ptsd_mh_model(valid_users=valid_users)
    probs = []
    for index, row in df.iterrows():
        print(index)
        probs.append(model.predict_proba(row['tweets'])[:, 1])
    df['mental_health_probs'] = probs

    return df


if __name__ == '__main__':
    # ctrl_depr = util.load_picke_file(config.CTRL_DEPR_DF)
    # valid_users = ctrl_depr.index.values

    # ctrl_depr = add_depr_and_mental_health_tweet_probs(ctrl_depr, valid_users=valid_users)
    # util.dump_picke_file(ctrl_depr, config.CTRL_DEPR_DF)
    # ctrl_depr = None

    # ctrl_depr_held_out = util.load_picke_file(config.CTRL_DEPR_HELD_OUT_DF)
    # ctrl_depr_held_out = add_depr_and_mental_health_tweet_probs(ctrl_depr_held_out, valid_users=valid_users)
    # util.dump_picke_file(ctrl_depr_held_out, config.CTRL_DEPR_HELD_OUT_DF)
    # ctrl_depr_held_out=None


    ctrl_ptsd = util.load_picke_file(config.CTRL_PTSD_DF)
    valid_users = None #ctrl_ptsd.index.values

    ctrl_ptsd = add_ptsd_and_mental_health_tweet_probs(ctrl_ptsd, valid_users=valid_users)
    util.dump_picke_file(ctrl_ptsd, config.CTRL_PTSD_DF)

    ctrl_ptsd_held_out = util.load_picke_file(config.CTRL_PTSD_HELD_OUT_DF)
    ctrl_ptsd_held_out = add_ptsd_and_mental_health_tweet_probs(ctrl_ptsd_held_out, valid_users=valid_users)
    util.dump_picke_file(ctrl_ptsd_held_out, config.CTRL_PTSD_HELD_OUT_DF)