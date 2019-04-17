import sqlite3
import config

def get_connection():
    return sqlite3.connect(config.ACTIVE_MENTIONS_DB_PATH)


def get_depr_labelled_tweets(valid_labels=[1, 0]):
    conn = get_connection()
    c = conn.cursor()
    res = c.execute("SELECT * FROM tweets WHERE tweets.is_about_depression IN (" + ",".join([str(i) for i in valid_labels]) + ")").fetchall()
    conn.close()
    return res


def get_ptsd_labelled_tweets(valid_labels=[1, 0]):
    conn = get_connection()
    c = conn.cursor()
    res = c.execute("SELECT * FROM tweets WHERE tweets.is_about_ptsd IN (" + ",".join([str(i) for i in valid_labels]) + ")").fetchall()
    conn.close()
    return res