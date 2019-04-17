import pickle
import config


def load_picke_file(path):
    return pickle.load(open(path, "rb"))


def dump_picke_file(obj, path):
    return pickle.dump(obj, open(path, "wb"))


def get_stopwords():
    with open(config.STOPWORDS_PATH) as f:
        stopwords = f.readlines()
        return [x.strip() for x in stopwords]