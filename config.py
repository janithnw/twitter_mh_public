import sys
import os


PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

TRAINING_TWEETS_PATH = PROJECT_PATH + '/data/tweets/'
TRAINING_LABELS_PATH = PROJECT_PATH + '/data/anonymized_user_info_by_chunk_training.csv'
LABEL_IDS = {'control': 0, 'depression': 1, 'ptsd': 2, 'condition': -1, '?': '?'}

STOPWORDS_PATH = PROJECT_PATH + '/data/stopwords.txt'

ACTIVE_MENTIONS_DB_PATH = PROJECT_PATH + '/active_mentions_filtering/tweets.db'
ACTIVE_MENTIONS_DEPR_MODEL_PATH = PROJECT_PATH + '/active_mentions_filtering/depr_model.p'
ACTIVE_MENTIONS_DEPR_MH_MODEL_PATH = PROJECT_PATH + '/active_mentions_filtering/depr_mh_model.p'
ACTIVE_MENTIONS_PTSD_MODEL_PATH = PROJECT_PATH + '/active_mentions_filtering/ptsd_model.p'
ACTIVE_MENTIONS_PTSD_MH_MODEL_PATH = PROJECT_PATH + '/active_mentions_filtering/ptsd_mh_model.p'

SOC_ESSAYS_PATH = PROJECT_PATH + '/data/soc_essays/essays_anon_full.csv'

CTRL_DEPR_DF = PROJECT_PATH + '/computed/ctrl_depr.p'
CTRL_DEPR_HELD_OUT_DF = PROJECT_PATH + '/computed/ctrl_depr_held_out.p'
CTRL_PTSD_DF = PROJECT_PATH + '/computed/ctrl_ptsd.p'
CTRL_PTSD_HELD_OUT_DF = PROJECT_PATH + '/computed/ctrl_ptsd_held_out.p'

CTRL_DEPR_FILTERED_DF = PROJECT_PATH + '/computed/ctrl_depr_filtered.p'
CTRL_DEPR_HELD_OUT_FILTERED_DF = PROJECT_PATH + '/computed/ctrl_depr_held_out_filtered.p'
CTRL_PTSD_FILTERED_DF = PROJECT_PATH + '/computed/ctrl_ptsd_filtered.p'
CTRL_PTSD_HELD_OUT_FILTERED_DF = PROJECT_PATH + '/computed/ctrl_ptsd_held_out_filtered.p'

CTRL_DEPR_VOCAB_PATH = PROJECT_PATH + '/computed/ctrl_depr_vocab.p'
CTRL_PTSD_VOCAB_PATH = PROJECT_PATH + '/computed/ctrl_ptsd_vocab.p'

CTRL_DEPR_TOPIC_PRIORS_INPUT = PROJECT_PATH + '/computed/slda_models/ctrl_depr_priors/ctrl_depr_priors_input.npy'
CTRL_DEPR_TOPIC_PRIORS_MODEL = PROJECT_PATH + '/computed/slda_models/ctrl_depr_priors/ctrl_depr_priors_model.npy'


CTRL_PTSD_TOPIC_PRIORS_INPUT = PROJECT_PATH + '/computed/slda_models/ctrl_ptsd_priors/ctrl_ptsd_priors_input.npy'
CTRL_PTSD_TOPIC_PRIORS_MODEL = PROJECT_PATH + '/computed/slda_models/ctrl_ptsd_priors/ctrl_ptsd_priors_model.npy'


SLDA_MODELS_DIR = PROJECT_PATH + '/computed/slda_models/trained_models/'