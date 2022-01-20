import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

def load_config_file():
    return pd.read_csv(os.path.join(os.getenv("PME4_DATA_PATH"), os.getenv("CONFIG_FILENAME")))

def get_train_test_split_index_by_subject(test_size=0.3, random_state = 42):
    trial_configs = load_config_file()
    train_indexes, test_indexes = [], []
    for subject, trial in trial_configs.groupby(['subject']):
        id_train, id_test = train_test_split(trial.index,stratify= trial['emotion_num'],
                                             test_size=test_size, random_state = random_state)
        train_indexes.extend(list(id_train))
        test_indexes.extend(list(id_test))
    return train_indexes, test_indexes, trial_configs

def get_cv_split_indexes(trial_indexes, stratify, n_splits):
    train_indexes, test_indexes = [], []
    skf = StratifiedKFold(n_splits=n_splits, random_state=int(time.time()), shuffle=True)
    for train_index, test_index in skf.split(trial_indexes, stratify):
        train_indexes.append(trial_indexes[train_index])
        test_indexes.append(trial_indexes[test_index])
    return train_indexes, test_indexes

def get_cv_split_indexes_by_subject_emotion(cv_n = 5):
    trial_configs = load_config_file()
    cv_train_indexes, cv_test_indexes = {}, {}
    for i in range(cv_n):
        cv_train_indexes[i] = []
        cv_test_indexes[i] = []

    skf = StratifiedKFold(n_splits=cv_n, random_state=int(time.time()), shuffle=True)
    for subject, trial in trial_configs.groupby(['subject']):
        cv_i = 0
        for train_indexes, test_indexes in skf.split(trial.index,trial['emotion_num']):
            cv_train_indexes[cv_i].extend(trial.index[train_indexes])
            cv_test_indexes[cv_i].extend(trial.index[test_indexes])
            cv_i += 1
    return cv_train_indexes, cv_test_indexes, trial_configs


