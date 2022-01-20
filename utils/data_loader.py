import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from utils.file_accessor import save_cv_indexes
from utils.data_split import load_config_file, get_train_test_split_index_by_subject, \
    get_cv_split_indexes_by_subject_emotion


def load_all_data(data_type: str = "audio_mfcc_100ms_50ms", emotion_type: str = "emotion"):
    trial_configs = load_config_file()
    data, labels = [], []
    for trial_filepath, trial_emotion in trial_configs[[f"{data_type}_filepath", emotion_type]].values:
        trial_filepath = os.path.join(os.getenv("PME4_DATA_PATH"), trial_filepath)
        if os.path.isfile(trial_filepath):
            data.append(np.load(trial_filepath))
            labels.append(trial_emotion)
        else:
            print("f{trial_filepath} - file not exist")
    data = np.array(data)
    labels = np.array(labels).reshape((-1, 1))
    return data, labels


def load_saved_cv_split_indexes_by_subject_emotion(cv_n=5):
    cv_indexes_path = os.path.join(os.getenv("PME4_DATA_PATH"), "cv_indexes")
    for cv_i in range(cv_n):
        cv_train_indexes = np.load(os.path.join(cv_indexes_path, f"cv{cv_i}_train_indexes.npy"))
        cv_test_indexes = np.load(os.path.join(cv_indexes_path, f"cv{cv_i}_test_indexes.npy"))
        yield cv_train_indexes, cv_test_indexes


def load_train_test_cv_split_data(data_type: str = "audio_mfcc_100ms_50ms", cv_n: int = 5, one_hot_encode: bool = True,
                                  presaved_index: bool = False, save_index: bool = False):
    emotion_type = "emotion" if one_hot_encode else "emotion_num"
    data, labels = load_all_data(data_type=data_type, emotion_type=emotion_type)
    if one_hot_encode:
        labels = one_hot_encode_label(labels)

    if presaved_index:
        for cv_train_indexes, cv_test_indexes in load_saved_cv_split_indexes_by_subject_emotion(cv_n=cv_n):
            yield data[cv_train_indexes], data[cv_test_indexes], labels[cv_train_indexes], labels[cv_test_indexes]
    else:
        cv_train_indexes, cv_test_indexes, _ = get_cv_split_indexes_by_subject_emotion(cv_n=cv_n)
        for cv_i in cv_train_indexes.keys():
            if save_index:
                save_cv_indexes(cv_i=cv_i, train_index=cv_train_indexes[cv_i], test_index=cv_test_indexes[cv_i])
            yield data[cv_train_indexes[cv_i]], data[cv_test_indexes[cv_i]], labels[cv_train_indexes[cv_i]], labels[
                cv_test_indexes[cv_i]]


def load_train_test_split_data(data_type: str = "audio_mfcc_100ms_50ms", test_size: float = 0.3, random_state: int = 42,
                               one_hot_encode: bool = True):
    train_indexes, test_indexes, trial_configs = get_train_test_split_index_by_subject(test_size=test_size,
                                                                                       random_state=random_state)
    emotion_type = "emotion" if one_hot_encode else "emotion_num"
    # load data and label
    train_data, test_data, train_labels, test_labels = [], [], [], []
    for trial_filepath, trial_emotion in trial_configs.loc[
        train_indexes, [f"{data_type}_filepath", emotion_type]].values:
        trial_filepath = os.path.join(os.getenv("PME4_DATA_PATH"), trial_filepath)
        if os.path.isfile(trial_filepath):
            trial_data = np.load(trial_filepath)
            train_data.append(trial_data)
            train_labels.append(trial_emotion)
        else:
            print("f{trial_filepath} - file not exist")
    train_data = np.array(train_data)
    train_labels = np.array(train_labels).reshape((-1, 1))

    for trial_filepath, trial_emotion in trial_configs.loc[
        test_indexes, [f"{data_type}_filepath", emotion_type]].values:
        trial_filepath = os.path.join(os.getenv("PME4_DATA_PATH"), trial_filepath)
        if os.path.isfile(trial_filepath):
            trial_data = np.load(trial_filepath)
            test_data.append(trial_data)
            test_labels.append(trial_emotion)
        else:
            print("f{trial_filepath} - file not exist")
    test_data = np.array(test_data)
    test_labels = np.array(test_labels).reshape((-1, 1))

    if one_hot_encode:
        train_labels, test_labels = one_hot_encode_labels(train_labels, test_labels)

    return train_data, test_data, train_labels, test_labels


def one_hot_encode_label(label):
    enc = OneHotEncoder(handle_unknown='ignore')
    label = enc.fit_transform(label).toarray()
    print(enc.get_feature_names())
    return label


def one_hot_encode_labels(train_labels, test_labels):
    enc = OneHotEncoder(handle_unknown='ignore')
    train_labels = enc.fit_transform(train_labels).toarray()
    test_labels = enc.fit_transform(test_labels).toarray()
    print(enc.get_feature_names())
    return train_labels, test_labels


