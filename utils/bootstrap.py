import os
import time
import numpy as np
import pandas as pd
from utils.settings import Settings
from utils.file_accessor import check_dirpath, load_pickle_file, save_pickle_file
from utils.data_split import load_config_file
from utils.data_loader import load_all_data


def get_bootstrap_data(data_type: str, num_average: int, num_train_bootstrap_samples: int, random_state: int,
                       one_hot_encode: bool = True, save_bootstrap_data: bool = True, num_test_bootstrap_samples: int = 100):
    bootstrap_indexes_filename = f"bootstrap_random{random_state}_{num_train_bootstrap_samples}sample_per_emotion_{num_average}average"
    bootstrap_train_data_filename = os.path.join(os.getenv("PME4_DATA_PATH"), "bootstrap_files",
                                                 f"{data_type}_{bootstrap_indexes_filename}_train.npy")
    if os.path.isfile(bootstrap_train_data_filename):
        return load_saved_bootstrap_data(data_type=data_type, num_average=num_average,
                                         num_train_bootstrap_samples=num_train_bootstrap_samples,
                                         random_state=random_state, one_hot_encode=one_hot_encode)
    else:
        return bootstrap_data(data_type=data_type, num_average=num_average,
                              num_train_bootstrap_samples=num_train_bootstrap_samples,
                              num_test_bootstrap_samples=num_test_bootstrap_samples,
                              random_state=random_state, save_bootstrap_data=save_bootstrap_data, one_hot_encode=one_hot_encode)


def load_saved_bootstrap_data(data_type: str, num_average: int, num_train_bootstrap_samples: int, random_state: int,
                              one_hot_encode: bool = True):
    bootstrap_indexes_filename = f"bootstrap_random{random_state}_{num_train_bootstrap_samples}sample_per_emotion_{num_average}average"
    bootstrap_train_data = np.load(file=os.path.join(os.getenv("PME4_DATA_PATH"), "bootstrap_files",
                                                     f"{data_type}_{bootstrap_indexes_filename}_train.npy"))
    bootstrap_test_data = np.load(file=os.path.join(os.getenv("PME4_DATA_PATH"), "bootstrap_files",
                                                    f"{data_type}_{bootstrap_indexes_filename}_test.npy"))

    bootstrap_train_label, bootstrap_test_label = get_bootstrap_label(bootstrap_train_data, bootstrap_test_data,
                                                                      one_hot_encode=one_hot_encode)
    return bootstrap_train_data, bootstrap_test_data, bootstrap_train_label, bootstrap_test_label


def get_bootstrap_label(train_data, test_data, one_hot_encode):
    # getting label
    (num_sub, num_emo, num_train_bootstrap) = train_data.shape[:3]
    num_test_bootstrap = test_data.shape[2]

    train_label = np.zeros((num_sub, num_emo, num_train_bootstrap), dtype=np.int32)
    test_label = np.zeros((num_sub, num_emo, num_test_bootstrap), dtype=np.int32)
    for i in range(num_emo):
        train_label[:, i, :] = i
        test_label[:, i, :] = i
    train_label, test_label = process_label(train_label, test_label, one_hot_encode)
    return train_label, test_label


def bootstrap_data(data_type: str, num_average=20, num_train_bootstrap_samples=100, num_test_bootstrap_samples=100,
                   random_state=None, save_bootstrap_data: bool = True, one_hot_encode: bool = True):
    check_dirpath(os.path.join(os.getenv("PME4_DATA_PATH"), "bootstrap_files"))
    bootstrap_indexes_filename = f"bootstrap_random{random_state}_{num_train_bootstrap_samples}sample_per_emotion_{num_average}average"
    trial_configs = load_config_file()
    # get bootstrap index
    if not os.path.isfile(
            os.path.join(os.getenv("PME4_DATA_PATH"), "bootstrap_files", f"{bootstrap_indexes_filename}_indexes.pkl")):
        bootstrap_record = {}
        random_state = int(time.time()) if random_state is None else random_state
        bootstrap_indexes_filename = f"bootstrap_random{random_state}_{num_train_bootstrap_samples}sample_per_emotion_{num_average}average"
        for subject_index, data in trial_configs.groupby(['subject']):
            bootstrap_record[subject_index] = {}
            for emotion, emotion_data in data.groupby(['emotion']):
                trial_index = np.random.RandomState(random_state).permutation(emotion_data.index)
                num_test_samples = len(emotion_data) // 2
                num_train_samples = len(emotion_data) - num_test_samples
                bootstrap_record[subject_index][emotion] = {
                    'train_config_rows': trial_index[:num_train_samples],
                    'test_config_rows': trial_index[num_train_samples:],
                    'subject_train_trials': emotion_data.loc[trial_index[:num_train_samples], 'trial'].values,
                    'subject_test_trials': emotion_data.loc[trial_index[num_train_samples:], 'trial'].values,
                    'train_index_groups': unique_group_index(num_average, num_train_samples,
                                                             num_train_bootstrap_samples),
                    'test_index_groups': unique_group_index(num_average, num_test_samples, num_test_bootstrap_samples)
                }
        save_pickle_file(filename=os.path.join(os.getenv("PME4_DATA_PATH"), "bootstrap_files",
                                               f"{bootstrap_indexes_filename}_indexes.pkl"),
                         data=bootstrap_record)
    else:
        bootstrap_record = load_pickle_file(filename=os.path.join(os.getenv("PME4_DATA_PATH"), "bootstrap_files",
                                                                  f"{bootstrap_indexes_filename}_indexes.pkl"))

    settings = Settings()
    sample_filepath = trial_configs.loc[0, f"{data_type}_filepath"]
    num_channels = settings.num_eeg_channel if "eeg" in sample_filepath else settings.num_emg_channel
    fs = 5000 if "5kHz" in sample_filepath else 1000

    # get bootstrap data
    bootstrap_train_data = np.zeros((settings.num_subjects, settings.num_emotions, num_train_bootstrap_samples,
                                     num_channels, int(fs * settings.trial_time)), dtype=np.float32)
    bootstrap_test_data = np.zeros((settings.num_subjects, settings.num_emotions, num_test_bootstrap_samples,
                                    num_channels, int(fs * settings.trial_time)), dtype=np.float32)

    for subject_index, bootstrap_indexes in bootstrap_record.items():
        data, labels = load_all_data(data_type=data_type, emotion_type="emotion_num")
        i = 0
        for emotion, bootstrap_index in bootstrap_indexes.items():
            print(f"Process Subject {subject_index} - Emotion {emotion}")
            train_trials = data[bootstrap_index['train_config_rows']]
            test_trials = data[bootstrap_index['test_config_rows']]

            for b_i, indexes in enumerate(bootstrap_index['train_index_groups']):
                bootstrap_train_data[subject_index - 1, i, b_i] = np.mean(train_trials[indexes], axis=0)
            for b_i, indexes in enumerate(bootstrap_index['test_index_groups']):
                bootstrap_test_data[subject_index - 1, i, b_i] = np.mean(test_trials[indexes], axis=0)

            i += 1
            pass

    if save_bootstrap_data:
        np.save(file=os.path.join(os.getenv("PME4_DATA_PATH"), "bootstrap_files",
                                  f"{data_type}_{bootstrap_indexes_filename}_train.npy"),
                arr=bootstrap_train_data)
        np.save(file=os.path.join(os.getenv("PME4_DATA_PATH"), "bootstrap_files",
                                  f"{data_type}_{bootstrap_indexes_filename}_test.npy"),
                arr=bootstrap_test_data)

    bootstrap_train_label, bootstrap_test_label = get_bootstrap_label(bootstrap_train_data, bootstrap_test_data,
                                                                      one_hot_encode=one_hot_encode)
    return bootstrap_train_data, bootstrap_test_data, bootstrap_train_label, bootstrap_test_label


def unique_group_index(group_size, total_size, number_group):
    groups = []
    while len(groups) < number_group:
        random_group = set(np.random.choice(total_size, group_size, replace=False))
        if random_group not in groups:
            groups.append(list(random_group))
    return groups


def process_label(train_label, test_label, one_hot_encode):
    train_label = np.reshape(train_label, (-1, 1))
    test_label = np.reshape(test_label, (-1, 1))

    if one_hot_encode:
        from sklearn.preprocessing import OneHotEncoder
        train_label = OneHotEncoder().fit_transform(train_label).toarray()
        test_label = OneHotEncoder().fit_transform(test_label).toarray()
    else:
        train_label = np.reshape(train_label, (-1,))
        test_label = np.reshape(test_label, (-1,))
    return train_label, test_label
