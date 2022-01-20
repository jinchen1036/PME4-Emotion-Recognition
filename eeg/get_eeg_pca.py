import time
import numpy as np
from sklearn.decomposition import PCA
from utils.bootstrap import get_bootstrap_data
from utils.data_loader import load_train_test_split_data, load_train_test_cv_split_data


def get_eeg_pca_feature(data_type: str, test_size: float, num_pca_components: int = 50,one_hot_encode: bool = True):
    train_data, test_data, train_labels, test_labels = load_train_test_split_data(
        data_type=data_type, test_size=test_size, one_hot_encode=one_hot_encode, random_state=int(time.time()))
    (num_train_sample, number_channels, num_features) = train_data.shape

    train_pca = np.zeros((num_train_sample, number_channels, num_pca_components), dtype=np.float32)
    test_pca = np.zeros((test_data.shape[0], number_channels, num_pca_components), dtype=np.float32)

    for channel in range(number_channels):
        pca = PCA(n_components=num_pca_components)
        train_channel = np.reshape(train_data[:, channel, :], (-1, train_data.shape[-1]))
        train_pca[:, channel, :] = pca.fit_transform(train_channel)
        test_pca[:, channel, :] = pca.transform(np.reshape(test_data[:, channel, :], (-1, test_data.shape[-1])))

    return train_pca, test_pca, train_labels, test_labels


def get_eeg_pca_feature_cv(data_type: str, cv_n: int, num_pca_components: int = 50,
                           presaved_index: bool = False, save_index: bool = False,one_hot_encode: bool = True):
    cv_i = 0
    for train_data, test_data, train_labels, test_labels in load_train_test_cv_split_data(data_type=data_type,
                                                                                          cv_n=cv_n,
                                                                                          one_hot_encode=one_hot_encode,
                                                                                          presaved_index=presaved_index,
                                                                                          save_index=save_index):
        (num_train_sample, number_channels, num_features) = train_data.shape

        train_pca = np.zeros((num_train_sample, number_channels, num_pca_components), dtype=np.float32)
        test_pca = np.zeros((test_data.shape[0], number_channels, num_pca_components), dtype=np.float32)

        for channel in range(number_channels):
            pca = PCA(n_components=num_pca_components)
            train_channel = np.reshape(train_data[:, channel, :], (-1, train_data.shape[-1]))
            train_pca[:, channel, :] = pca.fit_transform(train_channel)
            test_pca[:, channel, :] = pca.transform(np.reshape(test_data[:, channel, :], (-1, test_data.shape[-1])))

        yield train_pca, test_pca, train_labels, test_labels
        cv_i += 1

def get_bootstrap_eeg_pca_feature(data_type: str, num_average: int, num_train_bootstrap_samples: int,
                                  num_test_bootstrap_samples: int, random_state: int,
                                  one_hot_encode: bool = True, save_bootstrap_data: bool = True,
                                  num_pca_components: int = 50):
    train_bootstrap_data, test_bootstrap_data, train_labels, test_labels = get_bootstrap_data(data_type=data_type,
                                                                                              num_average=num_average,
                                                                                              num_train_bootstrap_samples=num_train_bootstrap_samples,
                                                                                              random_state=random_state,
                                                                                              one_hot_encode=one_hot_encode,
                                                                                              save_bootstrap_data=save_bootstrap_data,
                                                                                              num_test_bootstrap_samples=num_test_bootstrap_samples)

    (num_sub, num_emo, num_train_bootstrap, number_channels, num_features) = train_bootstrap_data.shape
    (_, _, num_test_bootstrap, _, _) = test_bootstrap_data.shape
    train_bootstrap_pca = np.zeros((num_sub * num_emo * num_train_bootstrap, number_channels, num_pca_components),
                                   dtype=np.float32)
    test_bootstrap_pca = np.zeros((num_sub * num_emo * num_test_bootstrap, number_channels, num_pca_components),
                                  dtype=np.float32)

    for channel in range(number_channels):
        pca = PCA(n_components=num_pca_components)
        train_channel = np.reshape(train_bootstrap_data[:, :, :, channel, :], (-1, train_bootstrap_data.shape[-1]))
        train_bootstrap_pca[:, channel, :] = pca.fit_transform(train_channel)
        test_bootstrap_pca[:, channel, :] = pca.transform(
            np.reshape(test_bootstrap_data[:, :, :, channel, :], (-1, test_bootstrap_data.shape[-1])))

    return train_bootstrap_pca, test_bootstrap_pca, train_labels, test_labels
