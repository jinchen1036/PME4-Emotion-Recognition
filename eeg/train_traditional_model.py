import numpy as np

from utils.bootstrap import get_bootstrap_data
from eeg.get_eeg_pca import get_eeg_pca_feature, get_eeg_pca_feature_cv, get_bootstrap_eeg_pca_feature
from model.traditional_models import train_traditional_model

def train_eeg_pca_traditional_model(data_type: str, test_size: float, num_pca_components: int = 50):
    train_data, test_data, train_labels, test_labels = get_eeg_pca_feature(
        data_type=data_type, test_size=test_size, one_hot_encode=False, num_pca_components=num_pca_components)

    traditional_model_result_df = train_traditional_model(data_type, train_data, test_data, train_labels, test_labels,
                                                          n_coef=num_pca_components)
    print(traditional_model_result_df)


def train_eeg_pca_traditional_model_cv(data_type: str, cv_n: int, num_pca_components: int = 50,
                                   presaved_index: bool = False, save_index: bool = False):
    cv_i = 0
    for train_data, test_data, train_labels, test_labels in get_eeg_pca_feature_cv(data_type=data_type,
                                                                                   cv_n=cv_n,
                                                                                   one_hot_encode=False,
                                                                                   num_pca_components=num_pca_components,
                                                                                   presaved_index=presaved_index,
                                                                                   save_index=save_index):
        traditional_model_result_df = train_traditional_model(data_type, train_data, test_data, train_labels,
                                                              test_labels, n_coef=num_pca_components)
        print(f"cv - {cv_i} - {data_type} pca\n", traditional_model_result_df)
        cv_i += 1


def train_bootstrap_eeg_pca_traditional_model(data_type: str, num_average: int, num_train_bootstrap_samples: int,
                                          num_test_bootstrap_samples: int, random_state: int,
                                          save_bootstrap_data: bool = True,
                                          num_pca_components: int = 50):
    train_data, test_data, train_labels, test_labels = get_bootstrap_eeg_pca_feature(
        data_type=data_type, num_average=num_average, num_train_bootstrap_samples=num_train_bootstrap_samples,
        num_test_bootstrap_samples=num_test_bootstrap_samples, random_state=random_state,
        one_hot_encode=False, save_bootstrap_data=save_bootstrap_data,
        num_pca_components=num_pca_components)

    traditional_model_result_df = train_traditional_model(data_type, train_data, test_data, train_labels, test_labels,
                                                          n_coef=num_pca_components)
    print(traditional_model_result_df)

def train_bootstrap_eeg_traditional_model(data_type: str, num_average: int, num_train_bootstrap_samples: int,
                                          num_test_bootstrap_samples: int, random_state: int,
                                          save_bootstrap_data: bool = True):
    train_bootstrap_data, test_bootstrap_data, train_labels, test_labels = get_bootstrap_data(data_type=data_type,
                                                                                              num_average=num_average,
                                                                                              num_train_bootstrap_samples=num_train_bootstrap_samples,
                                                                                              random_state=random_state,
                                                                                              one_hot_encode=False,
                                                                                              save_bootstrap_data=save_bootstrap_data,
                                                                                              num_test_bootstrap_samples=num_test_bootstrap_samples)

    (_, _, _, number_channels, num_features) = train_bootstrap_data.shape
    train_bootstrap_data = np.reshape(train_bootstrap_data, (-1, number_channels, num_features))
    test_bootstrap_data = np.reshape(test_bootstrap_data, (-1, number_channels, num_features))
    traditional_model_result_df = train_traditional_model(data_type, train_bootstrap_data, test_bootstrap_data,
                                                          train_labels, test_labels,
                                                          n_coef=None)
    print(traditional_model_result_df)