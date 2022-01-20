import time
from utils.data_loader import load_train_test_split_data, load_train_test_cv_split_data
from model.traditional_models import train_traditional_model


def train_audio_traditional_model(data_type: str, test_size: float):
    train_data, test_data, train_labels, test_labels = load_train_test_split_data(
        data_type=data_type, test_size=test_size, one_hot_encode=False, random_state=int(time.time()))
    traditional_model_result_df = train_traditional_model(data_type, train_data, test_data, train_labels, test_labels,
                                                          n_coef=None)
    print(traditional_model_result_df)


def train_audio_traditional_model_cv(data_type: str, cv_n: int, presaved_index: bool = False, save_index: bool = False):
    cv_i = 0
    for train_data, test_data, train_labels, test_labels in load_train_test_cv_split_data(data_type=data_type,
                                                                                          cv_n=cv_n,
                                                                                          one_hot_encode=False,
                                                                                          presaved_index=presaved_index,
                                                                                          save_index=save_index):
        traditional_model_result_df = train_traditional_model(data_type, train_data, test_data, train_labels,
                                                              test_labels, n_coef=None)
        print(f"cv - {cv_i} - {data_type}\n", traditional_model_result_df)
        cv_i += 1
