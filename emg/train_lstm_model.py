import os
import time
from utils.settings import Settings
from utils.data_formater import format_emg_mfcc_lstm
from utils.data_loader import load_train_test_split_data, load_train_test_cv_split_data
from model.lstm_model import LSTM


def train_emg_lstm(data_type: str, settings: Settings, test_size: float):
    train_data, test_data, train_labels, test_labels = load_train_test_split_data(
        data_type=data_type, test_size=test_size, one_hot_encode=True, random_state=int(time.time()))

    train_data, test_data, train_labels, test_labels = format_emg_mfcc_lstm(train_data=train_data,
                                                                            train_labels=train_labels,
                                                                            test_data=test_data,
                                                                            test_labels=test_labels)
    (train_size, num_timesteps, num_features) = train_data.shape
    (test_size, num_classes) = test_labels.shape


    model = LSTM(timesteps=num_timesteps, num_features=num_features, num_classes=num_classes,
                 name_prefix=f"mfcc_{num_timesteps}timesteps", data_type=data_type,
                 model_dir=os.path.join(os.getenv("PME4_DATA_PATH"), "model"),
                 model_num=1)

    model.build_model(lr=settings.learning_rate, lstm_units=settings.lstm_units,
                      l1_value=settings.l1, l2_value=settings.l2,
                      dropout=settings.dropout, isCuDNN=settings.isCuDNN)

    model.fit_model_save_epoch(train_data, train_labels, test_data, test_labels,
                               saved_epochs=settings.saved_epochs,
                               bs=settings.batch_size, init_epoch=0)


def train_emg_lstm_cv(data_type: str, settings: Settings, cv_n: int,
                      presaved_index: bool = False, save_index: bool = False):
    cv_i = 0
    for train_data, test_data, train_labels, test_labels in load_train_test_cv_split_data(data_type=data_type,
                                                                                          cv_n=cv_n,
                                                                                          one_hot_encode=True,
                                                                                          presaved_index=presaved_index,
                                                                                          save_index=save_index):
        train_data, test_data, train_labels, test_labels = format_emg_mfcc_lstm(train_data=train_data,
                                                                                train_labels=train_labels,
                                                                                test_data=test_data,
                                                                                test_labels=test_labels)
        (train_size, num_timesteps, num_features) = train_data.shape
        (test_size, num_classes) = test_labels.shape


        model = LSTM(timesteps=num_timesteps, num_features=num_features, num_classes=num_classes,
                     name_prefix=f"cv{cv_i}_mfcc_{num_timesteps}timesteps", data_type=data_type,
                     model_dir=os.path.join(os.getenv("PME4_DATA_PATH"), "model"),
                     model_num=1)

        model.build_model(lr=settings.learning_rate, lstm_units=settings.lstm_units,
                          l1_value=settings.l1, l2_value=settings.l2,
                          dropout=settings.dropout, isCuDNN=settings.isCuDNN)
        print(model.model.summary())
        model.fit_model_save_epoch(train_data, train_labels, test_data, test_labels,
                                   saved_epochs=settings.saved_epochs,
                                   bs=settings.batch_size, init_epoch=0)
        cv_i += 1
        del model
