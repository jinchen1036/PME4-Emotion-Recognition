import os
import time
from utils.settings import Settings
from utils.data_loader import load_train_test_split_data, load_train_test_cv_split_data
from model.lstm_model import LSTM


def train_audio_lstm(data_type: str, settings: Settings, test_size: float, number_model_to_trained: int = 1):
    train_data, test_data, train_labels, test_labels = load_train_test_split_data(
        data_type=data_type, test_size=test_size, one_hot_encode=True, random_state=int(time.time()))

    (train_size, timesteps, num_features) = train_data.shape
    (_, num_classes) = train_labels.shape

    for model_num in range(1, number_model_to_trained + 1):
        model = LSTM(timesteps=timesteps, num_features=num_features, num_classes=num_classes,
                     name_prefix=f"{timesteps}timesteps", data_type=data_type,
                     model_dir=os.path.join(os.getenv("PME4_DATA_PATH"), "model"),
                     model_num=model_num)

        model.build_model(lr=settings.learning_rate, lstm_units=settings.lstm_units,
                          l1_value=settings.l1, l2_value=settings.l2,
                          dropout=settings.dropout, isCuDNN=settings.isCuDNN)
        # model.save_model_structure()
        # print(model.model.summary())
        model.fit_model_save_epoch(train_data, train_labels, test_data, test_labels,
                                   saved_epochs=settings.saved_epochs,
                                   bs=settings.batch_size, init_epoch=0)


def train_audio_lstm_cv(data_type: str, settings: Settings, cv_n: int, number_model_to_trained: int = 1, presaved_index: bool = False, save_index: bool = False):
    cv_i = 0
    for train_data, test_data, train_labels, test_labels in load_train_test_cv_split_data(data_type=data_type,
                                                                                          cv_n=cv_n,
                                                                                          one_hot_encode=True,
                                                                                          presaved_index=presaved_index,
                                                                                          save_index=save_index):
        (train_size, timesteps, num_features) = train_data.shape
        (_, num_classes) = train_labels.shape

        for model_num in range(1, number_model_to_trained + 1):
            model = LSTM(timesteps=timesteps, num_features=num_features, num_classes=num_classes,
                         name_prefix=f"cv{cv_i}_{timesteps}timesteps", data_type=data_type,
                         model_dir=os.path.join(os.getenv("PME4_DATA_PATH"), "model"),
                         model_num=model_num)

            model.build_model(lr=settings.learning_rate, lstm_units=settings.lstm_units,
                              l1_value=settings.l1, l2_value=settings.l2,
                              dropout=settings.dropout, isCuDNN=settings.isCuDNN)
            # model.save_model_structure()
            print(model.model.summary())
            model.fit_model_save_epoch(train_data, train_labels, test_data, test_labels,
                                       saved_epochs=settings.saved_epochs,
                                       bs=settings.batch_size, init_epoch=0)
        cv_i += 1
        del model
