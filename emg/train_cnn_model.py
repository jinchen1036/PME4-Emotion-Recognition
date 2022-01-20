import os
import time
from utils.settings import Settings
from utils.data_formater import format_emg_mfcc_cnn
from utils.data_loader import load_train_test_split_data, load_train_test_cv_split_data
from model.cnn_model import CNN

def train_emg_cnn(data_type: str, settings: Settings, test_size: float):
    train_data, test_data, train_labels, test_labels = load_train_test_split_data(
        data_type=data_type, test_size=test_size, one_hot_encode=True, random_state=int(time.time()))

    train_data, test_data, train_labels, test_labels = format_emg_mfcc_cnn(train_data=train_data,
                                                                           train_labels=train_labels,
                                                                           test_data=test_data,
                                                                           test_labels=test_labels)
    (train_size, num_features, num_channels) = train_data.shape
    (test_size,num_classes) = test_labels.shape

    model = CNN(num_classes=num_classes, name_prefix=f"{num_features}mfcc", data_type=data_type,
                 model_dir=os.path.join(os.getenv("PME4_DATA_PATH"), "model"), sample=1)
    model.build_model(input_shape=(train_data.shape[1], train_data.shape[2]),
                      denses=[16],
                      filters=[8, 8, 16, 32],
                      str=[2, 2, 2, 2],
                      ks=3, ps=1, dropout=0.2, lr=1e-4, bn=True, regularize=False)
    model.fit_model_save_epoch(train_data, train_labels, test_data, test_labels,
                               saved_epochs=settings.saved_epochs,
                               bs=settings.batch_size, init_epoch=0)


def train_emg_cnn_cv(data_type: str, settings: Settings, cv_n: int, presaved_index: bool = False, save_index: bool = False):
    cv_i = 0
    for train_data, test_data, train_labels, test_labels in load_train_test_cv_split_data(data_type=data_type,
                                                                                          cv_n=cv_n,
                                                                                          one_hot_encode=True,
                                                                                          presaved_index=presaved_index,
                                                                                          save_index=save_index):
        train_data, test_data, train_labels, test_labels = format_emg_mfcc_cnn(train_data=train_data,
                                                                                train_labels=train_labels,
                                                                                test_data=test_data,
                                                                                test_labels=test_labels)
        (train_size, num_features, num_channels) = train_data.shape
        (test_size, num_classes) = test_labels.shape


        model = CNN(num_classes=num_classes, name_prefix=f"cv{cv_i}_{num_features}mfcc", data_type=data_type,
                    model_dir=os.path.join(os.getenv("PME4_DATA_PATH"), "model"), sample=1)
        model.build_model(input_shape=(train_data.shape[1], train_data.shape[2]),
                          denses=[16],
                          filters=[8, 8, 16, 32],
                          str=[2, 2, 2, 2],
                          ks=3, ps=1, dropout=0.2, lr=1e-4, bn=True, regularize=False)
        model.fit_model_save_epoch(train_data, train_labels, test_data, test_labels,
                                   saved_epochs=settings.saved_epochs,
                                   bs=settings.batch_size, init_epoch=0)

        cv_i += 1
