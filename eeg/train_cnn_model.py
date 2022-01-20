import os
from utils.settings import Settings
from utils.bootstrap import get_bootstrap_data
from eeg.get_eeg_pca import get_eeg_pca_feature, get_eeg_pca_feature_cv, get_bootstrap_eeg_pca_feature
from model.cnn_model import CNN

def train_eeg_pca_cnn(data_type: str, settings: Settings, test_size: float, num_pca_components: int = 50):
    train_data, test_data, train_labels, test_labels = get_eeg_pca_feature(
    data_type=data_type, test_size=test_size, one_hot_encode=True, num_pca_components=num_pca_components)

    (train_size, number_channels, num_pca_components) = train_data.shape
    (_, num_classes) = train_labels.shape

    model = CNN(num_classes=num_classes, name_prefix=f"{num_pca_components}pca", data_type=data_type,
                 model_dir=os.path.join(os.getenv("PME4_DATA_PATH"), "model"), sample=1)
    model.build_model(input_shape=(train_data.shape[1], train_data.shape[2]),
                      denses=[], filters=[16, 32, 64],str=[2,2,2],
                      ks=4, ps=2, dropout=0.2, lr=1e-5, bn=False, regularize=False)
    model.fit_model_save_epoch(train_data, train_labels, test_data, test_labels,
                               saved_epochs=settings.saved_epochs,
                               bs=settings.batch_size, init_epoch=0)


def train_eeg_pca_cnn_cv(data_type: str, settings: Settings, cv_n: int, num_pca_components: int = 50, presaved_index: bool = False, save_index: bool = False):
    cv_i = 0
    for train_data, test_data, train_labels, test_labels in get_eeg_pca_feature_cv(data_type=data_type,
                                                                                   cv_n=cv_n,
                                                                                   one_hot_encode=True,
                                                                                   num_pca_components=num_pca_components,
                                                                                   presaved_index=presaved_index,
                                                                                   save_index=save_index):
        (train_size,  number_channels, num_pca_components) = train_data.shape
        (_, num_classes) = train_labels.shape

        model = CNN(num_classes=num_classes, name_prefix=f"cv{cv_i}_{number_channels}pca", data_type=data_type,
                    model_dir=os.path.join(os.getenv("PME4_DATA_PATH"), "model"), sample=1)
        model.build_model(input_shape=(train_data.shape[1], train_data.shape[2]),
                          denses=[], filters=[16, 32, 64], str=[2, 2, 2],
                          ks=4, ps=2, dropout=0.2, lr=1e-5, bn=False, regularize=False)
        model.fit_model_save_epoch(train_data, train_labels, test_data, test_labels,
                                   saved_epochs=settings.saved_epochs,
                                   bs=settings.batch_size, init_epoch=0)

        cv_i += 1

def train_bootstrap_eeg_pca_cnn(data_type: str, settings: Settings, num_average: int, num_train_bootstrap_samples: int,
                                          num_test_bootstrap_samples: int, random_state: int,
                                          save_bootstrap_data: bool = True,
                                          num_pca_components: int = 50):
    train_data, test_data, train_labels, test_labels = get_bootstrap_eeg_pca_feature(
        data_type=data_type, num_average=num_average, num_train_bootstrap_samples=num_train_bootstrap_samples,
        num_test_bootstrap_samples=num_test_bootstrap_samples, random_state=random_state,
        one_hot_encode=False, save_bootstrap_data=save_bootstrap_data,
        num_pca_components=num_pca_components)

    bootstrap_indexes_filename = f"bootstrap_random{random_state}_{num_train_bootstrap_samples}sample_per_emotion_{num_average}average"

    (train_size, number_channels, num_pca_components) = train_data.shape
    (_, num_classes) = train_labels.shape

    model = CNN(num_classes=num_classes,data_type=data_type,
                name_prefix=f"{bootstrap_indexes_filename}_{num_pca_components}pca",
                model_dir=os.path.join(os.getenv("PME4_DATA_PATH"), "model"), sample=1)

    model.build_model(input_shape=(train_data.shape[1], train_data.shape[2]),
                      denses=[], filters=[16, 32, 64], str=[2, 2, 2],
                      ks=4, ps=2, dropout=0.2, lr=1e-5, bn=False, regularize=False)
    model.fit_model_save_epoch(train_data, train_labels, test_data, test_labels,
                               saved_epochs=settings.saved_epochs,
                               bs=settings.batch_size, init_epoch=0)


def train_bootstrap_eeg_cnn(data_type: str, settings: Settings, num_average: int, num_train_bootstrap_samples: int,
                                          num_test_bootstrap_samples: int, random_state: int,
                                          save_bootstrap_data: bool = True):
    train_bootstrap_data, test_bootstrap_data, train_labels, test_labels = get_bootstrap_data(data_type=data_type,
                                                                                              num_average=num_average,
                                                                                              num_train_bootstrap_samples=num_train_bootstrap_samples,
                                                                                              random_state=random_state,
                                                                                              one_hot_encode=True,
                                                                                              save_bootstrap_data=save_bootstrap_data,
                                                                                              num_test_bootstrap_samples=num_test_bootstrap_samples)

    bootstrap_indexes_filename = f"bootstrap_random{random_state}_{num_train_bootstrap_samples}sample_per_emotion_{num_average}average"

    (_, num_classes) = train_labels.shape

    model = CNN(num_classes=num_classes, name_prefix=f"{bootstrap_indexes_filename}", data_type=data_type,
                model_dir=os.path.join(os.getenv("PME4_DATA_PATH"), "model"), sample=1)
    model.build_model(input_shape=(train_bootstrap_data.shape[1], train_bootstrap_data.shape[2]),
                      denses=[64], filters=[16, 32, 64, 128],
                      str=[5, 5, 5, 5],
                      ks=5, ps=2, dropout=0.2, lr=1e-5, bn=False, regularize=False)
    model.fit_model_save_epoch(train_bootstrap_data, train_labels, test_bootstrap_data, test_labels,
                               saved_epochs=settings.saved_epochs,
                               bs=settings.batch_size, init_epoch=0)