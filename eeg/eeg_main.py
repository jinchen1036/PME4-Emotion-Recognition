import time

from dotenv import load_dotenv
from utils.settings import Settings
from eeg.train_traditional_model import train_eeg_pca_traditional_model, train_eeg_pca_traditional_model_cv, \
    train_bootstrap_eeg_pca_traditional_model, train_bootstrap_eeg_traditional_model
from eeg.train_lstm_model import train_eeg_pca_lstm, train_eeg_pca_lstm_cv, train_bootstrap_eeg_pca_lstm, \
    train_bootstrap_eeg_lstm
from eeg.train_cnn_model import train_eeg_pca_cnn,train_eeg_pca_cnn_cv,train_bootstrap_eeg_pca_cnn,train_bootstrap_eeg_cnn
if __name__ == '__main__':
    # set the environment variables
    load_dotenv()

    # set the parameter for audio data
    settings = Settings()
    settings.set_eeg()

    # type of process file
    data_type = "processed_eeg"
    num_average = 20
    num_train_bootstrap_samples = 400
    num_test_bootstrap_samples = 100
    random_state = int(time.time())
    save_bootstrap_data = True
    num_pca_components = 50

    test_size = 0.3  # for one model
    cv_n = 5  # for cross validation

    # processed eeg - pca - traditional model
    train_eeg_pca_traditional_model(data_type=data_type, test_size=test_size, num_pca_components=num_pca_components)

    # processed eeg - pca - traditional model - cv
    train_eeg_pca_traditional_model_cv(data_type=data_type, cv_n=cv_n, num_pca_components=num_pca_components,
                                       presaved_index=False, save_index=True)

    # processed eeg - pca - traditional model - bootstrap
    train_bootstrap_eeg_pca_traditional_model(data_type=data_type, num_average=num_average,
                                              num_train_bootstrap_samples=num_train_bootstrap_samples,
                                              num_test_bootstrap_samples=num_test_bootstrap_samples,
                                              random_state=random_state, save_bootstrap_data=True,
                                              num_pca_components=num_pca_components)

    # processed eeg - pca - lstm
    train_eeg_pca_lstm(data_type=data_type, settings=settings, test_size=test_size,
                       num_pca_components=num_pca_components)

    # processed eeg - pca - lstm - cv
    train_eeg_pca_lstm_cv(data_type=data_type, settings=settings, cv_n=cv_n, num_pca_components=num_pca_components,
                          presaved_index=True, save_index=False)

    # processed eeg -pca - lstm - bootstrap
    train_bootstrap_eeg_pca_lstm(data_type=data_type, settings=settings, num_average=num_average,
                                 num_train_bootstrap_samples=num_train_bootstrap_samples,
                                 num_test_bootstrap_samples=num_test_bootstrap_samples,
                                 random_state=random_state, save_bootstrap_data=True,
                                 num_pca_components=num_pca_components)

    # processed eeg - pca - cnn
    train_eeg_pca_cnn(data_type=data_type, settings=settings, test_size=test_size,
                       num_pca_components=num_pca_components)

    # processed eeg - pca - cnn - cv
    train_eeg_pca_cnn_cv(data_type=data_type, settings=settings, cv_n=cv_n, num_pca_components=num_pca_components,
                          presaved_index=True, save_index=False)

    # processed eeg -pca - cnn - bootstrap
    train_bootstrap_eeg_pca_cnn(data_type=data_type, settings=settings, num_average=num_average,
                                 num_train_bootstrap_samples=num_train_bootstrap_samples,
                                 num_test_bootstrap_samples=num_test_bootstrap_samples,
                                 random_state=random_state, save_bootstrap_data=True,
                                 num_pca_components=num_pca_components)

    # processed eeg - traditional model - bootstrap
    train_bootstrap_eeg_traditional_model(data_type=data_type, num_average=num_average,
                                          num_train_bootstrap_samples=num_train_bootstrap_samples,
                                          num_test_bootstrap_samples=num_test_bootstrap_samples,
                                          random_state=random_state, save_bootstrap_data=True)

    # processed eeg - lstm - bootstrap
    train_bootstrap_eeg_lstm(data_type=data_type, settings=settings, num_average=num_average,
                             num_train_bootstrap_samples=num_train_bootstrap_samples,
                             num_test_bootstrap_samples=num_test_bootstrap_samples,
                             random_state=random_state, save_bootstrap_data=True)
