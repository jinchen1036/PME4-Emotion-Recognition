import time

from dotenv import load_dotenv
from utils.settings import Settings

from emg.train_traditional_model import train_emg_traditional_model,train_emg_traditional_model_cv
from emg.train_lstm_model import train_emg_lstm,train_emg_lstm_cv
from emg.train_cnn_model import train_emg_cnn,train_emg_cnn_cv

if __name__ == '__main__':
    # set the environment variables
    load_dotenv()

    # set the parameter for audio data
    settings = Settings()
    settings.set_emg_mfcc_lstm()

    # type of process file
    data_type = "processed_emg" # mfcc file
    test_size = 0.3  # for one model
    cv_n = 5  # for cross validation

    # emg - mfcc - traditional model
    train_emg_traditional_model(data_type=data_type, test_size=test_size)

    # emg - mfcc - traditional model - cv
    train_emg_traditional_model_cv(data_type=data_type, cv_n=cv_n, presaved_index=False, save_index=True)

    # emg - mfcc - lstm
    train_emg_lstm(data_type=data_type, settings=settings, test_size=test_size)

    # emg - mfcc - lstm - cv
    train_emg_lstm_cv(data_type=data_type, settings=settings, cv_n=cv_n,
                     presaved_index=True, save_index=False)

    # emg - mfcc - cnn
    train_emg_cnn(data_type=data_type, settings=settings, test_size=test_size)

    # emg - mfcc - cnn - cv
    train_emg_cnn_cv(data_type=data_type, settings=settings, cv_n=cv_n, presaved_index=True, save_index=False)