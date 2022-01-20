from dotenv import load_dotenv
from utils.settings import Settings
from audio.train_lstm_model import train_audio_lstm, train_audio_lstm_cv
from audio.train_traditional_model import train_audio_traditional_model, train_audio_traditional_model_cv

if __name__ == '__main__':
    # set the environment variables
    load_dotenv()

    # set the parameter for audio data
    settings = Settings()
    settings.set_audio()

    # type of process file
    data_type = "audio_mfcc_100ms_50ms"  # "audio_mfcc_20ms_10ms"

    # train one sample
    test_size = 0.3
    train_audio_traditional_model(data_type=data_type, test_size=test_size)
    train_audio_lstm(data_type=data_type, settings=settings, test_size=test_size, number_model_to_trained=1)

    # using cross validation
    cv_n = 5
    train_audio_traditional_model_cv(data_type=data_type, cv_n=cv_n, presaved_index=False, save_index=True)
    train_audio_lstm_cv(data_type=data_type, settings=settings, cv_n=cv_n, number_model_to_trained=20,
                        presaved_index=True,
                        save_index=False)
