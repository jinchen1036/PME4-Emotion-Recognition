import numpy as np
def generate_lstm_result(data_type,cv_i,model_num, trained_epoch, accuracy):
    return {
        "data_type": data_type,
        "cv_i": cv_i,
        "model_num": model_num,
        "trained_epoch": trained_epoch,
        "accuracy": accuracy
    }


def format_emg_mfcc_lstm(train_data, test_data, train_labels, test_labels):

    (train_size, number_channels, num_timesteps, num_mfccs) = train_data.shape
    (test_size, _, _, _) = test_data.shape

    train_data = np.swapaxes(train_data, 1, 2)
    test_data = np.swapaxes(test_data, 1, 2)

    train_data = np.reshape(train_data, (train_size, num_timesteps, -1))
    test_data = np.reshape(test_data, (test_size, num_timesteps, -1))

    print("train_data size", train_data.shape)
    print("train_labels size", train_labels.shape)
    print("test_data size", test_data.shape)
    print("test_labels size", test_labels.shape)

    return train_data, test_data, train_labels, test_labels

def format_emg_mfcc_cnn(train_data, test_data, train_labels, test_labels):

    (train_size, number_channels, num_timesteps, num_mfccs) = train_data.shape
    (test_size, _, _, _) = test_data.shape

    train_data = np.reshape(train_data, (train_size, number_channels, -1))
    test_data = np.reshape(test_data, (test_size, number_channels, -1))

    train_data = np.transpose(train_data, (0,2,1))
    test_data = np.transpose(test_data, (0,2,1))

    print("train_data size", train_data.shape)
    print("train_labels size", train_labels.shape)
    print("test_data size", test_data.shape)
    print("test_labels size", test_labels.shape)

    return train_data, test_data, train_labels, test_labels