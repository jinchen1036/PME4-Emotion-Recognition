import numpy as np
from scipy.signal import butter, filtfilt

def filter_emg(emg_data, fs1 = 5000):
    fl1 = 20
    fh1 = 500
    [b, a] = butter(6, [fl1 * 2 / fs1, fh1 * 2 / fs1], 'bandpass')

    filter_emg_data = np.zeros(emg_data.shape)
    if len(emg_data.shape) > 1:
        for channel in range(emg_data.shape[0]):
            filter_emg_data[channel, :] = filtfilt(b, a, emg_data[channel, :],
                                                   axis=0, padtype='odd', padlen=3 * (max(len(b), len(a)) - 1))
        return filter_emg_data
    else:
        return filtfilt(b, a, emg_data, axis=0, padtype='odd', padlen=3 * (max(len(b), len(a)) - 1))
