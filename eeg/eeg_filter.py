import numpy as np
from scipy.signal import butter, filtfilt

def filter_eeg(eeg_data, fs1 = 5000):
    fl1 = 0.1
    fh1 = 30
    [b, a] = butter(1, [fl1 * 2 / fs1, fh1 * 2 / fs1], 'bandpass')

    filter_eeg_data = np.zeros(eeg_data.shape)
    if len(eeg_data.shape) > 1:
        for channel in range(eeg_data.shape[0]):
            filter_eeg_data[channel, :] = filtfilt(b, a, eeg_data[channel, :],
                                                   axis=0, padtype='odd', padlen=3 * (max(len(b), len(a)) - 1))
        return filter_eeg_data
    else:
        return filtfilt(b, a, eeg_data, axis=0, padtype='odd', padlen=3 * (max(len(b), len(a)) - 1))

