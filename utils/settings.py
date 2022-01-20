class Settings:
    def __init__(self):
        # other parameters
        self.num_emotions = 7
        self.num_subjects = 11
        self.subjects = list(range(1,12))
        self.IMG_SIZE = 224
        self.number_cvs = 5
        self.isCuDNN = False

        self.trial_time = 5 #second
        self.eeg_fs = 1000  # already downsampled
        self.eeg_pca_window = 100  #ms
        self.eeg_pca_overlap_window = 50  # ms


        self.psd_get_fq = 40
        self.periodgram_eeg_low_fq = 0
        self.periodgram_eeg_high_fq = 30
        self.periodgram_emg_low_fq = 20
        self.periodgram_emg_high_fq = 450
        self.eeg_window_size = 0.8
        self.eeg_window_overlap_size = 0.5
        self.num_eeg_channel = 8
        self.num_emg_channel = 6
        # model parameter
        self.learning_rate = 1e-4
        self.batch_size = 16

        self.allLabelNPY = {
            0: "anger",
            1: "disgust",
            2: "fear",
            3: "happy",
            4: "neutral",
            5: "sad",
            6: "surprise"
        }

    def set_audio(self):
        self.dropout = 0
        self.l2 = 0.1
        self.l1 = 0
        self.lstm_units = [128]
        self.saved_epochs=[30,60]

    def set_image(self):
        self.dropout = 0
        self.l2 = 0.1
        self.l1 = 0
        self.lstm_units = [512]
        self.saved_epochs=[200, 250]

    def set_image_ae_lstm(self):
        self.dropout = 0
        self.l2 = 0.1
        self.l1 = 0
        self.lstm_units = [256]
        self.saved_epochs=[100, 250, 500]

    def set_eeg(self):
        self.saved_epochs=[100, 200]
        self.dropout = 0
        self.l2 = 0.1
        self.l1 = 0
        self.lstm_units = [16]
        self.num_blocks_list = [2,2,2,2]
        self.number_dense = 128
        self.pool_size = 3

    def set_eeg_pca_lstm(self):
        self.dropout = 0.1
        self.l2 = 0.1
        self.l1 = 0
        self.batch_size = 16
        self.lstm_units = [32]
        self.saved_epochs=[10, 50]

    def set_eeg_ae(self):
        self.number_hidden = 100
        self.saved_epochs=[500, 1000,2000]
        self.batch_size = 128
        self.num_cluster = 10

    def set_emg_mfcc_lstm(self):
        self.dropout = 0.1
        self.l2 = 0.1
        self.l1 = 0
        self.batch_size = 64
        self.lstm_units = [16]
        self.saved_epochs=[50, 100, 200]


# 'x0_anger' 'x0_disgust' 'x0_fear' 'x0_happy' 'x0_neutral' 'x0_sad' 'x0_surprise'