import os
import keras.layers as kl
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l1_l2
from tensorflow.keras import activations
from utils.file_accessor import check_dirpath, save_model_summary
from model.tensor_board import TensorBoardWriter


class LSTM:
    def __init__(self, timesteps, num_features, num_classes, name_prefix, data_type, model_dir, model_num=1):
        self.timesteps = timesteps
        self.num_features = num_features
        self.num_classes = num_classes
        self.model_name = f"%s_Model%02d_LSTM" % (name_prefix, model_num)
        self.model_dir = os.path.join(model_dir, data_type)

    @staticmethod
    def construct_model_name(lr: float, lstm_units: list, l1_value: float, l2_value: float, dropout: float) -> str:
        model_name = "_".join(list(map(str, lstm_units)))
        if (0 < dropout < 1):
            model_name += "_dr%.0e" % dropout
        if l1_value > 0:
            model_name += "_l1_%.2f" % l1_value
        if l2_value > 0:
            model_name += "_l2_%.2f" % l2_value
        model_name += "_Adam_lr%.0e" % lr
        return model_name

    def build_model(self, lr: float, lstm_units: list, l1_value: float = 0, l2_value: float = 0.1, dropout: float = 0,
                    isCuDNN: bool = False):
        if len(lstm_units) < 0:
            return None
        self.model_name += self.construct_model_name(lr, lstm_units, l1_value, l2_value, dropout)
        self.model = Sequential()
        if isCuDNN:
            self.model.add(kl.CuDNNLSTM(units=lstm_units[0], input_shape=(self.timesteps, self.num_features),
                                        return_sequences=True, unit_forget_bias=True,
                                        kernel_regularizer=l1_l2(l1_value, l2_value), name="LSTM_L1"))
        else:
            self.model.add(kl.LSTM(units=lstm_units[0], input_shape=(self.timesteps, self.num_features),
                                   return_sequences=True, unit_forget_bias=True,
                                   kernel_regularizer=l1_l2(l1_value, l2_value), name="LSTM_L1"))
        for i, node in enumerate(lstm_units[1:], start=1):
            if isCuDNN:
                self.model.add(kl.CuDNNLSTM(units=lstm_units[i], return_sequences=True, unit_forget_bias=True,
                                            kernel_regularizer=l1_l2(l1_value, l2_value), name="LSTM_L%d" % i))
            else:
                self.model.add(kl.LSTM(units=lstm_units[i], return_sequences=True, unit_forget_bias=True,
                                       kernel_regularizer=l1_l2(l1_value, l2_value), name="LSTM_L%d" % i))

        if (0 < dropout < 1):
            self.model.add(kl.Dropout(dropout))
        self.model.add(kl.Dense(self.num_classes))
        self.model.add(kl.AveragePooling1D(self.timesteps, name="average_pooling"))
        self.model.add(kl.Flatten())
        self.model.add(kl.Activation(activations.softmax, name="softmax"))

        adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        self.model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

        self.model_path = os.path.join(self.model_dir, self.model_name)
        self.save_model_info()

    def update_model_name(self, model_name):
        self.model_name = model_name
        self.model_path = os.path.join(self.model_dir, self.model_name)

    def save_model_info(self):
        check_dirpath(self.model_path)
        # save_model_summary(self.model_dir, self.model_name, self.model)

    def save_model_structure(self):
        from keras.utils.vis_utils import plot_model
        model_strucuture_path = os.path.join(self.model_dir, self.model_name, 'model_plot.png')
        plot_model(self.model, to_file=model_strucuture_path, show_shapes=True, show_layer_names=True)

    def get_trained_model_name(self, trained_epoch):
        return "%s/%s_ep%d.h5" % (self.model_path, self.model_name, trained_epoch)

    def load_model(self, trained_epoch, model_name):
        from keras.models import load_model
        self.update_model_name(model_name)
        self.model = load_model(self.get_trained_model_name(trained_epoch=trained_epoch))

    def fit_model(self, train_data, train_label, test_data, test_label, n_epoch=5, epoch=1000, bs=128, init_epoch=0):
        for i in range(n_epoch):
            self.model.fit(train_data, train_label,
                           validation_data=(test_data, test_label),
                           initial_epoch=init_epoch + i * epoch,
                           epochs=init_epoch + (i + 1) * epoch,
                           batch_size=bs, shuffle=True, verbose=2,
                           callbacks=[TensorBoardWriter(log_dir=self.model_path, write_graph=False)])
            self.model.save(self.get_trained_model_name(trained_epoch=init_epoch + (i + 1) * epoch))

    def fit_model_save_epoch(self, train_data, train_label, test_data, test_label, saved_epochs=[], bs=128,
                             init_epoch=0):
        for saved_epoch in saved_epochs:
            self.model.fit(train_data, train_label,
                           validation_data=(test_data, test_label),
                           initial_epoch=init_epoch,
                           epochs=saved_epoch,
                           batch_size=bs, shuffle=True, verbose=2,
                           callbacks=[TensorBoardWriter(log_dir=self.model_path, write_graph=False)])
            self.model.save(self.get_trained_model_name(trained_epoch=saved_epoch))
            init_epoch = saved_epoch
