import os
from keras import Input, Model
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import load_model
from model.tensor_board import TensorBoardWriter
from utils.file_accessor import check_dirpath


class SingleLayerAutoEncoder:
    def __init__(self, input_dim, num_hidden, model_dir, filetype,name_prefix):
        self.input_dim = input_dim
        self.num_hidden = num_hidden
        self.model_dir = os.path.join(model_dir,filetype)
        self.model_name = "%s_SingleLayerAutoEncoder_%dhidden" % (name_prefix, num_hidden)
        self.model_path = os.path.join(self.model_dir, self.model_name)
        check_dirpath(self.model_path)
        self.build_model()

    def build_model(self):
        input_feature = Input(shape=(self.input_dim,))
        encoded = Dense(self.num_hidden, activation='relu')(input_feature)
        decoded = Dense(self.input_dim, activation='sigmoid')(encoded)
        self.autoencoder = Model(input_feature, decoded)

        adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999,epsilon=None, decay=0.0,amsgrad=False)
        self.autoencoder.compile(optimizer=adam,loss='mse')

    def get_trained_model_name(self, trained_epoch):
        return "%s/%s_ep%d.h5" % (self.model_path,self.model_name, trained_epoch)

    def fit_autoencoder(self, train_data, test_data, n_epoch=5, epoch=100, bs=128,
                  init_epoch=0):
        for i in range(n_epoch):
            self.autoencoder.fit(train_data, train_data,
                           validation_data=(test_data, test_data),
                           initial_epoch=init_epoch + i * epoch,
                           epochs=init_epoch + (i + 1) * epoch,
                           batch_size=bs, shuffle=True, verbose=2,
                           callbacks=[TensorBoardWriter(log_dir=self.model_path, write_graph=False)])
            self.autoencoder.save(self.get_trained_model_name(trained_epoch=init_epoch + (i + 1) * epoch))

    def fit_model_save_epoch(self,train_data, test_data, saved_epochs = [], bs=128, init_epoch = 0):
        for saved_epoch in saved_epochs:
            self.autoencoder.fit(train_data, train_data,
                           validation_data=(test_data, test_data),
                           initial_epoch=init_epoch,
                           epochs=saved_epoch,
                           batch_size=bs, shuffle=True, verbose=2,
                           callbacks=[TensorBoardWriter(log_dir=self.model_path, write_graph=False)])
            self.autoencoder.save(self.get_trained_model_name(trained_epoch=saved_epoch))
            init_epoch = saved_epoch
    def load_autoencoder(self,trained_epoch):
        self.autoencoder = load_model(self.get_trained_model_name(trained_epoch=trained_epoch))