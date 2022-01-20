import os
import keras
import tensorflow as tf
from keras.regularizers import l1, l2
from keras.layers import AveragePooling1D, Input, Conv1D,Dense,Dropout,BatchNormalization,Flatten
from keras.models import Model
from utils.file_accessor import check_dirpath
from model.tensor_board import TensorBoardWriter

class CNN:
    def __init__(self, num_classes, name_prefix, data_type, model_dir, sample = 1):
        self.num_classes = num_classes
        self.model_name = "%s_Model%02d_CNN" % (name_prefix, sample)
        self.model_dir = os.path.join(model_dir,data_type)

    def build_model(self,input_shape=(5000, 8), denses=[128, 32], filters=[16, 32, 64, 128, 256, 512],
                      str=[1, 1, 1, 1, 1, 2], ks=4, ps=2, dropout=0.3, lr=1e-5, bn=True, regularize=True):
        size = input_shape[0]
        filter_name = "_".join("{0}".format(n) for n in filters)
        dense_name = "_".join("{0}".format(n) for n in denses)  # ([str(i) for i in denses])
        model_name = "_filter%s_ks%d_ps%d_dr%.0e_dense%s_lr%.0e" % (filter_name, ks, ps, dropout, dense_name, lr)

        net = {}
        net['Input'] = Input(input_shape)
        i = 1
        connect_layer = 'Input'
        for filter in filters:
            layer_name = 'Conv1d_%d' % i
            if regularize:
                net[layer_name] = Conv1D(filter, ks, strides=str[i - 1], activation='relu', kernel_regularizer=l2(1e-3),
                                         bias_regularizer=l2(1e-3), name=layer_name)(net[connect_layer])
            else:
                net[layer_name] = Conv1D(filter, ks, strides=str[i - 1], activation='relu', name=layer_name)(
                    net[connect_layer])

            size = size//2 -1
            connect_layer = layer_name
            if i % 2 == 0:
                index = i // 2
                if size > 1 and ps > 1:
                    size = size // ps - 1
                    net['AvgPooling_%d' % index] = AveragePooling1D(pool_size=ps, name='AvgPooling_%d' % index)(
                        net[connect_layer])
                    connect_layer = 'AvgPooling_%d' % index
                net['Dropout_%d' % index] = Dropout(dropout, name='Dropout_%d' % index)(net[connect_layer])
                connect_layer = 'Dropout_%d' % index
            if i % 4 == 0 and 'batch_normalization' not in net and bn:
                net['batch_normalization'] = BatchNormalization(name='batch_normalization')(net[connect_layer])
                connect_layer = 'batch_normalization'
            i += 1

        net['Flatten'] = Flatten()(net[connect_layer])
        connect_layer = 'Flatten'
        i = 1
        for dense in denses:
            if regularize:
                net['Dense_%d' % i] = Dense(dense, activation=tf.nn.relu, activity_regularizer=l1(1e-3))(
                    net[connect_layer])
            else:
                net['Dense_%d' % i] = Dense(dense, activation=tf.nn.relu)(net[connect_layer])
            connect_layer = 'Dense_%d' % i
            i += 1

        net['prediction'] = Dense(self.num_classes, activation='softmax', name="prediction_layer")(net[connect_layer])
        self.model = Model(inputs=net['Input'], outputs=net['prediction'])
        adam = keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        self.model.compile(optimizer=adam,
                      loss='categorical_crossentropy',
                      metrics=['accuracy']
                      )
        self.model_name += model_name
        self.model_name += "_Adam_lr%.0e" % lr

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
        model_strucuture_path = os.path.join(self.model_dir, self.model_name,'model_plot.png' )
        plot_model(self.model, to_file=model_strucuture_path, show_shapes=True, show_layer_names=True)

    def get_trained_model_name(self, trained_epoch):
        return "%s/%s_ep%d.h5" % (self.model_path,self.model_name, trained_epoch)

    def load_model(self,trained_epoch, model_name):
        from keras.models import load_model
        self.update_model_name(model_name)
        self.model = load_model(self.get_trained_model_name(trained_epoch=trained_epoch))

    def fit_model(self,train_data, train_label, test_data, test_label, n_epoch=5, epoch=1000, bs=128, init_epoch = 0):
        for i in range(n_epoch):
            self.model.fit(train_data, train_label,
                           validation_data=(test_data, test_label),
                           initial_epoch=init_epoch + i * epoch,
                           epochs=init_epoch + (i + 1) * epoch,
                           batch_size=bs, shuffle=True, verbose=2,
                           callbacks=[TensorBoardWriter(log_dir=self.model_path, write_graph=False)])
            self.model.save(self.get_trained_model_name(trained_epoch=init_epoch + (i + 1) * epoch))


    def fit_model_save_epoch(self,train_data, train_label, test_data, test_label, saved_epochs = [], bs=128, init_epoch = 0):
        for saved_epoch in saved_epochs:
            self.model.fit(train_data, train_label,
                           validation_data=(test_data, test_label),
                           initial_epoch=init_epoch,
                           epochs=saved_epoch,
                           batch_size=bs, shuffle=True, verbose=2,
                           callbacks=[TensorBoardWriter(log_dir=self.model_path, write_graph=False)])
            self.model.save(self.get_trained_model_name(trained_epoch=saved_epoch))
            init_epoch = saved_epoch