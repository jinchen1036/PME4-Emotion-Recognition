from keras.models import Model
from keras.layers import Conv2D,Conv2DTranspose
from keras.layers import Input,Flatten, Dense, Reshape
from keras.optimizers import Adam

class Image_AE:
    def __init__(self, IMG_SIZE, CHANNEL_SIZE):
        self.img_size = IMG_SIZE
        self.channel_size = CHANNEL_SIZE
    def init_model(self,filters=[32, 128, 512, 2048], lr=1e-4):
        img_deduct_size = int(self.img_size/64)
        if self.img_size % 64 == 0:
            pad3 = 'same'
        else:
            pad3 = 'valid'
        self.net = {}
        self.net['input'] = Input(shape=(self.img_size, self.img_size, self.channel_size),name='input')

        self.net['encode_conv1'] = Conv2D(filters[0], kernel_size=(4, 4), strides=(3, 3), padding='same', activation='relu',name ='encode_conv1')(self.net['input'])

        self.net['encode_conv2'] = Conv2D(filters[1], kernel_size=(4, 4), strides=(3, 3), padding='same', activation='relu',name ='encode_conv2')(self.net['encode_conv1'])

        self.net['encode_conv3'] = Conv2D(filters[2], kernel_size=(4, 4), strides=(3, 3), padding='same', activation='relu',name ='encode_conv3')(self.net['encode_conv2'])

        self.net['encode_conv4'] = Conv2D(filters[3], kernel_size=(4, 4), strides=(3, 3), padding=pad3, activation='relu',name='encode_conv6')(self.net['encode_conv3'])

        self.net['encode_flat'] = Flatten(name='encode_flat')(self.net['encode_conv4'])

        self.net['encode_dense1'] = Dense(filters[4],name='encode_dense1')(self.net['encode_flat'])

        self.net['decode_dense1'] = Dense(filters[3]*img_deduct_size*img_deduct_size,name='decode_dense1')(self.net['encode_dense1'])

        self.net['decode_reshape'] = Reshape((img_deduct_size, img_deduct_size, filters[3]),name='decode_reshape')(self.net['decode_dense1'])

        self.net['decode_conv1'] = Conv2DTranspose(filters[2], kernel_size=(4, 4), strides=(3, 3), padding=pad3, activation='relu',name='decode_conv1')(self.net['decode_reshape'])

        self.net['decode_conv2'] = Conv2DTranspose(filters[1], kernel_size=(4, 4), strides=(3, 3), padding='same', activation='relu',name='decode_conv2')(self.net['decode_conv1'])

        self.net['decode_conv3'] = Conv2DTranspose(filters[0], kernel_size=(4, 4), strides=(3, 3), padding='same', activation='relu',name='decode_conv3')(self.net['decode_conv2'])

        self.net['decode_img'] = Conv2DTranspose(self.channel_size, (3, 3), strides=(3, 3), padding='same', activation='sigmoid',name='decode_img')(self.net['decode_conv3'])

        self.autoencoder = Model(self.net['input'], self.net['decode_img'])
        self.autoencoder.summary()
        self.encoder = Model(self.net['input'], self.net['encode_dense1'])
        self.autoencoder.compile(optimizer=Adam(lr=lr), loss='mse')