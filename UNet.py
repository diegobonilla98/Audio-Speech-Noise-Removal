from tensorflow.keras.layers import Conv2D, MaxPool2D, Conv2DTranspose, Input, LeakyReLU, add, Dropout, BatchNormalization, Activation, Lambda
from InstanceNormalization import InstanceNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
import tensorflow.keras.backend as K
from DataLoader import DataLoader
import tensorflow as tf

import os
import matplotlib.pyplot as plt
import numpy as np


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)


class UNet:
    def __init__(self, continue_from_epoch=None, test=False):
        self.data_loader = DataLoader(root_path='/media/bonilla/HDD_2TB_basura/databases/LibriSpeech/train-clean-100')
        if not test:
            if continue_from_epoch is None:
                self.input_dim = 512
                self.input_tensor = Input(shape=(self.input_dim, self.input_dim, 1))
                self.gf = 64
                self.channels = 2
                self.auto_encoder = self.build_model()
            else:
                self.continue_from_epoch = continue_from_epoch
                self.load_weights(self.continue_from_epoch)
            self.auto_encoder.summary()
            plot_model(self.auto_encoder, to_file='segmentation_model.png')

    def build_model(self):
        def conv2d(layer_input, filters=16, strides=1, name=None, f_size=4):
            d = Conv2D(filters, kernel_size=f_size, strides=strides, padding='same', name=name)(layer_input)
            # d = BatchNormalization(name=name + "_bn")(d)
            d = InstanceNormalization(name=name + "_bn")(d)
            d = Dropout(0.2)(d)
            d = Activation('relu')(d)
            return d

        def residual(layer_input, filters=16, strides=1, name=None, f_size=3):
            d = conv2d(layer_input, filters=filters, strides=strides, name=name, f_size=f_size)
            d = Conv2D(filters, kernel_size=f_size, strides=strides, padding='same', name=name + "_2")(d)
            # d = BatchNormalization(name=name + "_bn2")(d)
            d = InstanceNormalization(name=name + "_bn2")(d)
            d = Dropout(0.2)(d)
            d = add([d, layer_input])
            return d

        def conv2d_transpose(layer_input, filters=16, strides=1, name=None, f_size=4):
            u = Conv2DTranspose(filters, strides=strides, name=name, kernel_size=f_size, padding='same')(layer_input)
            # u = BatchNormalization(name=name + "_bn")(u)
            u = InstanceNormalization(name=name + "_bn")(u)
            u = Dropout(0.2)(u)
            u = Activation('relu')(u)
            return u

        # Image input
        c0 = self.input_tensor
        c1 = conv2d(c0, filters=self.gf, strides=1, name="g_e1", f_size=7)
        c2 = conv2d(c1, filters=self.gf * 2, strides=2, name="g_e2", f_size=3)
        c3 = conv2d(c2, filters=self.gf * 4, strides=2, name="g_e3", f_size=3)

        r1 = residual(c3, filters=self.gf * 4, name='g_r1')
        r2 = residual(r1, self.gf * 4, name='g_r2')
        r3 = residual(r2, self.gf * 4, name='g_r3')
        r4 = residual(r3, self.gf * 4, name='g_r4')
        r5 = residual(r4, self.gf * 4, name='g_r5')
        r6 = residual(r5, self.gf * 4, name='g_r6')
        r7 = residual(r6, self.gf * 4, name='g_r7')
        r8 = residual(r7, self.gf * 4, name='g_r8')
        r9 = residual(r8, self.gf * 4, name='g_r9')

        d1 = conv2d_transpose(r9, filters=self.gf * 2, f_size=3, strides=2, name='g_d1_dc')
        d2 = conv2d_transpose(d1, filters=self.gf, f_size=3, strides=2, name='g_d2_dc')

        output_mask = Conv2D(self.channels, kernel_size=7, strides=1, padding='same', name='output_noise_mask')(d2)
        output_mask = Activation('sigmoid')(output_mask)

        return Model(inputs=[c0], outputs=[output_mask])

    def get_model(self):
        return self.auto_encoder

    def save_weights(self, epoch):
        self.auto_encoder.save(os.path.join(self.data_loader.ROOT_PATH.replace('databases', 'models'), f'epoch_{epoch}.h5'))

    def load_weights(self, epoch):
        self.auto_encoder = load_model(os.path.join(self.data_loader.ROOT_PATH.replace('databases', 'models'), f'epoch_{epoch}.h5'), custom_objects={'mask_consistent_loss': self.mask_consistent_loss})

    @staticmethod
    def mask_consistent_loss(y_true, y_pred):
        noise_pred = Lambda(lambda x: x[:, :, :, 0])(y_pred)
        voice_pred = Lambda(lambda x: x[:, :, :, 1])(y_pred)

        noise_true = Lambda(lambda x: x[:, :, :, 0])(y_true)
        voice_true = Lambda(lambda x: x[:, :, :, 1])(y_true)

        bce = K.binary_crossentropy(noise_true, noise_pred) + K.binary_crossentropy(voice_true, voice_pred)
        cons_mask = K.binary_crossentropy(1. - noise_pred, voice_pred)
        return bce + cons_mask * 0.5

    def compile_and_fit(self, epochs, batch_size):
        if self.continue_from_epoch is not None:
            epochs -= self.continue_from_epoch
        losses = []
        initial_lr = 0.0002
        optimizer = Adam(lr=initial_lr, beta_1=0.5)
        self.auto_encoder.compile(optimizer=optimizer, loss=self.mask_consistent_loss)
        for epoch in range(epochs):
            if self.continue_from_epoch is not None:
                epoch += self.continue_from_epoch
            x, y = self.data_loader.load_batch(batch_size=batch_size)
            loss = self.auto_encoder.train_on_batch(x, np.concatenate([y, 1. - y], axis=-1))
            print(f'Epoch: {epoch}/{epochs}\tloss: {loss}')
            optimizer.learning_rate.assign(initial_lr * (0.43 ** epoch))
            if epoch % 25 == 0:
                self.save_weights(epoch)
                self.data_loader.sample_results(1, 'results', self.auto_encoder, epoch)
            losses.append(loss)
        plt.clf()
        plt.plot(losses)
        plt.show()

    def save_model(self):
        self.auto_encoder.save('autoencoder_model.h5')


asr = UNet(continue_from_epoch=15950)
asr.compile_and_fit(50000, 1)
asr.save_model()
