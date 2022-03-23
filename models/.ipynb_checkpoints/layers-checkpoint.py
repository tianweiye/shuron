import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Lambda, Input, Dense, BatchNormalization
from tensorflow.keras.models import Model,Sequential



inputs3d =  keras.layers.Input(shape=[80,80,80,1], name='cnn3d_inputs')

##### LVAE funcs #####
class sampler(keras.layers.Layer):
    def call(self, inputs):
        mu, sigma = inputs
        return K.random_normal(tf.shape(mu), seed=5) * sigma + mu

class vae_sampler(keras.layers.Layer):
    def __init__(self,size, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.Dense_1 = Dense(size, kernel_initializer='zeros')  # mu
        self.Dense_2  = Dense(size, activation=activation, kernel_initializer='zeros')  # 2log_var
        self.sampler = sampler()

    def call(self, inputs):
        h = inputs
        mu = self.Dense_1(h)
        logsigma = self.Dense_2(h)
        logsigma = tf.clip_by_value(logsigma, 1e-7, 50)
        sigma = K.exp(logsigma)
        return self.sampler([mu, sigma]), mu, logsigma
    # 保存设置
    def get_config(self):
        base_config = super().get_config()
        return {**base_config}

class precision_weighted_sampler(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sampler = sampler()
    def call(self, inputs):
        mu1, sigma1, mu2, sigma2 = inputs
        sigma1__2 = 1 / K.square(sigma1)
        sigma2__2 = 1 / K.square(sigma2)
        mu = ( mu1*sigma1__2 + mu2*sigma2__2 )/(sigma1__2 + sigma2__2)
        sigma = 1 / (sigma1__2 + sigma2__2)
        logsigma = K.log(sigma + 1e-7)
        z = self.sampler([mu, sigma])
        return z, mu, logsigma
    def get_config(self):
        base_config = super().get_config()
        return {**base_config}

class get_loss_kl(keras.layers.Layer):
    def call(self, inputs):
        d_mu, d_logsigma, p_mu, p_logsigma = inputs
        d_sigma = K.exp(d_logsigma)
        p_sigma = K.exp(p_logsigma)
        d_sigma2, p_sigma2 = K.square(d_sigma), K.square(p_sigma)
        _lambda = 10.
        Lzs = 0.5 * K.sum((K.square(d_mu) + d_sigma2) - 2 * d_logsigma - 0.5, 1)
        Lzs = tf.math.maximum(_lambda, Lzs)

        Lzs2 = 0.5 * K.sum( (K.square(d_mu - p_mu) + d_sigma2) / p_sigma2 - 2 * K.log((d_sigma / p_sigma) + 1e-7) - 0.5, 1)
        return K.mean(Lzs), K.mean(Lzs2)