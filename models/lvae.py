from .layers import sampler,vae_sampler,precision_weighted_sampler,get_loss_kl
from tensorflow.keras.layers import Lambda, Input, Dense, BatchNormalization
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras import backend as K
from tensorflow import keras
import tensorflow as tf


original_dim = 250
Z_SIZES   = [64,48,18]
MLP_SIZES = [250,128,64]
L = len(Z_SIZES)
MLPs = 2
inputs = Input(shape=250)

def lvae_encoder(L=L, MLPs=MLPs):
    ##### cache for mu and sigma #####
    e_mus, e_logsigmas = [0] * L, [0] * L  # q(z_i+1 | z_i), bottom-up inference as Eq.7-9
    ##### Encoder ######
    h = inputs
    for l in range(L):
        for _ in range(MLPs):
            h = Dense(MLP_SIZES[l])(h)
            h = BatchNormalization(trainable=True)(h)
            h = keras.layers.ELU()(h)
        #### prepare for bidirectional inference ####
        _, e_mus[l], e_logsigmas[l] = vae_sampler( Z_SIZES[l])(h)

    encoder = Model(inputs, [*e_mus, *e_logsigmas], name='encoder')
    return encoder

def lvae_decoder(L=L, MLPs=MLPs):
    ##### cache for mu and sigma #####
    z = [0] * L
    KL = [0] * L
    e_mus, e_logsigmas = [Input(shape=(Z_SIZES[l],)) for l in range(L)], [Input(shape=(Z_SIZES[l],)) for l in range(L)]
    p_mus, p_logsigmas = [0] * L, [0] * L  # p(z_i | z_i+1), top-down prior as Eq.1-3
    d_mus, d_logsigmas = [0] * L, [0] * L  # q(z_i | .), bidirectional inference as Eq.17-19
    ##### Decoder ######
    for l in range(L)[::-1]:
        if l == L-1:
            mu, logsigma = e_mus[l], e_logsigmas[l]
            d_mus[l], d_logsigmas[l] = mu, logsigma
            z[l] = sampler()([d_mus[l], K.exp(d_logsigmas[l])])
            ##### prior of z_L is set as standard Gaussian, N(0,I). #####
            p_mus[l], p_logsigmas[l] = K.zeros(tf.shape(mu)), K.ones(tf.shape(logsigma))
            KL[l], _ = get_loss_kl()([d_mus[l],d_logsigmas[l],p_mus[l], p_logsigmas[l]])
        else:
            h = z[l+1]
            for _ in range(MLPs):
                h = Dense(MLP_SIZES[l+1])(h)
                h = BatchNormalization(trainable=True)(h)
                h = keras.layers.ELU()(h)
            _  ,  p_mus[l], p_logsigmas[l] = vae_sampler(Z_SIZES[l])(h) # Eq.13-15
            z[l], d_mus[l], d_logsigmas[l] = precision_weighted_sampler()([e_mus[l], K.exp(e_logsigmas[l]),p_mus[l], K.exp(p_logsigmas[l])])# Eq.17-19
            _, KL[l] = get_loss_kl()([d_mus[l], d_logsigmas[l], p_mus[l], p_logsigmas[l]])

    #decoder = Model(inputs=[*e_mus, *e_logsigmas],outputs=[*z], name='decoder')
    o = Dense(original_dim, 'elu')(z[0])
    decoder = Model([*e_mus, *e_logsigmas], [o, *z, *KL], name='decoder')
    #get loss
    return decoder

if __name__ == '__main__':
    # encoder = lvae_encoder()
    # encoder.summary()
    # decoder = lvae_decoder()
    # decoder.summary()
    
    ##### LVAE setting #####
    Z_SIZES   = [64,48,18]
    L = len(Z_SIZES)
    MLPs = 2


    encoder=lvae_encoder(L,MLPs)
    decoder=lvae_decoder(L,MLPs)
    encoder.summary()
    decoder.summary()