from tensorflow.keras.layers import Lambda, Input, Dense, BatchNormalization
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras import backend as K
from tensorflow import keras
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random, time, h5py, _thread, os
from random import shuffle
from models.cnn3d import *
from models.lvae import *
import tensorflow as tf
import numpy as np
from data.data_loader import *

######## DATA ########
x_train, x_test, x_test_label, x_train_label, y_test_label, y_train_label = load_data(0,25)
y_train, y_test = y_train_label, y_test_label

######## Model ########
random_seed=202107
np.random.seed(random_seed)
tf.random.set_seed(random_seed)
random.seed(random_seed)

# LVAE setting
frame = 15
original_dim = 250
input_shape = (original_dim, )
latent_dim = 2
#alpha = [0.005, 0.002,0.0005]
alpha = [0.02, 0.0001, 0.0001]
alpha = [0.02, 0.02, 0.02]
MLP_SIZES = [250,128,64]
Z_SIZES   = [64,48,18]
L = len(Z_SIZES)
MLPs = 2

# LVAE model
inputs = keras.layers.Input(shape=input_shape)
inputs3d =  keras.layers.Input(shape=[80,80,80,10], name='cnn3d_inputs')
encoder  =  cnn_encoder()
decoder  =  cnn_decoder()
lvaeinput       = encoder(inputs3d)
z_output        = lvae_encoder(L,MLPs)(lvaeinput)
output, *others = lvae_decoder(L,MLPs)(z_output)
cnnoutput       = decoder(output)
z = others[:L]
KL = others[-L:]

#MLP
MLP = Sequential([Dense(32, activation='relu', input_shape=(Z_SIZES[-1]+22,),kernel_initializer=tf.keras.initializers.RandomNormal(0,0.02)),
                Dense(32, activation='relu',kernel_initializer=tf.keras.initializers.RandomNormal(0,0.02)),
                Dense(1, activation='sigmoid',kernel_initializer=tf.keras.initializers.RandomNormal(0,0.02))],name='MLP')


input_y = Input(shape=(22,),name='daily_info')
output_y = Input(shape=(1,))
tmp = tf.concat([others[L-1],input_y],-1)
y_pre  = MLP(tmp)
lvae = Model([inputs3d,input_y,output_y], [cnnoutput,y_pre, *z], name='variational_autoencoder')
######## Model ########
lvae.summary()

######## 训练LVAE  ########
checkpoint_path = "lvae1-{epoch:02d}-{loss:.2f}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# callback save
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
def schedule(epoch ,offset = 0):
    N = (epoch+80)//10 - 2
    if N >=0: return float(0.002 *(0.75**N))
    else: return 0.002
lr_cb = tf.keras.callbacks.LearningRateScheduler(schedule, verbose=2)

import tensorflow as tf

# model
lr=0.0002
optimizer = keras.optimizers.Adam(learning_rate=lr)#best 0.001
lvae.compile(optimizer=optimizer)
history = lvae.fit([x_train,x_train_label,y_train], 
         epochs=80, callbacks=[cp_callback], batch_size=15)