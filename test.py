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
import pandas as pd
from data.data_loader import *
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

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


lvae.load_weights('lvae2-80-47.39.ckpt')
# テスト（一括でテストするとメモリオーバーですから、少しずつテストしていきます）
results_test_img =[]
results_test_y =[]
results_train_y =[]
results_train_img =[]
model = lvae
results_test_img, results_test_y, _,_,test_resultz3 = model([x_test,x_test_label,y_test])

results_train_img, results_train_y, _,_,train_resultz3 = model([x_train[:15],x_train_label[:15], y_train[:15]])
for i in [1,2,3,4,5,6,7,8,9]:
    results_train_img2, results_train_y2, _,_,tmp = model([x_train[i*15:i*15+15],x_train_label[i*15:i*15+15], y_train[i*15:i*15+15]])
    results_train_img = tf.concat([results_train_img,results_train_img2],0)
    results_train_y = tf.concat([results_train_y,results_train_y2],0)
    train_resultz3 = tf.concat([train_resultz3,tmp],0)
    print(i)

results_test_img = results_test_img[...].numpy()
results_test_y = results_test_y[...,0].numpy()
results_train_y =results_train_y[...,0].numpy()
results_train_img = results_train_img[...].numpy()

######## 評価 ########
# ROC AUC 
from sklearn import metrics
fpr_ts_3, tpr_ts_3, thresholds_ts_3 = metrics.roc_curve(y_test, results_test_y, pos_label=1)
fpr_tr_3, tpr_tr_3, thresholds_tr_3 = metrics.roc_curve(y_train, results_train_y, pos_label=1)

from sklearn.metrics import auc  
auc_ts_3 = metrics.auc(fpr_ts_3, tpr_ts_3)  
auc_tr_3 = metrics.auc(fpr_tr_3, tpr_tr_3)  

maxindex = (tpr_ts_3-fpr_ts_3).tolist().index(max(tpr_ts_3-fpr_ts_3))
threshold = thresholds_ts_3[maxindex]

import matplotlib.pyplot as plt
plt.plot(fpr_ts_3, tpr_ts_3, 'k-',label="test  (auc={:.2f})".format(auc_ts_3))
plt.plot(fpr_tr_3, tpr_tr_3, 'y-',label="train (auc={:.2f})".format(auc_tr_3))
plt.plot([fpr_ts_3[maxindex]],[tpr_ts_3[maxindex]], 'ro')
plt.text(fpr_ts_3[maxindex]+0.05,tpr_ts_3[maxindex]-0.2, 'FPR:{:.2f}\nTPR:{:.2f}\nThreshold:{:.2f}'.format(fpr_ts_3[maxindex],tpr_ts_3[maxindex],thresholds_ts_3[maxindex])
)
plt.ylabel("True Negative Rate")
plt.xlabel("False Positive Rate")
plt.legend(loc=4)
######## 評価 ########

######## テスト ########
# 混同行列
def plot_confusion_matrix(matrix, fig_name = 'lvae1459_h_map_ce+20.jpg'):
    df_cm = pd.DataFrame(matrix,
                     index = [i for i in ['Positive','Negative']],
                     columns = [i for i in ['Positive','Negative']])

    plt.figure(figsize = (5,3))
    sn.heatmap(df_cm, annot=True,fmt='.20g',cmap="BuPu")
    plt.ylabel("Actual Values")
    plt.xlabel("Predicted Values")
    plt.savefig(fig_name,dpi=400)
    
from sklearn.metrics import confusion_matrix

threshold = 0.5
y_train_2 = (results_train_y>threshold).astype('float')
tmp = confusion_matrix(y_train, y_train_2)
plot_confusion_matrix(tmp,'train2_CM_th05.jpg')

y_test_2 = (results_test_y>threshold).astype('float')
tmp = confusion_matrix(y_test, y_test_2)
plot_confusion_matrix(tmp,'test2_CM_th05.jpg')
######## テスト ########