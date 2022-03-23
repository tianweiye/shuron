from tensorflow import keras
from tensorflow.keras import Model

def cnn_encoder():
    encoder = keras.models.Sequential([
        keras.layers.Reshape([80,80,80,10], input_shape=[80,80,80,10]),
        keras.layers.Conv3D(8 ,3,strides=2,padding='same',activation='selu',kernel_initializer=keras.initializers.RandomNormal(0,0.02)),
        keras.layers.BatchNormalization(trainable=True),
        keras.layers.Conv3D(16,3,strides=2,padding='same',activation='selu',kernel_initializer=keras.initializers.RandomNormal(0,0.02)),
        keras.layers.BatchNormalization(trainable=True),
        keras.layers.Conv3D(32,3,strides=2,padding='same',activation='selu',kernel_initializer=keras.initializers.RandomNormal(0,0.02)),
        keras.layers.BatchNormalization(trainable=True),
        keras.layers.Conv3D(64,3,strides=2,padding='same', activation='selu',kernel_initializer=keras.initializers.RandomNormal(0,0.02)),
        keras.layers.BatchNormalization(trainable=True),
        keras.layers.Conv3D(2,1,strides=1,padding='same', activation='selu',kernel_initializer=keras.initializers.RandomNormal(0,0.02)),
        keras.layers.Flatten()
    ],name='cnn_encoder')
    return encoder

#cnn decoder
def cnn_decoder():
    decoder = keras.models.Sequential([
        keras.layers.Reshape(target_shape=[5, 5, 5, 2],input_shape=[250]),
        keras.layers.Conv3DTranspose(64,3,strides=1,padding='same',activation='selu',kernel_initializer=keras.initializers.RandomNormal(0,0.02)),
        keras.layers.BatchNormalization(trainable=True),
        keras.layers.Conv3DTranspose(64,3,strides=2,padding='same',activation='selu',kernel_initializer=keras.initializers.RandomNormal(0,0.02)),
        keras.layers.BatchNormalization(trainable=True),
        keras.layers.Conv3DTranspose(32,3,strides=2,padding='same',activation='selu',kernel_initializer=keras.initializers.RandomNormal(0,0.02)),
        keras.layers.BatchNormalization(trainable=True),
        keras.layers.Conv3DTranspose(32,3,strides=2,padding='same',activation='selu',kernel_initializer=keras.initializers.RandomNormal(0,0.02)),
        keras.layers.BatchNormalization(trainable=True),
        keras.layers.Conv3DTranspose(28,3,strides=2,padding='same',activation='selu',kernel_initializer=keras.initializers.RandomNormal(0,0.02)),
        keras.layers.BatchNormalization(trainable=True),
        keras.layers.Conv3DTranspose(20,3,strides=1,padding='same',activation='selu',kernel_initializer=keras.initializers.RandomNormal(0,0.02)),
        keras.layers.BatchNormalization(trainable=True),
        keras.layers.Conv3DTranspose(10,3,strides=1,padding='same',activation='sigmoid',kernel_initializer=keras.initializers.RandomNormal(0,0.02)),
        keras.layers.Reshape([80,80,80,10])
    ],name='cnn_decoder')
    return decoder

inputs3d =  keras.layers.Input(shape=[80,80,80,10], name='cnn3d_inputs')


def cnn_model():
    cnn = keras.models.Sequential([cnn_encoder(), cnn_decoder()],name='cnn')
    return cnn

if __name__ == '__main__' :
        print('\n--------------in model cnn.py-------------\n')
        
        encoder  =  cnn_encoder()
        decoder  =  cnn_decoder()
        feature   = encoder(inputs3d)
        cnnoutput = decoder(feature)
        CNNnet = Model([inputs3d], [cnnoutput], name='CNN')
        CNNnet.summary()
        print('\n-------------end of model cnn.py-------------\n')
