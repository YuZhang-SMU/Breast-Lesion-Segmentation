
import tensorflow as tf
from keras.layers import Dense, Conv2D, Reshape, Multiply,Lambda, Add, Activation
import keras.layers as KL
import keras.backend as KB

def Trans1(Convf):
    Convf = Lambda(lambda x: tf.transpose(x, [0, 3, 2, 1]))(Convf)
    return Convf

def Trans2(Convf):
    Convf = Lambda(lambda x: tf.transpose(x, [0, 3, 2, 1]))(Convf)
    return Convf

def Trans3(Convf):
    Convf = Lambda(lambda x: tf.transpose(x, [0, 3, 1, 2]))(Convf)
    return Convf

def Trans4(Convf):
    Convf = Lambda(lambda x: tf.transpose(x, [0, 2, 3, 1]))(Convf)
    return Convf

def tf_cov(x):
    mean_x = KB.mean(x, axis=2, keepdims=True)
    cov_x = KB.batch_dot(x - mean_x, x - mean_x, axes=[2, 2]) / KB.cast(KB.shape(x)[2] - 1, KB.floatx())
    return cov_x

def GlobalPooling2order(Convf):
    size = Convf.shape
    a = Conv2D(int(size[3]), (1, 1))(Convf)
    a = KL.BatchNormalization()(a)
    a = Activation('relu')(a)
    a = Reshape((int(size[1]) * int(size[2]), int(size[3])))(a)
    a = Lambda(lambda x: tf.transpose(x, [0, 2, 1]))(a)
    a = Lambda(tf_cov)(a)
    # print(a.shape)
    a = KL.BatchNormalization()(a)
    a = Reshape((int(size[3]), int(size[3]), 1))(a)
    a = Conv2D(int(size[3]), (int(size[3]), int(size[3])))(a)
    a = KL.BatchNormalization()(a)
    a = KL.LeakyReLU(alpha=0.1)(a)
    # a = Conv2D(size[3], (1,1), activation='sigmoid')(a)
    a = Dense(int(size[3]), activation='relu')(a)
    # output = Multiply()([Convf,a])
    # print(output.shape)
    return a

def attention2(Convs):
    GPooling = GlobalPooling2order(Convs)
    Dense1 = Dense(32, activation='relu')(GPooling)
    Size = GPooling.shape
    c = int(Size[1])
    Dense2 = Dense(c, activation='relu')(Dense1)
    weight1 = Reshape((1, 1, Size[1]))(Dense2)
    # weight1 = KB.reshape(Dense2, (1,1,Size[1]))
    # weight1 = Lambda(lambda x:KB.reshape(x, (1,1,Size[1])))(Dense2)
    attention = Multiply()([Convs, weight1])
    # print('a',attention.shape)
    return GPooling

def convs2(input,n):
    x = Conv2D(n, (3, 3), padding='same')(input)
    x = KL.BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def SFM(inputs,n):
    X2 = convs2(inputs, n)
    # h
    X2 = attention2(X2)
    # w
    X2b = Trans1(X2)
    X2bw = attention2(X2b)
    X2bw = Trans2(X2bw)
    # c
    X2a = Trans3(X2)
    X2aw = attention2(X2a)
    X2aw = Trans4(X2aw)

    X2w = Add()([X2, X2bw, X2aw])
    return  X2w