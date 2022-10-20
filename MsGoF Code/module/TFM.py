
import tensorflow as tf
from keras.layers import Dense, Conv2D, Conv3D, Reshape, Multiply,Lambda, Add, Activation
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

def tf_cov_3d(x):
    size_x = x.shape
    mean_x = KB.mean(x, axis=2, keepdims=True)
    x1 = Reshape((int(size_x[1]), 1, int(size_x[2])))(x - mean_x)
    cov_2d = Multiply()([x - mean_x, x1])
    # print((x-mean_x).shape,cov_2d.shape)
    cov_3d = KB.batch_dot(x - mean_x, cov_2d, axes=[2, 3]) / KB.cast(KB.shape(x)[2] - 1, KB.floatx())
    # print(cov_3d.shape)
    return cov_3d

def GlobalPooling3order(Convf):
    size = Convf.shape
    a = Conv2D(int(size[3]), (1, 1))(Convf)
    a = KL.BatchNormalization()(a)
    a = Activation('relu')(a)
    a = Reshape((int(size[1]) * int(size[2]), int(size[3])))(a)
    a = Lambda(lambda x: tf.transpose(x, [0, 2, 1]))(a)
    a = Lambda(tf_cov_3d)(a)
    a = KL.BatchNormalization()(a)
    a = Reshape((int(size[3]), int(size[3]), int(size[3]), 1))(a)
    a = Conv3D(int(size[3]), (int(size[3]), int(size[3]), int(size[3])))(a)  ##残数量很大
    a = KL.BatchNormalization()(a)
    a = KL.LeakyReLU(alpha=0.1)(a)
    # a = Conv2D(size[3], (1,1), activation='sigmoid')(a)
    a = Dense(int(size[3]), activation='relu')(a)
    return a

def attention3(Convs):
    GPooling = GlobalPooling3order(Convs)
    Dense1 = Dense(32, activation='relu')(GPooling)
    Size = GPooling.shape
    c = int(Size[1])
    Dense2 = Dense(c, activation='relu')(Dense1)
    weight1 = Reshape((1, 1, Size[1]))(Dense2)
    # weight1 = KB.reshape(Dense2, (1,1,Size[1]))
    # weight1 = Lambda(lambda x:KB.reshape(x, (1,1,Size[1])))(Dense2)
    attention = Multiply()([Convs, weight1])
    return attention

def convs2(input,n):
    x = Conv2D(n, (3, 3), padding='same')(input)
    x = KL.BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def TFM(inputs, n):
    X3 = convs2(inputs, n)
    # h
    X3 = attention3(X3)
    # w
    X3b = Trans1(X3)
    X3bw = attention3(X3b)
    X3bw = Trans2(X3bw)
    # c
    X3a = Trans3(X3)
    X3aw = attention3(X3a)
    X3aw = Trans4(X3aw)

    X3w = Add()([X3, X3bw, X3aw])
    return  X3w