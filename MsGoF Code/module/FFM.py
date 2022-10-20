import tensorflow as tf
from keras.layers import Dense, Conv2D, GlobalMaxPooling2D, Reshape, MaxPooling2D, Multiply, Average, Lambda, Concatenate, UpSampling2D, Add


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

def RN(input1, n_kernel):
    Layer0 = MaxPooling2D((2, 2))(input1)
    Layer1 = Conv2D(n_kernel, (1, 1), activation='relu', padding='same')(Layer0)
    Layer1 = Conv2D(n_kernel, (3, 3), activation='relu', padding='same')(Layer1)
    Layer1 = Conv2D(n_kernel, (1, 1), activation='relu', padding='same')(Layer1)
    Layer1 = UpSampling2D((2, 2))(Layer1)
    output1 = Concatenate()([input1, Layer1])
    return output1

def attention1(Convs):
    GPooling = GlobalMaxPooling2D()(Convs)
    Dense1 = Dense(32, activation='relu')(GPooling)
    Size = GPooling.shape
    c = int(Size[1])
    Dense2 = Dense(c, activation='relu')(Dense1)
    weight1 = Reshape((1, 1, Size[1]))(Dense2)
    # weight1 = Lambda(lambda x:KB.reshape(x, (1,1,Size[1])))(Dense2)
    attention = Multiply()([Convs, weight1])
    return attention

def fusion1(attention_1, attention_2, attention_3):
    attention_1_re = Average()([attention_1, attention_2])
    attention_2_re = Average()([attention_2, attention_3])
    attention_3_re = Average()([attention_3, attention_1])
    # print(attention_1_re.shape)
    return attention_1_re, attention_2_re, attention_3_re

# FFM
def FFM(SG, n):
    X1 = RN(SG, n)
    # h
    X1 = attention1(X1)
    # w
    X1b = Trans1(X1)
    X1bw = attention1(X1b)
    X1bw = Trans2(X1bw)
    # c
    X1a = Trans3(X1)
    X1aw = attention1(X1a)
    X1aw = Trans4(X1aw)

    X1w = Add()([X1, X1bw, X1aw])
    return X1w



