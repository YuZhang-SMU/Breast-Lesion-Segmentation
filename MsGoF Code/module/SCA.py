import tensorflow as tf
from keras.layers import Reshape, Lambda, Concatenate, ZeroPadding2D, Cropping2D

# spatial context aggregation
def SCA1(Input_):
    size = Input_.shape
    H = size[1]
    W = size[2]
    C = size[3]
    x = Lambda(lambda x: tf.transpose(x, [0, 3, 1, 2]))(Input_)
    f1 = Reshape((C, H * W, 1))(x)
    w1 = Lambda(matmul_channel)(f1)
    xx1 = Lambda(lambda x: tf.matmul(x[0], x[1]))([w1, f1])
    xx1 = Lambda(lambda x: tf.transpose(x, [0, 2, 3, 1]))(xx1)
    xx1 = Reshape((H, W, C))(xx1)
    return xx1

def SCA2(Input_):
    size = Input_.shape
    H = size[1]
    W = size[2]
    C = size[3]
    x = Lambda(lambda x: tf.transpose(x, [0, 3, 1, 2]))(Input_)
    f1 = Reshape((C, H * W, 1))(x)
    x2 = ZeroPadding2D(((1, 0), (1, 0)), data_format='channels_first')(x)
    x21 = Cropping2D(((0, 1), (0, 1)), data_format='channels_first')(x2)
    x22 = Cropping2D(((0, 1), (1, 0)), data_format='channels_first')(x2)
    x23 = Cropping2D(((1, 0), (0, 1)), data_format='channels_first')(x2)
    x24 = Cropping2D(((1, 0), (1, 0)), data_format='channels_first')(x2)
    f21 = Reshape((C, H * W, 1))(x21)
    f22 = Reshape((C, H * W, 1))(x22)
    f23 = Reshape((C, H * W, 1))(x23)
    f24 = Reshape((C, H * W, 1))(x24)
    f2 = Concatenate(axis=-1)([f21, f22, f23, f24])  # receptive field 4 -----(C,HW,4)
    w2 = Lambda(matmul_channel)(f2)
    xx2 = Lambda(lambda x: tf.matmul(x[0], x[1]))([w2, f1])
    xx2 = Lambda(lambda x: tf.transpose(x, [0, 2, 3, 1]))(xx2)
    xx2 = Reshape((H, W, C))(xx2)

    return xx2

def SCA3(Input_):
    size = Input_.shape
    H = size[1]
    W = size[2]
    C = size[3]
    #S=1x1
    x = Lambda(lambda x: tf.transpose(x, [0, 3, 1, 2]))(Input_)
    f1 = Reshape((C, H * W, 1))(x)
    x3 = ZeroPadding2D(((1, 1), (1, 1)), data_format='channels_first')(x)
    x31 = Cropping2D(((0, 2), (0, 2)), data_format='channels_first')(x3)
    x32 = Cropping2D(((0, 2), (1, 1)), data_format='channels_first')(x3)
    x33 = Cropping2D(((0, 2), (2, 0)), data_format='channels_first')(x3)
    x34 = Cropping2D(((1, 1), (0, 2)), data_format='channels_first')(x3)
    x35 = Cropping2D(((1, 1), (1, 1)), data_format='channels_first')(x3)
    x36 = Cropping2D(((1, 1), (2, 0)), data_format='channels_first')(x3)
    x37 = Cropping2D(((2, 0), (0, 2)), data_format='channels_first')(x3)
    x38 = Cropping2D(((2, 0), (1, 1)), data_format='channels_first')(x3)
    x39 = Cropping2D(((2, 0), (2, 0)), data_format='channels_first')(x3)
    f31 = Reshape((C, H * W, 1))(x31)
    f32 = Reshape((C, H * W, 1))(x32)
    f33 = Reshape((C, H * W, 1))(x33)
    f34 = Reshape((C, H * W, 1))(x34)
    f35 = Reshape((C, H * W, 1))(x35)
    f36 = Reshape((C, H * W, 1))(x36)
    f37 = Reshape((C, H * W, 1))(x37)
    f38 = Reshape((C, H * W, 1))(x38)
    f39 = Reshape((C, H * W, 1))(x39)
    f3 = Concatenate(axis=-1)([f31, f32, f33, f34, f35, f36, f37, f38, f39])  # receptive field 9 -----(C,HW,9)
    w3 = Lambda(matmul_channel)(f3)
    xx3 = Lambda(lambda x: tf.matmul(x[0], x[1]))([w3, f1])
    xx3 = Lambda(lambda x: tf.transpose(x, [0, 2, 3, 1]))(xx3)
    xx3 = Reshape((H, W, C))(xx3)
    return xx3

def matmul_channel(a):
    a1 = tf.transpose(a, [0, 1, 3, 2])
    weight = tf.matmul(a, a1)
    return weight
