#!/usr/bin/env python
# -*- coding:utf-8 -*-

from keras.layers import Input, Lambda,Conv2D, BatchNormalization, Activation, Add, concatenate, MaxPooling2D
from keras.layers import UpSampling2D
import tensorflow as tf
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import mean_squared_error
from loss import bce_dice_loss
from metric import dice_score
from keras import backend as K
K.set_image_dim_ordering('tf')

def conv_unit(inputs, kernel_size, filters):
    conv_f = Conv2D(filters=filters, kernel_size=kernel_size, padding='same', strides=1, kernel_initializer='he_normal')(inputs)
    conv_bn = BatchNormalization()(conv_f)
    conv_a = Activation(activation='relu')(conv_bn)

    return conv_a

def conv_block(conv_input, nb_filters):
    conv1 = conv_unit(conv_input, 3, nb_filters)
    conv2 = conv_unit(conv1, 3, nb_filters)
    conv3 = conv_unit(conv2, 3, nb_filters)

    return conv3

def encoder_block(inputs, ker_nums):
    conv = conv_block(inputs, ker_nums)
    maxp = MaxPooling2D(pool_size=(2, 2))(conv)
    return conv, maxp

def encoder(inputs, ker_nums=None):
    out1_e1, out2_e1 = encoder_block(inputs, ker_nums=ker_nums[0])
    out1_e2, out2_e2 = encoder_block(out2_e1, ker_nums=ker_nums[1])
    out1_e3, out2_e3 = encoder_block(out2_e2, ker_nums=ker_nums[2])
    out1_e4, out2_e4 = encoder_block(out2_e3, ker_nums=ker_nums[3])

    return (out1_e1, out1_e2, out1_e3, out1_e4), out2_e4

def decoder_block(de_inputs, from_encoder, ker_nums):
    up = UpSampling2D(size=(2, 2), interpolation='bilinear')(de_inputs)
    conca = concatenate([up, from_encoder], axis=3)
    conv = conv_block(conca, ker_nums)
    return conv

def decoder(inputs, from_encoder, ker_nums=None):
    out_d1 = decoder_block(inputs, from_encoder[3], ker_nums=ker_nums[3])
    out_d2 = decoder_block(out_d1, from_encoder[2], ker_nums=ker_nums[2])
    out_d3 = decoder_block(out_d2, from_encoder[1], ker_nums=ker_nums[1])
    out_d4 = decoder_block(out_d3, from_encoder[0], ker_nums=ker_nums[0])

    return out_d1, out_d2, out_d3, out_d4

#===================main network===================#
def Main_net(input_ori, input_fore):
    inputs = concatenate([input_ori, input_fore], axis=-1)
    (out_fe1, out_fe2, out_fe3, out_fe4), mid_in = encoder(inputs, ker_nums=[64, 128, 256, 512])
    mid_out = conv_block(mid_in, 1024)
    out_fd1, out_fd2, out_fd3, out_fd4 = decoder(mid_out, [out_fe1, out_fe2, out_fe3, out_fe4], ker_nums=[64, 128, 256, 512])

    out_main = Conv2D(filters=1, kernel_size=(1, 1), padding='same', activation='sigmoid', name='out_main',kernel_initializer='he_normal')(out_fd4)

    return (out_fe1, out_fe2, out_fe3, out_fe4), out_main, (out_fd1, out_fd2, out_fd3, out_fd4), mid_out

#================auxiliary network=================#
def Auxiliary_net(input_ori, input_back):
    inputs = concatenate([input_ori, input_back], axis=-1)
    (out_be1, out_be2, out_be3, out_be4), mid_in = encoder(inputs, ker_nums=[64, 128, 256, 512])
    mid_out = conv_block(mid_in, 1024)

    out_bd1, out_bd2, out_bd3, out_bd4 = decoder(mid_out, [out_be1, out_be2, out_be3, out_be4], ker_nums=[64, 128, 256, 512])

    return (out_be1, out_be2, out_be3, out_be4), (out_bd1, out_bd2, out_bd3, out_bd4), mid_out

#================middle stream block===============#
def MS_block(input_fore, input_back, input_pos, ker_num):
    out_fore, out_back = BAF_unit(input_fore, input_back, ker_num)
    out_shape1, out_shape2 = Shape_aware_unit(out_fore, ker_num)
    out_edge = Edge_aware_unit(out_shape1, ker_num)
    out_pos = Position_aware_unit(out_edge, ker_num, input_pos)

    return out_pos, out_back, out_shape2

def Middle_stream(fore_enconv, fore_mid, fore_deconv, back_enconv, back_mid, back_deconv, input_pos, ker_num):
    out_pos1, out_back1, out_shape1 = MS_block((fore_enconv[3], fore_mid, fore_deconv[0]), (back_enconv[3], back_mid, back_deconv[0]),
                                input_pos[3], ker_num[0])

    out_pos2, out_back2, out_shape2 = MS_block((fore_enconv[2], out_pos1, fore_deconv[1]), (back_enconv[2], out_back1, back_deconv[1]),
                               input_pos[2], ker_num[1])

    out_pos3, out_back3, out_shape3 = MS_block((fore_enconv[1], out_pos2, fore_deconv[2]), (back_enconv[1], out_back2, back_deconv[2]),
                               input_pos[1], ker_num[2])

    out_pos4, out_back4, out_shape4 = MS_block((fore_enconv[0], out_pos3, fore_deconv[3]), (back_enconv[0], out_back3, back_deconv[3]),
                               input_pos[0], ker_num[3])

    out_shape1 = UpSampling2D(size=(8, 8), data_format='channels_last', interpolation='bilinear', name='shape11')(
        out_shape1)
    out_shape2 = UpSampling2D(size=(4, 4), data_format='channels_last', interpolation='bilinear', name='shape22')(
        out_shape2)
    out_shape3 = UpSampling2D(size=(2, 2), data_format='channels_last', interpolation='bilinear', name='shape33')(
        out_shape3)
    shape_fuse = concatenate([out_shape1, out_shape2, out_shape3, out_shape4], axis=3, name='cc5')

    out_shape = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid', name='out_shape',
                       kernel_initializer='he_normal')(shape_fuse)

    out_middle = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid', name='out_middle',
                       kernel_initializer='he_normal')(out_pos4)

    return out_middle, out_shape

## self-reversal operation in background-assisted fusion unit
def Self_reversal(input):
    f_shape = input.get_shape()
    f_r = tf.reshape(input, (-1, f_shape[1]*f_shape[2], f_shape[-1]))
    m = tf.reduce_max(f_r, axis=1)
    m = tf.reshape(m, [-1, 1, 1, f_shape[-1]])
    out = tf.subtract(m, input)
    return out

## background-assisted fusion unit
def BAF_unit(input_fore, input_back, ker_num):
    #back_path
    up_back = UpSampling2D(size=(2, 2), interpolation='bilinear')(input_back[1])
    conv_dila_b = conv_unit2(up_back, 1, ker_num, dilation_rate=2)
    conv_be = conv_unit2(input_back[0], 1, ker_num // 2)
    conv_bd = conv_unit2(input_back[2], 1, ker_num // 2)
    conca_b1 = concatenate([conv_be, conv_bd], axis=-1)
    add_fuse_b = Add()([conv_dila_b, conca_b1])
    add_fuse_b = Activation(activation='relu')(add_fuse_b)

    conv_b1 = conv_unit(add_fuse_b, 1, ker_num)
    conv_b2 = conv_unit(conv_b1, 3, ker_num)
    conv_b3 = conv_unit(conv_b2, 1, ker_num//8)
    conca_b2 = concatenate([add_fuse_b, conv_b3], axis=-1)
    out_back = conv_unit(conca_b2, 1, 128)
    inv_baf = Lambda(Self_reversal)(out_back)

    #fore_path
    up_fore = UpSampling2D(size=(2, 2), interpolation='bilinear')(input_fore[1])
    conv_dila_f = conv_unit2(up_fore, 1, ker_num, dilation_rate=2)
    conv_fe = conv_unit2(input_fore[0], 1, ker_num//2)
    conv_fd = conv_unit2(input_fore[2], 1, ker_num//2)
    conca_f1 = concatenate([conv_fe, conv_fd], axis=-1)
    add_fuse_f = Add()([conv_dila_f, conca_f1])
    add_fuse_f = Activation(activation='relu')(add_fuse_f)

    conv_f1 = conv_unit(add_fuse_f, 1, ker_num)
    conv_f2 = conv_unit(conv_f1, 3, ker_num//2)
    conca_f2 = concatenate([conv_f2, inv_baf])
    conv_f3 = conv_unit(conca_f2, 3, ker_num // 4)
    out_fore = concatenate([add_fuse_f, conv_f3], axis=3)

    return out_fore, out_back

def conv_unit2(inputs, kernel_size, filters, dilation_rate=1):
    conv = Conv2D(filters, kernel_size, dilation_rate=dilation_rate, kernel_initializer='he_normal', padding='same')(inputs)
    out = BatchNormalization()(conv)

    return out

## shape-aware unit
def Shape_aware_unit(inputs, num):
    conv1 = conv_unit(inputs, 1, num)
    conv2 = conv_unit(conv1, 3, num//2)
    conv3 = conv_unit(conv2, 3, num//4)
    conca = concatenate([inputs, conv3], axis=-1)
    out_shape = conv_unit(conv2, 1, 16)

    return conca, out_shape

## edge-aware unit
def Edge_aware_unit(inputs, kernel_num):
    conv1 = conv_unit(inputs, 1, kernel_num)
    conv2 = conv_unit(conv1, 3, kernel_num//2)
    edge_f = Lambda(edge_enhance)(conv2)
    conv3 = conv_unit(edge_f, 3, kernel_num//4)
    x_final = concatenate([inputs, conv3], axis=3)

    return x_final

def get_sobel_kernel(kernel, length):
    kernel_1 = []
    kernel_2 = []
    for i in range(length):
        kernel_1.append(kernel)
    for i in range(length):
        kernel_2.append(kernel_1)
    kernel_out = tf.constant(kernel_2, dtype="float32")
    kernel_out = tf.reshape(kernel_out, [3, 3, length, length])

    return kernel_out

def sobel_operation(input, kernel):
    input_shape = input.get_shape()
    sobel_kernel = get_sobel_kernel(kernel, input_shape[3])
    sobel_out = tf.nn.conv2d(input, sobel_kernel, strides=[1, 1, 1, 1], padding='SAME')

    return sobel_out

def edge_enhance(input):
    sobel1 = sobel_operation(input, [[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    sobel2 = sobel_operation(input, [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel3 = sobel_operation(input, [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]])
    sobel4 = sobel_operation(input, [[2, 1, 0], [1, 0, -1], [0, -1, -2]])
    sobel_enhance = Add()([sobel1, sobel2, sobel3, sobel4])

    return sobel_enhance

## position-aware unit
def Position_aware_unit(input_f, kernel_num, input_pos):
    conv1 = conv_unit(input_f, 1, kernel_num)
    conv2 = conv_unit(conv1, 3, kernel_num//2)
    conca = concatenate([conv2, input_pos], axis=-1)
    conv3 = conv_unit(conca, 3, kernel_num // 4)
    x1 = concatenate([input_f, conv3], axis=3)

    return x1




def SMU_net():
    input_ori = Input(shape=(256, 256, 1))
    input_fore = Input(shape=(256, 256, 1))
    input_back = Input(shape=(256, 256, 1))

    input_pos1 = Input(shape=(256, 256, 1))
    input_pos2 = Input(shape=(128, 128, 1))
    input_pos3 = Input(shape=(64, 64, 1))
    input_pos4 = Input(shape=(32, 32, 1))

    fore_enconv, out_main, fore_deconv, fore_mid = Main_net(input_ori, input_fore)
    back_enconv, back_deconv, back_mid = Auxiliary_net(input_ori, input_back)
    out_middle, out_shape = Middle_stream(fore_enconv, fore_mid, fore_deconv, back_enconv, back_mid, back_deconv, (input_pos1, input_pos2, input_pos3, input_pos4),[256,128,64,32])
    with tf.device('/cpu:0'):
        model = Model(inputs=[input_ori, input_fore, input_back, input_pos1, input_pos2, input_pos3, input_pos4],outputs=[out_main, out_middle, out_shape])
    opt = Adam(lr=0.001, amsgrad=True)
    model.compile(optimizer=opt, loss={'out_main': bce_dice_loss, 'out_middle': bce_dice_loss,'out_shape': mean_squared_error},
                      loss_weights={'out_main': 1, 'out_middle': 1, 'out_shape': 0.5},
                      metrics={'out_main': dice_score, 'out_middle': dice_score})
    model.summary()

if __name__ == '__main__':
    SMU_net()

