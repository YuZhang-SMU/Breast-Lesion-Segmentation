# coding=utf-8

import keras.backend as K
from keras.utils.generic_utils import get_custom_objects
import tensorflow as tf

#============================Dice score======================================

def dice_score(gt, pr, class_weights=1., beta=1., smooth=0.0001, per_image=True):
    if per_image:
        axes = [1, 2]
    else:
        axes = [0, 1, 2]

    tp = K.sum(gt * tf.round(pr), axis=axes)
    fp = K.sum(tf.round(pr), axis=axes) - tp
    fn = K.sum(gt, axis=axes) - tp

    score = ((1 + beta ** 2) * tp + smooth) \
            / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    if per_image:
        score = K.mean(score, axis=0)

    # weighted mean per class
    score = K.mean(score * class_weights)

    return score

def dice_coef(gt, pr, class_weights=1., beta=1., smooth=0.0001, per_image=True):
    if per_image:
        axes = [1, 2]
    else:
        axes = [0, 1, 2]

    # calculate weights
    count_neg = tf.reduce_sum(1. - gt, axis=axes)
    count_neg = tf.cast(count_neg, dtype=tf.float32)
    count_pos = tf.reduce_sum(gt, axis=axes)
    count_pos = tf.cast(count_pos, dtype=tf.float32)
    weights = tf.div(count_neg, count_pos+count_neg)
    tp = K.sum(gt * pr, axis=axes)
    fp = K.sum(pr, axis=axes) - tp
    fn = K.sum(gt, axis=axes) - tp
    tp2 = K.sum((1-gt)*(1-pr), axis=axes)
    fp2 = K.sum((1-pr), axis=axes)
    fn2 = K.sum((1-gt), axis=axes)

    score = (weights*((1 + beta ** 2) * tp + smooth) \
                / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)+\
                (1-weights)*((1 + beta ** 2) * tp2 + smooth)\
                / ((1 + beta ** 2) * tp2 + beta ** 2 * fn2 + fp2 + smooth))

    # mean per image
    if per_image:
        score = K.mean(score, axis=0)

    # weighted mean per class
    score = K.mean(score * class_weights)

    return score

get_custom_objects().update({
    'dice_score': dice_score,
})
