# coding=utf-8

import keras.backend as K
from keras.losses import binary_crossentropy
from keras.utils.generic_utils import get_custom_objects
from metric import dice_coef
import tensorflow as tf

# ============================== Dice Losses ================================
def dice_loss(y_true, y_pred, class_weights=1., smooth=0.0001, per_image=True):

    return 1 - dice_coef(y_true, y_pred, class_weights=class_weights, smooth=smooth, per_image=per_image, beta=1.)


def bce_dice_loss(y_true, y_pred, bce_weight=1., smooth=0.0001, per_image=True):
    bce = K.mean(binary_crossentropy(y_true, y_pred))
    loss = bce_weight * bce + dice_loss(y_true, y_pred, smooth=smooth, per_image=per_image)
    return loss

# Update custom objects
get_custom_objects().update({
    'bce_dice_loss': bce_dice_loss,
})
