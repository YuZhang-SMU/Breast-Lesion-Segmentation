# coding=utf-8

import keras.backend as K
from keras.losses import binary_crossentropy
from keras.utils.generic_utils import get_custom_objects
from metric import dice_coef
import tensorflow as tf

# ============================== Dice Losses ================================

def dice_loss(y_true, y_pred, class_weights=1., smooth=0.0001, per_image=True):
    r"""Dice loss function for imbalanced datasets:

    .. math:: L(precision, recall) = 1 - (1 + \beta^2) \frac{precision \cdot recall}
        {\beta^2 \cdot precision + recall}

    Args:
        gt: ground truth 4D keras tensor (B, H, W, C)
        pr: prediction 4D keras tensor (B, H, W, C)
        class_weights: 1. or list of class weights, len(weights) = C
        smooth: value to avoid division by zero
        per_image: if ``True``, metric is calculated as mean over images in batch (B),
            else over whole batch

    Returns:
        Dice loss in range [0, 1]

    """
    return 1 - dice_coef(y_true, y_pred, class_weights=class_weights, smooth=smooth, per_image=per_image, beta=1.)


def bce_dice_loss(y_true, y_pred, bce_weight=1., smooth=0.0001, per_image=True):
    bce = K.mean(binary_crossentropy(y_true, y_pred))
    loss = bce_weight * bce + dice_loss(y_true, y_pred, smooth=smooth, per_image=per_image)
    return loss

# Update custom objects
get_custom_objects().update({
    'bce_dice_loss': bce_dice_loss,
})
