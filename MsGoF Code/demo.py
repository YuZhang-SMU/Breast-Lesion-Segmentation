####################################code for MsGoF################################################
# version requirement
# keras = 2.3.1
# tensorflow = 2.1.0
# Feel free to contact the author: shengzhouzhong@foxmail.com
##################################################################################################


import numpy as np
# import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from keras.callbacks import ModelCheckpoint
from keras.layers import Multiply
import keras.layers as KL
import keras.backend as KB
from keras.callbacks import ReduceLROnPlateau
from keras.models import Model, load_model
from keras import optimizers as Op
from PIL import Image
import os
import tensorflow as tf
import pandas

from model.MsGoF import MsGoF

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def train(train_x, train_y, val_x, val_y, save_path):
    pre_model = MsGoF(64)
    Opt = Op.Adam(lr=0.0005, decay=1e-4)
    pre_model.compile(optimizer=Opt, loss={'Output_1': 'binary_crossentropy'}, metrics=['binary_accuracy'])
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=20, verbose=1, mode='min')
    checkpoint_name = save_path+'model.h5'
    model_checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_binary_accuracy', save_best_only=True, mode='max',
                                       save_weights_only=False, verbose=1)
    callbacks_list = [reduce_lr, model_checkpoint]
    pre_model.fit(x=[train_x], y=[train_y], epochs=300, batch_size=10, verbose=1, callbacks=callbacks_list,
                  validation_data=([val_x], [val_y]), shuffle=True)
    # del pre_model


def prediction(model_name, val_x, y_val, path):
    test_model = load_model(str(model_name), custom_objects={'tf': tf, 'KB': KB, 'KL': KL, 'Multiply': Multiply})
    pre = test_model.predict(np.array(val_x), batch_size=10)
    fpr1, tpr1, threshold = roc_curve(y_val, pre[:, 0])
    J = tpr1 - fpr1
    ix = np.argmax(J)
    best_thre = threshold[ix]
    pre_class = (pre >= best_thre).astype(int)
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(np.squeeze(pre_class))):
        if y_val[i] == 1 and pre_class[i, 0] == 1:
            TP += 1
        elif y_val[i] == 0 and pre_class[i, 0] == 0:
            TN += 1
        elif y_val[i] == 1 and pre_class[i, 0] == 0:
            FN += 1
        elif y_val[i] == 0 and pre_class[i, 0] == 1:
            FP += 1
    if TP + TN + FP + FN != len(np.squeeze(pre_class)):
        print('number of sample is wrong！！！！！！！')

    acc = (TP + TN) / (TP + TN + FP + FN + 0.001)
    precision = TP / (TP + FP + 0.001)
    sensitivity = TP / (TP + FN + 0.001)
    f1 = precision * sensitivity * 2 / (precision + sensitivity + 0.001)
    specificity = TN / (TN + FP + 0.001)
    fpr = FP / (TN + FP + 0.001)
    tpr = TP / (TP + FP + 0.001)
    roc_auc = auc(fpr1, tpr1)
    roc_auc = pandas.DataFrame([roc_auc])
    acc = pandas.DataFrame([acc])
    precision = pandas.DataFrame([precision])
    sensitivity = pandas.DataFrame([sensitivity])
    f1 = pandas.DataFrame([f1])
    specificity = pandas.DataFrame([specificity])
    writer = pandas.ExcelWriter(str(path)+'results.xls')
    acc.to_excel(writer, sheet_name='Sheet1', header=['accuracy'], startcol=0, index=False)
    sensitivity.to_excel(writer, sheet_name='Sheet1', header=['sensitivity'], startcol=2, index=False)
    f1.to_excel(writer, sheet_name='Sheet1', header=['f1_score'], startcol=3, index=False)
    specificity.to_excel(writer, sheet_name='Sheet1', header=['specificity'], startcol=4, index=False)
    roc_auc.to_excel(writer, sheet_name='Sheet1', header=['auc'], startcol=5, index=False)
    writer.save()
    print('acc:', acc)
    print('auc:', roc_auc)
    print('precision:', precision)
    print('sensitivity:', sensitivity)
    print('f1_score:', f1)
    print('specificity:', specificity)


def img_load(Dir_path, size):
    img = Image.open(Dir_path)
    img1 = img.resize((size, size))
    img1 = np.array(img1).astype('float32')
    img1 = img1.reshape((size, size, 1))
    img1 = img1 / 255.
    return img1


def get_data(dir_path, name, size):
    data = []
    for i in range(len(name)):
        img = img_load(dir_path + str(name[i]), size)
        data.append(img)
    return data

def read_files(path):
    files = os.listdir(path)
    return files

import xlrd

def read_label(file_path):
    wb = xlrd.open_workbook(file_path)
    sheet = wb.sheet_by_name('Sheet1')
    label = sheet.col_values(1)
    return label

def main(train_image_path, train_class, val_image_path, val_class, test_image_path, test_class, model_path, save_pre_path):
    train_name = read_files(train_image_path)
    train_data = get_data(train_image_path, train_name, 64)

    val_name = read_files(val_image_path)
    val_data = get_data(val_image_path, val_name, 64)

    train(train_data, train_class, val_data, val_class, model_path)
    prediction(model_path, test_image_path, test_class, save_pre_path)


if __name__ == "__main__":
    #################training######################
    # this is a demo for MsGoF, you need to prepare your own data to train your model.
    print('trainging model.................')
    train_image_path = './data/train_image/'
    val_image_path = './data/val_image/'
    model_path = './checkpoint/'
    try:
        os.makedirs(model_path)
    except FileExistsError:
        pass
    train_class = read_label('./data/train_class.xls')
    val_class = read_label('./data/val_class.xls')
    train_name = read_files(train_image_path)
    train_data = get_data(train_image_path, train_name, 64)
    val_name = read_files(val_image_path)
    val_data = get_data(val_image_path, val_name, 64)
    train(train_data, train_class, val_data, val_class, model_path)


    print('###################################################')
    print('testing model.................')
    test_image_path = './data/test_image/'
    save_pre_path = './prediction/'
    try:
        os.makedirs(save_pre_path)
    except FileExistsError:
        pass
    test_name = read_files(test_image_path)
    test_data = get_data(test_image_path, test_name, 64)
    test_class = read_label('./data/test_class.xls')
    prediction(model_path+'model.h5', test_data, test_class, save_pre_path)



