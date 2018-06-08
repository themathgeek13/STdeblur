# http://learnandshare645.blogspot.in/2016/06/3d-cnn-in-keras-action-recognition.html

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, Conv3D
from keras.layers import Input, BatchNormalization
from keras.callbacks import ModelCheckpoint

import keras

from numpy.testing import assert_allclose

from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils, generic_utils
from keras.layers.convolutional import Convolution3D

import matplotlib
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
from sklearn import preprocessing

import os
import cv2
import numpy as np
import random

from keras import backend as K

from imgaug import augmenters as iaa
import imgaug as ia

#image specification
img_rows,img_cols,img_depth = 128,128,5     #using only Y channel of YCrCb

def show(img):
    cv2.imshow("image",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def returnpred(i):
    out=X_train[i]
    out=np.expand_dims(out,axis=0)
    out=np.expand_dims(out,axis=0)
    ypred=model.predict(out)
    ypred=ypred.reshape((128,128,3))
    show(ypred)
    show(y_train[i].reshape((128,128,3)))


def customLoss(yTrue, yPred):
    val = K.sum(K.sum(K.sum(K.sum(K.sum(K.square(yTrue-yPred))))))
    return val

def returnlist(N):
    l=os.listdir("blurred_sharp/blurred/")
    b=[int(x.split('.')[0]) for x in l]
    b.sort(key=int)
    l=[str(x)+".png" for x in b]
    return l[:N]

def createdata():
    aug = iaa.CropAndPad(px=((-300,0), (-300,0), (-300, 0), (-300, 0)), pad_mode=ia.ALL, pad_cval=(0,128), keep_size=False)
    l=returnlist(1151)
    x_tr=[]
    y_tr=[]
    for i in range(len(l)-5):
        frames=[]
        for item in l[i:i+5]:
            imgYCC = cv2.cvtColor(cv2.imread("blurred_sharp/blurred/"+item), cv2.COLOR_BGR2YCR_CB)
            frames.append(imgYCC[:,:,0])        #append the Y component
        outY = cv2.cvtColor(cv2.imread("blurred_sharp/sharp/"+l[i+2]), cv2.COLOR_BGR2YCR_CB)[:,:,0]
        frames = np.array(frames)
        #print frames.shape
        frames=np.rollaxis(frames,0,3)
        y_tr.append(outY)
        x_tr.append(frames)
        #print i
    print "Created data"

    images_augX=x_tr[:]
    images_augY=y_tr[:]

    seq_det = aug.to_deterministic()
    x_tr = seq_det.augment_images(x_tr)
    y_tr = seq_det.augment_images(y_tr)

    for i in range(len(images_augX)):
        images_augY[i] = cv2.resize(y_tr[i],(128,128))
        images_augX[i] = cv2.resize(x_tr[i],(128,128))
        #print i
    print "Augmented data"
    return np.array(images_augX), np.array(images_augY)
