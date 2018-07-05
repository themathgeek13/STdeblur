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
K.set_image_dim_ordering('tf')

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
        print(frames.shape)
        frames=np.rollaxis(frames,0,3)
        y_tr.append(outY)
        x_tr.append(frames)
        print(i)

    images_augX=x_tr[:]
    images_augY=y_tr[:]

    seq_det = aug.to_deterministic()
    x_tr = seq_det.augment_images(x_tr)
    y_tr = seq_det.augment_images(y_tr)

    for i in range(len(images_augX)):
        images_augY[i] = cv2.resize(y_tr[i],(128,128))
        images_augX[i] = cv2.resize(x_tr[i],(128,128))
        print(i)
    return images_augX, images_augY

def generator(batch_size):
    aug = iaa.CropAndPad(px=((-100,0), (-50,0), (-100, 0), (-50, 0)), pad_mode=ia.ALL, pad_cval=(0,128), keep_size=False)
    l=os.listdir("blurred_sharp/blurred/")
    l=random.sample(l,batch_size)
    x_tr=[]
    y_tr=[]
    while True:
        for i in range(len(l)-5):
            frames=[]
            for item in l[i:i+5]:
                imgRGB = cv2.imread("blurred_sharp/blurred/"+item)
                frames.append(imgRGB)
                #imgYCC = cv2.cvtColor(cv2.imread("blurred_sharp/blurred/"+item), cv2.COLOR_BGR2YCR_CB)
                #frames.append(imgYCC[:,:,0])        #append the Y component
            #outY = cv2.cvtColor(cv2.imread("blurred_sharp/sharp/"+l[i+2]), cv2.COLOR_BGR2YCR_CB)[:,:,0]
            #outY = cv2.resize(outY,(128,128))
            frames=np.array(frames)
            frames=frames.reshape((720,720,15))
            out = cv2.imread("blurred_sharp/sharp/"+l[i+2])
            out=cv2.resize(out,(128,128))
            y_tr.append(out)
            x_tr.append(frames)
            print(i)

        images_aug=aug.augment_images(x_tr)

        for i in range(len(images_aug)):
            images_aug[i]=cv2.resize(images_aug[i],(128,128))
        images_aug=np.expand_dims(np.array(images_aug),1)
        yield images_aug, np.array(y_tr)

def spatempblock(inputshape):
    l1 = Conv2D(64, (3,3), padding="same")(inputshape)
    bn1 = BatchNormalization()(l1)
    ac1 = Activation('relu')(bn1)
    l2 = Conv2D(64, (3,3), padding="same")(ac1)
    bn2 = BatchNormalization()(l2)

    return bn2

X_train,y_train=createdata()
np.save("x_train.npy",X_train)
np.save("y_trin.npy",y_train)

#exit(0)
X_train=np.load("x_train.npy")
y_train=np.load("y_trin.npy")
X_train=np.array(X_train)
y_train=np.array(y_train)
print(y_train.shape)
#X_train=np.array(X_train)
num_samples=len(X_train)
print(X_train.shape)

X_train = X_train/255.0
y_train = y_train/255.0

out = np.zeros((num_samples,1,img_rows,img_cols,img_depth))

yt = np.zeros((num_samples,img_rows,img_cols,1))

for h in range(num_samples):
    # out[h,0,:,:,:]=X_train[h,:,:,:]
    yt[h,:,:,0]=y_train[h,:,:]
# print out.shape
print(yt.shape)
#X_train=np.swapaxes(X_train, 2,3)
#print X_train.shape
#X_train=np.swapaxes(X_train, 1, 2)

input_img = Input(shape=(img_rows,img_cols,img_depth))

l1 = Conv2D(32, (3,3), input_shape=(img_rows, img_cols, img_depth),padding="same", activation='relu',data_format="channels_last")(input_img)
l2 = Conv2D(64, (3,3), activation='relu', padding="same",data_format="channels_last" )(l1)

bn4 = spatempblock(l2)
out1 = keras.layers.add([l2, bn4])
bn6 = spatempblock(out1)
out2 = keras.layers.add([out1, bn6])
bn8 = spatempblock(out2)
out3 = keras.layers.add([out2, bn8])
bn10 = spatempblock(out3)
out4 = keras.layers.add([out3, bn10])
bn12 = spatempblock(out4)
out5 = keras.layers.add([out4, bn12])
bn14 = spatempblock(out5)
out6 = keras.layers.add([out5, bn14])
bn16 = spatempblock(out6)
out7 = keras.layers.add([out6, bn16])
bn18 = spatempblock(out7)
out8 = keras.layers.add([out7, bn18])
bn20 = spatempblock(out8)
out9 = keras.layers.add([out8, bn20])
bn22 = spatempblock(out9)
out10 = keras.layers.add([out9, bn22])
bn24 = spatempblock(out10)
out11 = keras.layers.add([out10, bn24])
bn26 = spatempblock(out11)
out12 = keras.layers.add([out11, bn26])
bn28 = spatempblock(out12)
out13 = keras.layers.add([out12, bn28])
bn30 = spatempblock(out13)
out14 = keras.layers.add([out13, bn30])
bn32 = spatempblock(out14)
out15 = keras.layers.add([out14, bn32])
out15 = keras.layers.add([out15, l2])

l33 = Conv2D(256, (3,3), activation='relu', padding="same")(out15)
l34 = Conv2D(256, (3,3), activation='relu', padding="same")(l33)
l35 = Conv2D(1, (3,3), padding="same")(l34)

model = Model(inputs=input_img, outputs=l35)
model.compile(optimizer='rmsprop', loss=customLoss)
model.summary()
X_train_new, X_val_new, y_train_new,y_val_new =  train_test_split(X_train, yt, test_size=0.2, random_state=4)
hist = model.fit(X_train_new, y_train_new, validation_data=(X_val_new,y_val_new),
          batch_size=4, epochs=10,shuffle=True)
#model.fit(X_train, y_train, batch_size=5, epochs=10)

show(X_train[0,:,:,0])
show(X_train[1,:,:,1])
show(X_train[0,:,:,2])
