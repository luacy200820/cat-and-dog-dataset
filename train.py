from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import tensorflow as tf 
import keras 
import sys
import cv2
import os
import numpy as np
import glob
from keras import backend as K
from keras.models import Model
import pandas as pd
# from focal_loss import BinaryFocalLoss
IMAGE_SIZE = (224,224)
def train(lf):
    # 影像類別數
    # NUM_CLASSES = 2
    BATCH_SIZE = 8
    NUM_EPOCHS =20
    absoulate_path = os.path.abspath(__file__)
    fileDirectory = os.path.dirname(absoulate_path) # 資料夾路徑
    train_dog = sorted(glob.glob(fileDirectory+'\\training_dataset\\Dog\\*.jpg'))
    train_cat = sorted(glob.glob(fileDirectory+'\\training_dataset\\Cat\\*.jpg'))
    
    valid_dog = sorted(glob.glob(fileDirectory+'\\validation_dataset\\Dog\\*.jpg'))
    valid_cat = sorted(glob.glob(fileDirectory+'\\validation_dataset\\Cat\\*.jpg'))
    
    cls_list = ['cats', 'dogs']
    
    image_size = (224,224)

    train_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            fileDirectory+'/training_dataset',  # this is the target directory
            target_size=image_size,  # all images will be resized to 224x224
            batch_size=32,
            class_mode='binary')

    validation_datagen = ImageDataGenerator(rescale=1.0/255)
    validation_generator = validation_datagen.flow_from_directory(
            fileDirectory+'/validation_dataset',  # this is the target directory
            target_size=image_size,  # all images will be resized to 224x224
            batch_size=32,
            class_mode='binary')
    
    model = model_train(lf)
    
    model.fit(train_generator,
            validation_data = validation_generator,
            epochs = NUM_EPOCHS,
            )

    model.save(fileDirectory+f'\\{lf}_resnet50_non_2.h5')
    
    score = model.evaluate(validation_generator)
    # print('focal evaluate',focal_score)
    return score
    
def model_train(lf):
    
    net = ResNet50(include_top=False, weights='imagenet', input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3))
    x = net.output
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=net.input, outputs=output)
    if lf == 'focal':
        loss_function=tf.keras.losses.BinaryFocalCrossentropy(gamma=1.0)
    elif lf == 'binary':
        loss_function = 'binary_crossentropy'
    model.compile(optimizer=Adam(lr=1e-5), loss=loss_function, metrics=['accuracy'])
    return model

    
score =train("focal")
