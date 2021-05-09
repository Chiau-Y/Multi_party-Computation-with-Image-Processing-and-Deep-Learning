import cv2
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models


def VGG16():
    global model
    
    model= models.Sequential() 
     #stack1    
    model.add(layers.Conv2D(64,kernel_size=[3,3],input_shape=(150,150,3),padding='same',activation=activation_function))   
    model.add(layers.Conv2D(64,kernel_size=[3,3],padding='same',activation=activation_function))   
    model.add(layers.MaxPool2D(pool_size=[2,2],strides=2,padding='same'))    
    #stack2    
    model.add(layers.Conv2D(128,kernel_size=[3,3],padding='same',activation=activation_function)) 
    model.add(layers.Conv2D(128,kernel_size=[3,3],padding='same',activation=activation_function) )  
    model.add(layers.MaxPool2D(pool_size=[2,2],strides=2,padding='same'))
     #stack3    
    model.add(layers.Conv2D(256,kernel_size=[3,3],padding='same',activation=activation_function))
    model.add(layers.Conv2D(256,kernel_size=[3,3],padding='same',activation=activation_function))   
    model.add(layers.Conv2D(256,kernel_size=[1,1],padding='same',activation=activation_function))   
    model.add(layers.MaxPool2D(pool_size=[2,2],strides=2,padding='same'))
     #stack4    
    model.add(layers.Conv2D(512,kernel_size=[3,3],padding='same',activation=activation_function))    
    model.add(layers.Conv2D(512,kernel_size=[3,3],padding='same',activation=activation_function))   
    model.add( layers.Conv2D(512,kernel_size=[1,1],padding='same',activation=activation_function))   
    model.add(layers.MaxPool2D(pool_size=[2,2],strides=2,padding='same')) 
     #stack5    
    model.add(layers.Conv2D(512,kernel_size=[3,3],padding='same',activation=activation_function))    
    model.add(layers.Conv2D(512,kernel_size=[3,3],padding='same',activation=activation_function))   
    model.add(layers.Conv2D(512,kernel_size=[1,1],padding='same',activation=activation_function))   
    model.add(layers.MaxPool2D(pool_size=[2,2],strides=2,padding='same'))
    # FC
    model.add(layers.Flatten())
    model.add(layers.Dense(4096,activation=activation_function))    
    model.add(layers.Dense(4096,activation=activation_function))    
    model.add(layers.Dense(2,activation='softmax'))
    
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate_set),
                      loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    history = model.fit(x = x_train, y = y_train, validation_data = 
                            (x_val, y_val),epochs = epochs_set, batch_size = batch_size_set)
    
    
#------------------------------------  Main ----------------------------------#
if __name__ == '__main__':    
    
    x_train = np.load('photos_train.npy')
    y_train = np.load('labels_train.npy',)
    x_val = np.load('photos_val.npy')
    y_val = np.load('labels_val.npy')
    
    #setting
    learning_rate_set = 0.0001
    batch_size_set = 100
    epochs_set = 5
    activation_function = 'softplus'
       
    # x_train_normalize = x_train.astype('float32')/255.0
    # x_val_normalize = x_val.astype('float32')/255.0
    
    y_train = tf.keras.utils.to_categorical(y_train, 2)
    y_val = tf.keras.utils.to_categorical(y_val, 2)    
    
    VGG16()

