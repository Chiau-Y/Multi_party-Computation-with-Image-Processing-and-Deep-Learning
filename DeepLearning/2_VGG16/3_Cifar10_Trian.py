import cv2
import time
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

def learning_rate_scheduler(epoch, lr): 
    #Say you want to decay linearly by 5 after every 10 epochs the lr
    #(epoch + 1) since it starts from epoch 0
    if (epoch+1) % 3 == 0:
        lr = lr / 10

    return lr

#setting
learning_rate_set = 0.0005
batch_size_set = 50
epochs_set = 15
activation_function = 'softplus'

(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

x_train_normalize = x_train.astype('float32')/255.0
x_test_normalize = x_test.astype('float32')/255.0

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Reserve 10,000 samples for validation
x_val_normalize = x_train_normalize[-10000:]
y_val = y_train[-10000:]
x_train_normalize = x_train_normalize[:-10000]
y_train = y_train[:-10000]


model= models.Sequential() 
 #stack1    
model.add(layers.Conv2D(64,kernel_size=[3,3],input_shape=(32,32,3),padding='same',activation=activation_function))   
model.add(layers.Conv2D(64,kernel_size=[3,3],padding='same',activation=activation_function))   
model.add(layers.AveragePooling2D(pool_size=[2,2],strides=2,padding='same'))    
#stack2    
model.add(layers.Conv2D(128,kernel_size=[3,3],padding='same',activation=activation_function)) 
model.add(layers.Conv2D(128,kernel_size=[3,3],padding='same',activation=activation_function) )  
model.add(layers.AveragePooling2D(pool_size=[2,2],strides=2,padding='same'))
 #stack3    
model.add(layers.Conv2D(256,kernel_size=[3,3],padding='same',activation=activation_function))
model.add(layers.Conv2D(256,kernel_size=[3,3],padding='same',activation=activation_function))   
model.add(layers.Conv2D(256,kernel_size=[1,1],padding='same',activation=activation_function))   
model.add(layers.AveragePooling2D(pool_size=[2,2],strides=2,padding='same'))
 #stack4    
model.add(layers.Conv2D(512,kernel_size=[3,3],padding='same',activation=activation_function))    
model.add(layers.Conv2D(512,kernel_size=[3,3],padding='same',activation=activation_function))   
model.add( layers.Conv2D(512,kernel_size=[1,1],padding='same',activation=activation_function))   
model.add(layers.AveragePooling2D(pool_size=[2,2],strides=2,padding='same')) 
 #stack5    
model.add(layers.Conv2D(512,kernel_size=[3,3],padding='same',activation=activation_function))    
model.add(layers.Conv2D(512,kernel_size=[3,3],padding='same',activation=activation_function))   
model.add(layers.Conv2D(512,kernel_size=[1,1],padding='same',activation=activation_function))   
model.add(layers.AveragePooling2D(pool_size=[2,2],strides=2,padding='same'))
# FC
model.add(layers.Flatten())
model.add(layers.Dense(256,activation=activation_function))    
model.add(layers.Dense(256,activation=activation_function))    
model.add(layers.Dense(10,activation='softmax'))

model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate_set),
                  loss = 'categorical_crossentropy', metrics = ['accuracy'])

history = model.fit(x = x_train_normalize, y = y_train, validation_data = 
                        (x_val_normalize, y_val),epochs = epochs_set, batch_size = batch_size_set,
                        callbacks=[tf.keras.callbacks.LearningRateScheduler(learning_rate_scheduler, verbose=1)])

model.save("VGG16_b50_e15.h5")