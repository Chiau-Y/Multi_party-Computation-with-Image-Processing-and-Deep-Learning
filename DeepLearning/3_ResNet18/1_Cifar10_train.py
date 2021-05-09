import cv2
import time
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models


def Add_Layer(inputs):    
        
    # initialization
    out = layers.Conv2D(64,kernel_size=[3,3],padding='same')(inputs)
    out = layers.BatchNormalization()(out)
    out = layers.Activation(activation_function)(out)
    out = layers.AveragePooling2D(pool_size=[2,2],strides=1,padding='same')(out)  
    
    y = Block_Build(64,out)
    y = Block_Build(64,y)
    y = Block_Build(128,y,stride=2)
    y = Block_Build(128,y)
    y = Block_Build(256,y,stride=2)
    y = Block_Build(256,y)
    y = Block_Build(512,y,stride=2)
    y = Block_Build(512,y)
    
    y = layers.GlobalAveragePooling2D()(y)
    y = layers.Dense(10,layers.Activation('softmax'))(y)  
   
    return y

def Block_Build(output_num, input_x, stride=1):
      
    x = layers.Conv2D(output_num,kernel_size=[3,3],strides=stride,padding='same')(input_x) 
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation_function)(x)
    
    x = layers.Conv2D(output_num,kernel_size=[3,3],padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    if stride != 1:         # Residual block
        out_x = layers.Conv2D(output_num,(1,1),strides=stride)(input_x)
        out_x = layers.BatchNormalization()(out_x)  
        x = tf.keras.layers.Add()([out_x,x])         
    else :
        x = tf.keras.layers.Add()([input_x,x]) 
        
    output_x = layers.Activation(activation_function)(x)    
    
    return output_x


def learning_rate_scheduler(epoch, lr): 
    #Say you want to decay linearly by 5 after every 10 epochs the lr
    #(epoch + 1) since it starts from epoch 0
    if (epoch+1) % 3 == 0:
        lr = lr / 10

    return lr

# --------------------------- Main --------------------------- #

#setting
learning_rate_set = 0.0005 # 0.0005/5/4, 0.0005/10/4, 0.0005/10/3
batch_size_set = 10 #10
epochs_set = 20
activation_function = 'softplus'

(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

x_train_normalize = x_train.astype('float32')/255.0
x_test_normalize = x_test.astype('float32')/255.0

y_train = tf.squeeze(y_train,axis=1)
y_test = tf.squeeze(y_test,axis=1)


y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Reserve 10,000 samples for validation
x_val_normalize = x_train_normalize[-10000:]
y_val = y_train[-10000:]
x_train_normalize = x_train_normalize[:-10000]
y_train = y_train[:-10000]

inputs = tf.keras.Input(shape=(32,32,3)) # inpu

model_output = Add_Layer(inputs)

model =  tf.keras.Model(inputs=inputs, outputs=model_output)

model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate_set),
                  loss = 'categorical_crossentropy', metrics = ['accuracy'])

history = model.fit(x = x_train_normalize, y = y_train, validation_data = 
                        (x_val_normalize, y_val),epochs = epochs_set, batch_size = batch_size_set,
                        callbacks=[tf.keras.callbacks.LearningRateScheduler(learning_rate_scheduler, verbose=1)])

plt.subplot(2,1,1)
plt.title("Accuracy & Loss")
plt.plot(history.history['accuracy'], 'r', label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.xlabel('epoch')
plt.ylabel('Accuracy')
plt.legend(loc="lower right",fontsize='small')
plt.subplot(2,1,2)
plt.plot(history.history['loss'], 'r', label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right',fontsize='small')
plt.show()

model.save("ResNet18_b10_e10_lr0005_08.h5")