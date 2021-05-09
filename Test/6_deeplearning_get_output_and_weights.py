import cv2
import time
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
# from tensorflow.examples.tutorials.mnist import input_data

# sess = tf.compat.v1.InteractiveSession()
# sess.close()
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

#setting
learning_rate_set = 0.001
batch_size_set = 1
epochs_set = 1


x_train_padded = np.pad(x_train[:,], ((0,0),(2,2), (2,2)), 'constant')
x_test_padded = np.pad(x_test[:,], ((0,0),(2,2), (2,2)), 'constant')

x_train_reshaped = x_train_padded.reshape((60000, 32, 32, 1))
x_test_reshaped = x_test_padded.reshape((10000, 32, 32, 1))

#pretreatment
x_train_normalize = x_train_reshaped.astype('float32')/255.0
x_test_normalize = x_test_reshaped.astype('float32')/255.0

x_train_normalize1  = x_train_normalize[0:1]
x_test_normalize1 = x_test_normalize[0:1]

model= models.Sequential()
model.add(layers.Conv2D(filters=10,kernel_size=(32,32),input_shape=(32,32,1)))
# model.add(layers.Dense(10,activation='softmax'))

model.compile(optimizer = tf.optimizers.Adam(learning_rate = learning_rate_set),
              loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

history = model.fit(x = x_train_normalize1, y = y_train[0:1], validation_data = (x_test_normalize1, y_test[0:1]),
                    epochs = epochs_set, batch_size = batch_size_set)


predict = model.predict(x_test_normalize[1:2])
 
# dense_layer.set_weights(weights)
# print('after set weights...')
dense_layer = model.get_layer(index=0)
a = dense_layer.weights
# a[0].shape # kernel weight
# a[1].shape # bias weigt
b = a[0].numpy() # tensor to array
c = b[:,:,:,0] # first kernel

ww = x_test_normalize[1]*c

print()
# print(predict, predict.shape)
print(sum(sum(ww[:,:,0])))