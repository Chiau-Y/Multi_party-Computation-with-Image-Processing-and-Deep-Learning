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
batch_size_set = 100
epochs_set = 5

x_train_padded = np.pad(x_train[:,], ((0,0),(2,2), (2,2)), 'constant')
x_test_padded = np.pad(x_test[:,], ((0,0),(2,2), (2,2)), 'constant')

x_train_reshaped = x_train_padded.reshape((60000, 32, 32, 1))
x_test_reshaped = x_test_padded.reshape((10000, 32, 32, 1))

#pretreatment
x_train_normalize = x_train_reshaped.astype('float32')/255.0
x_test_normalize = x_test_reshaped.astype('float32')/255.0

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

model= models.Sequential()
model.add(layers.Conv2D(filters=6,kernel_size=(5,5),input_shape=(32,32,1),activation='softplus'))
model.add(layers.AveragePooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(filters=16, kernel_size=(5,5),activation='softplus'))
model.add(layers.AveragePooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(120,activation='softplus'))
model.add(layers.Dense(84,activation='softplus'))
model.add(layers.Dense(10,activation='softmax'))

model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate_set),
              loss = 'mean_squared_error', metrics = ['accuracy'])

# mean_squared_error/sparse_categorical_crossentropy/categorical_crossentropy

history = model.fit(x = x_train_normalize, y = y_train, validation_data = 
                    (x_test_normalize, y_test),epochs = epochs_set, batch_size = batch_size_set)

# ----------------------------------------------------------------------------------- #
testing_num = random.randint(0,10000-1)


probabilities = model.predict(x_test_normalize)

class_name = ['0', '1', '2', '3', '4','5', '6', '7', '8', '9']

# show image
probabilities_array, true_label, img = probabilities[testing_num], y_test[testing_num], x_test_normalize[testing_num]
plt.figure(3)
plt.grid(False)
plt.xticks([])
plt.yticks([])
plt.imshow(img[:,:,0], cmap='binary')
predicted_label = np.argmax(probabilities_array)
plt.xlabel("{} {:2.0f}% ".format(class_name[predicted_label], 100*np.max(probabilities_array)))

# estimate this test image
probabilities_array, true_label = probabilities[testing_num], y_test[testing_num]
plt.figure(2)
plt.grid(False)
plt.xticks(range(10), class_name, fontsize=15, rotation=45)
plt.ylim([0, 1])
plt.bar(range(10), probabilities_array)

 
