import os
import random
import numpy as np
from os import listdir
import tensorflow as tf
from shutil import copyfile
import matplotlib.pyplot as plt
from matplotlib.image import imread
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array


def Label_Resize():
    global image, output
    
    if file.startswith('cat',8):
        output = 1.0    # determine class 
    else:
        output = 0.0   # determine class 
    image = load_img(file , target_size=(200, 200))   # load image
    # Convert the image pixels to a numpy array
    image = img_to_array(image)
    
# def VGG16():
    
# define location of dataset
folder = 'train_2/'
imagePaths = []
for file in listdir(folder):
    imagePath = os.path.join(folder, file) # create path to dogs and cats
    imagePaths.append(imagePath)
    
random.seed(42)
random.shuffle(imagePaths)

val_ratio = 0.2 # 0.2 for validation
photos_train, labels_train, photos_val, labels_val = [], [], [], []
# enumerate files in the directory
for i, file in enumerate(imagePaths): 
    
    if (i%1000 == 0): 
        print("")
        print("Resize {} files".format(i+1), end ='')
    if (i%100 == 0):
        print('.', end = '')   
        
    if random.random() < val_ratio:
        Label_Resize()
        # Reshape data for the model
        photos_val.append(image.reshape((image.shape[0], image.shape[1], image.shape[2])))
        labels_val.append(output)        
                        
    else:       
        Label_Resize()
        # Reshape data for the model
        photos_train.append(image.reshape((image.shape[0], image.shape[1], image.shape[2])))
        labels_train.append(output)
      
# convert to a numpy arrays
photos_train = np.array(photos_train, dtype='uint8')
labels_train = np.array(labels_train)
photos_val = np.array(photos_val, dtype='uint8')
labels_val = np.array(labels_val)

# save the reshaped photos
np.save('photos_train.npy', photos_train)
np.save('labels_train', labels_train)
np.save('photos_val.npy', photos_val)
np.save('labels_val', labels_val)