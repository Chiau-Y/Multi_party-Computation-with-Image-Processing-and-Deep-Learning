from google.colab import drive
drive.mount('/content/drive')

import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models 
# ---------------------- Main ---------------------- #  
n = 6
model = tf.keras.models.load_model("/content/drive/MyDrive/Model/ResNet18_b10_e13_lr0005_08")

# predict the same model but no softmax
model2 = tf.keras.Model(inputs=model.input, outputs=model.get_layer(index=-1).output)

print("\n------- Start --------\n")
while(True):
  

  filepath = '/content/drive/MyDrive/Party'+str(n)+'_dp.npy'
  filepath_complete = '/content/drive/MyDrive/Party'+str(n)+'_dp_complete.npy'

  # 檢查原檔是否存在
  while(not os.path.isfile(filepath)):
    pass

  if (os.path.isfile(filepath_complete)):
    !rm /content/drive/MyDrive/Party1_dp_complete.npy
    !rm /content/drive/MyDrive/Party2_dp_complete.npy
    !rm /content/drive/MyDrive/Party3_dp_complete.npy
    !rm /content/drive/MyDrive/Party4_dp_complete.npy
    !rm /content/drive/MyDrive/Party5_dp_complete.npy
    !rm /content/drive/MyDrive/Party6_dp_complete.npy

  # 檢查輸出檔是否刪除
  while(os.path.isfile(filepath_complete)):
    pass 

  img_party = []
  for i in range(n):
    img_party.append(np.load('/content/drive/MyDrive/Party'+str(i+1)+'_dp.npy')) 

  img_size = np.shape(img_party[0])

  # predict
  for k in range (n):
    np.save('/content/drive/MyDrive/Party'+str(k+1)+'_dp_complete.npy', model2.predict(img_party[k].reshape(1,32,32,3)))
    print("=== Party{} Done ===\n".format(k+1)) 

  !rm /content/drive/MyDrive/Party1_dp.npy
  !rm /content/drive/MyDrive/Party2_dp.npy
  !rm /content/drive/MyDrive/Party3_dp.npy
  !rm /content/drive/MyDrive/Party4_dp.npy
  !rm /content/drive/MyDrive/Party5_dp.npy
  !rm /content/drive/MyDrive/Party6_dp.npy

  # 檢查原檔是否刪除
  while(os.path.isfile(filepath)):
    pass 

  print("------- The End --------")