from google.colab import drive
drive.mount('/content/drive')

import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models 

import os
from scipy import misc, ndimage
import time
import psutil
from datetime import datetime
from threading import Timer
import matplotlib.pyplot as plt
# ------------------------------------Subroutine------------------------------------
def hello():
    global cpu,time_axis   
    cpu.append(psutil.cpu_percent(0.1))
    time_axis.append(datetime.utcnow().strftime("%M:%S"))
    if len(cpu) != len(time_axis):
      time_axis.pop()
    
class RepeatingTimer(Timer): 
    def run(self):
        while not self.finished.is_set():
            self.function(*self.args, **self.kwargs)
            self.finished.wait(self.interval)

# ---------------------- Main ---------------------- #  

n = 6
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

  cpu = []
  time_axis = []

  t_cpu = RepeatingTimer(1, hello)
  t_cpu.start()
  time.sleep(3)

  n = 6
  model = tf.keras.models.load_model("/content/drive/MyDrive/Model/ResNet18_b10_e13_lr0005_08")

  # predict the same model but no softmax
  model2 = tf.keras.Model(inputs=model.input, outputs=model.get_layer(index=-1).output)
  ans = []
  # predict
  for k in range (n):
    ans.append(model2.predict(img_party[k].reshape(1,32,32,3)))

  time.sleep(3)
  t_cpu.cancel() 
  data_x = cpu[0:len(cpu)]
  data_y = time_axis[0:len(time_axis)]

  for k in range (n):
    np.save('/content/drive/MyDrive/Party'+str(k+1)+'_dp_complete.npy', ans[k])
    print("=== Party{} Done ===\n".format(k+1))

 

  plt.figure(20)
  plt.plot(data_y,data_x, label='CPU')
  plt.ylabel('CPU%')
  plt.ylim(0,100)
  plt.xticks(rotation=45)
  plt.show()       

  data = []
  data.append(data_x)
  data.append(data_y)
  np.save('/content/drive/MyDrive/Colab_cpu', data)

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