import cv2
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


print("Load .npy")
cp = np.load('D:/Master_2019_2021/Muti_Party_Computation/CloudComputation/CP_cpu.npy')
colab = np.load('D:/Master_2019_2021/Muti_Party_Computation/CloudComputation/Colab_cpu.npy')
ImageP = np.load('D:/Master_2019_2021/Muti_Party_Computation/CloudComputation/IP_cpu.npy')

plt.figure(0,figsize=(15,5))

tick_spacing = 1.5

plt.subplot(2,2,1)
plt.plot(cp[1],cp[0].astype(np.float),LineWidth = 5)
plt.ylabel('CPU Utilization (%)')
plt.xticks([])
plt.ylim(0,100)
plt.legend(labels=['Computer'], loc='upper right',fontsize = 'x-large')

colab_0 = colab[0].astype(np.float)

pad_front = cp[1][0:list(cp[1]).index(colab[1][0])]
pad_back = cp[1][list(cp[1]).index(colab[1][-1]):len(cp[1])]

y_axis = colab_0[0:len(colab[0])]
x_axis = colab[1][0:len(colab[1])]

new_x_axis = np.hstack((pad_front,x_axis,pad_back))
new_y_axis = np.hstack((np.zeros(len(pad_front)),y_axis,np.zeros(len(pad_back))))

a = plt.subplot(2,2,3)
plt.plot(new_x_axis,new_y_axis ,color='#db7093',LineWidth = 5)
plt.ylabel('CPU Utilization (%)')
plt.xlabel('Time')
plt.ylim(0,100)
plt.xticks(rotation=45)
plt.legend(labels=['Colab'], loc='upper right',fontsize = 'x-large')
a.axes.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

plt.subplot(1,2,2)
plt.plot(ImageP[1],ImageP[0].astype(np.float) ,color='#FFA500',LineWidth = 5)
plt.ylabel('CPU Utilization (%)')
plt.xlabel('Time')
plt.ylim(0,100)
plt.xticks(rotation=45)
plt.legend(labels=['Computer(Image Processing)'], loc='upper right',fontsize = 'x-large')

plt.show()   
    