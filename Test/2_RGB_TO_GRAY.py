import matplotlib.pyplot as plt # plt 用於顯示圖片
import matplotlib.image as mpimg # mpimg 用於讀取圖片
import numpy as np

img = mpimg.imread('F:\Master_2019_2021\Muti_Party_Computation\python/2.jpg') 

img_size = np.shape (img)
img_R = img[:,:,0]
img_G = img[:,:,1]
img_B = img[:,:,2]
img_Gray = np.zeros([img_size[0], img_size[1]], dtype='uint8')

for i in range (img_size[0]):
    for j in range (img_size[1]):
        img_Gray[i][j] = int(((img_R[i][j])*299 + (img_G[i][j])*587 + (img_B[i][j])*114 + 500) / 1000)
        
plt.figure(0)
plt.imshow(img_Gray, cmap='Greys_r') # 顯示圖片
plt.axis('off') # 不顯示座標軸
plt.show()