import matplotlib.pyplot as plt
import numpy as np
import cv2

def salt(imgg, n):    
    for k in range(n):    
        i = int(np.random.random() * imgg.shape[1])
        j = int(np.random.random() * imgg.shape[0])
        if imgg.ndim == 2:     
            imgg[j,i] = 255    
        elif imgg.ndim == 3:     
            imgg[j,i,0]= 255    
            imgg[j,i,1]= 255    
            imgg[j,i,2]= 255    
    return imgg   

def pepper(imggg, n):
    for k in range(n):
        i = int(np.random.random() * imggg.shape[1])
        j = int(np.random.random() * imggg.shape[0])
        if imggg.ndim == 2:
            imggg[j, i] == 0
        elif imggg.ndim == 3:
            imggg[j,i,0]= 0    
            imggg[j,i,1]= 0    
            imggg[j,i,2]= 0
    return imggg

img = np.load('D:/Master_2019_2021/Muti_Party_Computation/ErrorCorrection/parties/ImageProcessing/Party6_original.npy') 
mean = 0
var = 100
sigma = var ** 0.5
gaussian = np.random.normal(mean, sigma, (img.shape[0], img.shape[1])) #  np.zeros((224, 224), np.float32)

# ------------------ Gaussian noise ------------------ #
noisy_image = np.zeros(img.shape, np.float32)
if len(img.shape) == 2:
    noisy_image = img + 50*np.trunc(gaussian)
else:
    noisy_image[:, :, 0] = img[:, :, 0] + 10*np.trunc(gaussian)
    noisy_image[:, :, 1] = img[:, :, 1] + 10*np.trunc(gaussian)
    noisy_image[:, :, 2] = img[:, :, 2] + 10*np.trunc(gaussian)


# ------------------ salt and pepper ------------------ #
saltRe = salt(noisy_image, 1000)
result = pepper(saltRe, 1000).astype('int64')%33292801
  
np.save('D:/Master_2019_2021/Muti_Party_Computation/ErrorCorrection/parties/ImageProcessing/Party6.npy',result) 

 
plt.figure(0)
plt.imshow(img, cmap='Greys_r') # 顯示圖片
plt.axis('off') # 不顯示座標軸
# plt.figure(1)
# plt.imshow(noisy_image, cmap='Greys_r') # 顯示圖片
# plt.axis('off') # 不顯示座標軸
# plt.figure(2)
# plt.imshow(saltRe, cmap='Greys_r') # 顯示圖片
# plt.axis('off') # 不顯示座標軸
plt.figure(3)
plt.imshow(result, cmap='Greys_r')
plt.axis('off') # 不顯示座標軸 