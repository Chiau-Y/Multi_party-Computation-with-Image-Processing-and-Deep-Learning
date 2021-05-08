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

# img = cv2.imread('D:/Master_2019_2021/Muti_Party_Computation/ErrorCorrection/parties/party6_original.bmp')
img = np.load('D:/Master_2019_2021/Muti_Party_Computation/ErrorCorrection/parties/ImageProcessing/party2_real_original.npy') 
mean = 0
var = 100
sigma = var ** 0.5
gaussian = np.random.normal(mean, sigma, (img.shape[0], img.shape[1])) #  np.zeros((224, 224), np.float32)

# ------------------ Gaussian noise ------------------ #
noisy_image = np.zeros(img.shape, np.float32)
if len(img.shape) == 2:
    noisy_image = img + gaussian
else:
    noisy_image[:, :, 0] = img[:, :, 0] + gaussian
    noisy_image[:, :, 1] = img[:, :, 1] + gaussian
    noisy_image[:, :, 2] = img[:, :, 2] + gaussian

cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)
# noisy_image = noisy_image.astype(np.uint8)

# ------------------ salt and pepper ------------------ #
saltRe = salt(noisy_image, 10000)
result = pepper(saltRe, 10000)
  
# cv2.imwrite('D:/Master_2019_2021/Muti_Party_Computation/ErrorCorrection/parties/party6.bmp',  result) 
np.save('D:/Master_2019_2021/Muti_Party_Computation/ErrorCorrection/parties/ImageProcessing/party2_real.npy',result)

 
# plt.figure(0)
# plt.imshow(img) # 顯示圖片
# plt.axis('off') # 不顯示座標軸
# plt.figure(1)
# plt.imshow(noisy_image) # 顯示圖片
# plt.axis('off') # 不顯示座標軸
# plt.figure(2)
# plt.imshow(saltRe) # 顯示圖片
# plt.axis('off') # 不顯示座標軸
plt.figure(3)
plt.imshow(saltRe)
plt.axis('off') # 不顯示座標軸 