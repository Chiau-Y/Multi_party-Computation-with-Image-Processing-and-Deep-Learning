from google.colab import drive
drive.mount('/content/drive')

import os
import cv2
import random
import numpy as np
from scipy import misc, ndimage

#----------------------------------- Gaussian Filter ---------------------------------#
def Guassian(img_22): 
    
    GaussianKernel = np.array([[1,2,1],[2,4,2],[1,2,1]])
    img_result3 = ndimage.convolve(img_22, GaussianKernel, mode='nearest')#*31212001) % 33292801 # 31212001
    img_result3 = img_result3 * 31212001 % 33292801
        
    return img_result3

#---------------------------------------- Sobel --------------------------------------#
def Sobel(img_afterGaussian):   
    
    x = (cv2.Sobel(img_afterGaussian, cv2.CV_64F, 1, 0))%33292801
    y = (cv2.Sobel(img_afterGaussian, cv2.CV_64F, 0, 1))%33292801
    x = x.astype('int64')
    y = y.astype('int64')
        
    img_sobel =  (x**2 + y**2)%33292801
    
    return img_sobel
#---------------------------------- Noise --------------------------------#
def Gaussian_noise(img_sss):
  mean = 0
  var = 100
  sigma = var ** 0.5
  gaussian = np.random.normal(mean, sigma, (img.shape[0], img.shape[1])) #  np.zeros((224, 224), np.float32)
  
  noisy_image = np.zeros(img_sss.shape, np.float32)
  noisy_image = img_sss + 50*np.trunc(gaussian)

  return noisy_image

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
# ---------------------------------------- Main ---------------------------------------- #
while(True):

  filepath = '/content/drive/MyDrive/Party7.npy'
  filepath_complete = '/content/drive/MyDrive/Party7_complete.npy'

  # 檢查原檔是否存在
  while(not os.path.isfile(filepath)):
    pass

  if (os.path.isfile(filepath_complete)):
    !rm /content/drive/MyDrive/Party1_complete.npy
    !rm /content/drive/MyDrive/Party2_complete.npy
    !rm /content/drive/MyDrive/Party3_complete.npy
    !rm /content/drive/MyDrive/Party4_complete.npy
    !rm /content/drive/MyDrive/Party5_complete.npy
    !rm /content/drive/MyDrive/Party6_complete.npy
    !rm /content/drive/MyDrive/Party7_complete.npy

  # 檢查輸出檔是否刪除
  while(os.path.isfile(filepath_complete)):
    pass 

  error = random.sample(range(1,7),2)
  error.sort()
  print("\nThe error party is : ", error)
  for i in range(7):
    
    img_party = np.load('/content/drive/MyDrive/Party'+str(i+1)+'.npy')

    img_size = np.shape(img_party)
    # print("Gaussian")
    img_G = Guassian(img_party)

    # print("Sobel")
    img = Sobel(img_G) 

    if (i+1) in error:
      noise_G = Gaussian_noise(img)
      saltRe = salt(noise_G, 2000)
      img = pepper(saltRe, 2000).astype('int64')%33292801
    
    np.save('/content/drive/MyDrive/Party'+str(i+1)+'_complete.npy', img)
    print("=== Party{} Done ===\n".format(i+1))

 
  !rm /content/drive/MyDrive/Party1.npy
  !rm /content/drive/MyDrive/Party2.npy
  !rm /content/drive/MyDrive/Party3.npy
  !rm /content/drive/MyDrive/Party4.npy
  !rm /content/drive/MyDrive/Party5.npy
  !rm /content/drive/MyDrive/Party6.npy
  !rm /content/drive/MyDrive/Party7.npy

  # 檢查原檔是否刪除
  while(os.path.isfile(filepath)):
    pass 
  print("------- The End --------")

