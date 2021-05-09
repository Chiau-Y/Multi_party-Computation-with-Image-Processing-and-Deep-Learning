from __future__ import print_function
import os
import io
import time
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from httplib2 import Http
from oauth2client import file, client, tools

import cv2
import time
import random
import numpy as np
from scipy import misc, ndimage
import matplotlib.pyplot as plt

start_time = time.time()

#------------------------------------ Distribution ----------------------------------# 
def Distribution(img):
        
    img_party1_result = np.zeros([img_size[0] , img_size[1]])
    img_party2_result = np.zeros([img_size[0] , img_size[1]])
    img_party3_result = np.zeros([img_size[0] , img_size[1]])
    img_party4_result = np.zeros([img_size[0] , img_size[1]])
    
    for i in range (img_size[0]):       
        for j in range(img_size[1]):
            random_num = random.randint(0,100000)
            img_party1_result[i][j] = (img[i][j] + random_num * 1)*16 % 33292801
            img_party2_result[i][j] = (img[i][j] + random_num * 2)*16 % 33292801
            img_party3_result[i][j] = (img[i][j] + random_num * 3)*16 % 33292801
            
            
    return img_party1_result, img_party2_result, img_party3_result

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
    
    img_sobel =  (x**2 + y**2)%33292801
    
    return img_sobel

#------------------------------------ Reconstruction ----------------------------------# 
def Construction(share1, share2, share3):

    img_result12 = np.zeros([img_size[0], img_size[1]], dtype='uint32') 
    img_result23 = np.zeros([img_size[0], img_size[1]], dtype='uint32') 
    img_result13 = np.zeros([img_size[0], img_size[1]], dtype='uint32') 
    img_result123 = np.zeros([img_size[0], img_size[1]], dtype='uint32') 
    
    # party 1, 2
    for i in range (img_size[0]):
        for j in range(img_size[1]):   
            img_result12[i][j] = (share1[i][j]*2 - share2[i][j]*1)%33292801

    # party 2, 3
    for i in range (img_size[0]):
        for j in range(img_size[1]):   
            img_result23[i][j] = (share2[i][j]*3 - share3[i][j]*2)%33292801

    # party 1, 3
    for i in range (img_size[0]):
        for j in range(img_size[1]):
            img_result13[i][j] = (share1[i][j]*3*16646401 - share3[i][j]*1*16646401)%33292801 
            
    # party 1, 2, 3
    for i in range (img_size[0]):
        for j in range(img_size[1]):   
            img_result123[i][j] = (share1[i][j]*3 - share2[i][j]*3 + share3[i][j]*1)%33292801

    return img_result12, img_result23, img_result13, img_result123
        
#------------------------------------  Main ----------------------------------#
if __name__ == '__main__':
    print("----------- Local -----------")
    img_original = cv2.imread('D:/Master_2019_2021/Muti_Party_Computation/python/image/Lenna_.bmp',0)  # GRAY    
    img_size = np.shape (img_original)
    
    plt.figure(0)
    plt.imshow(img_original, cmap='Greys_r') # 顯示圖片
    plt.axis('off') # 不顯示座標軸  
 
    
    print("Distributon")
    img_party1, img_party2, img_party3 = Distribution(img_original) 
    print("Save .npy")
    np.save('Party1', img_party1)
    np.save('Party2', img_party2)
    np.save('Party3', img_party3)

    
    print("----------- Cloud -----------")
    print("Load .npy")
    img_party1_ = np.load('Party1.npy')
    img_party2_ = np.load('Party2.npy')
    img_party3_ = np.load('Party3.npy')
        
    
    img_size = np.shape (img_party1_)
    print("Gaussian")
    img1_G= Guassian(img_party1_) 
    img2_G= Guassian(img_party2_) 
    img3_G= Guassian(img_party3_) 
    
    
    # print("Reconstruction - Gaussian")
    # Image12, Image23, Image13, Image = Construction(img1_G,img2_G,img3_G)
    
    # plt.figure(8)
    # plt.imshow(Image12/16, cmap='Greys_r') # 顯示圖片
    # plt.axis('off') # 不顯示座標軸
    # plt.figure(9)
    # plt.imshow(Image23/16, cmap='Greys_r') # 顯示圖片
    # plt.axis('off') # 不顯示座標軸
    # plt.figure(10)
    # plt.imshow(Image13/16, cmap='Greys_r')
    # plt.axis('off') # 不顯示座標軸
    # plt.figure(11)   
    # plt.imshow(Image/16, cmap='Greys_r')
    # plt.axis('off') # 不顯示座標軸    

    print("Sobel")
    img1= Sobel(img1_G) 
    img2= Sobel(img2_G) 
    img3= Sobel(img3_G) 
    
    print("Save .npy")
    np.save('Party1_S', img1)
    np.save('Party2_S', img2)
    np.save('Party3_S', img3)

    
    print("----------- Local -----------")
    print("Load .npy")
    img_party1_S = np.load('Party1_S.npy')
    img_party2_S = np.load('Party2_S.npy')
    img_party3_S = np.load('Party3_S.npy')       
    
    print("Reconstruction")
    img_size = np.shape (img_party1_S)
    Image12, Image23, Image13, Image = Construction(img_party1_S,img_party2_S,img_party3_S)
          
    plt.figure(4)
    plt.imshow((Image12**(1/2)/16), cmap='Greys_r') # 顯示圖片
    plt.axis('off') # 不顯示座標軸
    plt.figure(5)
    plt.imshow((Image23**(1/2)/16), cmap='Greys_r') # 顯示圖片
    plt.axis('off') # 不顯示座標軸
    plt.figure(6)
    plt.imshow((Image13**(1/2)/16), cmap='Greys_r')
    plt.axis('off') # 不顯示座標軸
    plt.figure(7)   
    plt.imshow((Image**(1/2)/16), cmap='Greys_r')
    plt.axis('off') # 不顯示座標軸 
 

end_time = time.time()     
print("")        
print("Time : ",round(end_time-start_time, 2),"sec")            
            
# plt.figure(8)
# plt.imshow(np.uint8(img_party3), cmap='Greys_r') # 顯示圖片
# plt.axis('off') # 不顯示座標軸
