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
import math
import random
import numpy as np
from random import sample
import matplotlib.pyplot as plt


start_time = time.time()
        
#------------------------------------ Distribution ----------------------------------# 
def Distribution(img):
        
    img_party1_result = np.zeros([img_size[0] + 2, img_size[1] + 2])    # party 7, let the secret be more secure
    img_party2_result = np.zeros([img_size[0] + 2, img_size[1] + 2])    # party 8
    img_party3_result = np.zeros([img_size[0] + 2, img_size[1] + 2])    # party 9
    img_party4_result = np.zeros([img_size[0] + 2, img_size[1] + 2])    # party 9
    img_party5_result = np.zeros([img_size[0] + 2, img_size[1] + 2])    # party 9
    img_party6_result = np.zeros([img_size[0] + 2, img_size[1] + 2])    # party 9
    
    img_2 = cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_REPLICATE)        # padding
    
    for i in range (img_size[0] + 2):       
        for j in range(img_size[1] + 2):      
            while(img_party1_result[i][j]==0):
                if (img_2[i][j]>=127):                    
                    random_num = random.randint(-25,15)
                else:
                    random_num = random.randint(0,25)
                share1 = (img_2[i][j] + random_num * 1)   # shares
                share2 = (img_2[i][j] + random_num * 2)   # shares
                share3 = (img_2[i][j] + random_num * 3)   # shares    
                share4 = (img_2[i][j] + random_num * 4)   # shares  
                share5 = (img_2[i][j] + random_num * 5)   # shares  
                share6 = (img_2[i][j] + random_num * 6)   # shares                    
                if (share6 <= 540 and share6 > -1):               
                    img_party1_result[i][j] = share1   # shares
                    img_party2_result[i][j] = share2   # shares
                    img_party3_result[i][j] = share3   # shares    
                    img_party4_result[i][j] = share4   # shares
                    img_party5_result[i][j] = share5   # shares
                    img_party6_result[i][j] = share6   # shares   
                        
    return img_party1_result, img_party2_result, img_party3_result, img_party4_result, img_party5_result, img_party6_result

#------------------------------------ Array to Image ----------------------------------#
def Array2Image(party):
    
    layer1 = np.zeros([img_size[0], img_size[1]],dtype='float32')
    layer2 = np.zeros([img_size[0], img_size[1]],dtype='float32')
    layer3 = np.zeros([img_size[0], img_size[1]],dtype='float32')    
    
    for i in range (img_size[0]):       
        for j in range(img_size[1]): 
            layer_list = [ layer1, layer2, layer3 ]
            random_list = sample(layer_list, 3) 
            random_list[0][i][j] = round(max(min(party[i][j],255),0))
            random_list[1][i][j] = round(max(min(party[i][j] - 255,255),0))
            random_list[2][i][j] = round(max(min(party[i][j] - 255*2,255),0))
            
    img_RGB = np.zeros([img_size[0], img_size[1], 3], dtype=np.uint8)
    img_RGB[:,:,0] = layer1
    img_RGB[:,:,1] = layer2
    img_RGB[:,:,2] = layer3
    
    return img_RGB    

#------------------------------------ Image to Array ----------------------------------#
def Image2Array(img_party):  
    
    img_gray = np.zeros([img_size[0], img_size[1]], dtype='float32')
    
    for i in range (img_size[0]):       
        for j in range(img_size[1]): 
            img_gray[i][j] = int(img_party[:,:,0][i][j]) + int(img_party[:,:,1][i][j]) + int(img_party[:,:,2][i][j])
    
    return img_gray 

#----------------------------------- Gaussian Filter ---------------------------------#
def Guassian(img_22):
    global gaussianBlur,gaussianBlurKernel
    
    gaussianBlur = (cv2.filter2D(src=img_22, kernel=gaussianBlurKernel, ddepth= cv2.CV_64F))    # 32 bits for float, cv2.CV_32F   
            
    return gaussianBlur

#---------------------------------------- Sobel --------------------------------------#
def Sobel(img_afterGaussian):  
    global x,y,img_sobel,img_sobel_temp
    
    x = (cv2.Sobel(img_afterGaussian, cv2.CV_64F, 1, 0))    # 64 bits for float
    y = (cv2.Sobel(img_afterGaussian, cv2.CV_64F, 0, 1))
    
    img_sobel_temp =  (x**2 +y**2)
    img_sobel =  img_sobel_temp**(1/2)
 
    return img_sobel

#------------------------------------ Reconstruction ----------------------------------# 
def Construction(share1, share2, share3):

    img_result12 = np.zeros([img_size[0], img_size[1]], dtype='uint8') 
    img_result23 = np.zeros([img_size[0], img_size[1]], dtype='uint8') 
    img_result13 = np.zeros([img_size[0], img_size[1]], dtype='uint8') 
    img_result123 = np.zeros([img_size[0], img_size[1]], dtype='uint8') 
       
    # for i in range (img_size[0]):
    #     for j in range(img_size[1]):    
    #         img_result12[i][j] = max(min((share1[i][j]*2 - share2[i][j]*1),255),0)                          
    #         img_result23[i][j] = max(min((share2[i][j]*3 - share3[i][j]*2),255),0)                         
    #         img_result13[i][j] = max(min((share1[i][j]*3/2 - share3[i][j]*1/2),255),0)                      
    #         img_result123[i][j] = max(min((share1[i][j]*3 - share2[i][j]*3 + share3[i][j]*1),255),0)        


    # return img_result12, img_result23, img_result13, img_result123
    for i in range (img_size[0]):
        for j in range(img_size[1]):    
            img_result12[i][j] = max(min((share1[i][j]*8 - share2[i][j]*7),255),0)                          # party 7, 8, [8,-7]
            img_result23[i][j] = max(min((share2[i][j]*9 - share3[i][j]*8),255),0)                          # party 8, 9, [9,-8]
            img_result13[i][j] = max(min((share1[i][j]*9/2 - share3[i][j]*7/2),255),0)                       # party 7, 9, [9/2,-7/2]
            img_result123[i][j] = max(min((share1[i][j]*36 - share2[i][j]*63 + share3[i][j]*28),255),0)      # party 7, 8, 9, [36,-63,28]    


    return img_result12, img_result23, img_result13, img_result123    
    
#------------------------------------  Main ----------------------------------#
if __name__ == '__main__':
    print("----------- Local -----------")
    img_original = cv2.imread('D:/Master_2019_2021/Muti_Party_Computation/python/image/Lenna_.bmp',0)  # GRAY    
    img_size = np.shape (img_original)
    
    
    plt.figure(0)
    plt.imshow(img_original, cmap='Greys_r') # ????????????
    plt.axis('off') # ??????????????????  
 
    
    print("Distributon")
    party1, party2, party3, party4, party5, party6 = Distribution(img_original) 
  
    
    print("GaussianBlur") 
    gaussianBlurKernel = np.array(([[1,2,1],[2,4,2],[1,2,1]]), np.float64)/16
    img1_G= Guassian(np.float64(party1)) 
    img2_G= Guassian(np.float64(party2)) 
    img3_G= Guassian(np.float64(party3)) 
    img4_G= Guassian(np.float64(party4)) 
    img5_G= Guassian(np.float64(party5)) 
    img6_G= Guassian(np.float64(party6)) 
    
    print("Sobel") 
    img1_s = Sobel(img1_G) 
    img2_s = Sobel(img2_G) 
    img3_s = Sobel(img3_G)        
    img4_s = Sobel(img4_G) 
    img5_s = Sobel(img5_G) 
    img6_s = Sobel(img6_G)   
    
    
    np.save('party1_real', img1_s)
    np.save('party2_real', img2_s)
    np.save('party3_real', img3_s)
    np.save('party4_real', img4_s)
    np.save('party5_real', img5_s)
    np.save('party6_real', img6_s)
            

