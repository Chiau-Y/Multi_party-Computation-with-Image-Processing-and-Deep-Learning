import cv2
import time
import random
import numpy as np
from scipy import misc, ndimage
import matplotlib.pyplot as plt


start_time = time.time()
        
#------------------------------------ Distribution ----------------------------------# 
def Distribution(img):
        
    img_party1_result = np.zeros([img_size[0] , img_size[1] ])   
    img_party2_result = np.zeros([img_size[0] , img_size[1] ])   
    img_party3_result = np.zeros([img_size[0] , img_size[1] ])  
    img_party4_result = np.zeros([img_size[0] , img_size[1] ])   
    img_party5_result = np.zeros([img_size[0], img_size[1] ])    
    img_party6_result = np.zeros([img_size[0], img_size[1]])    
    img_party7_result = np.zeros([img_size[0], img_size[1]])   
    
    
    for i in range (img_size[0]):       
        for j in range(img_size[1]):      
            random_num = random.randint(0,100)
            img_party1_result[i][j] = (img[i][j] + random_num * 1)*16 % 33292801
            img_party2_result[i][j] = (img[i][j] + random_num * 2)*16 % 33292801
            img_party3_result[i][j] = (img[i][j] + random_num * 3)*16 % 33292801     
            img_party4_result[i][j] = (img[i][j] + random_num * 4)*16 % 33292801
            img_party5_result[i][j] = (img[i][j] + random_num * 5)*16 % 33292801
            img_party6_result[i][j] = (img[i][j] + random_num * 6)*16 % 33292801  
            img_party7_result[i][j] = (img[i][j] + random_num * 7)*16 % 33292801
            
                        
    return img_party1_result, img_party2_result, img_party3_result, img_party4_result, img_party5_result, img_party6_result, img_party7_result
 
#----------------------------------- Gaussian Filter ---------------------------------#
def Guassian(img_22): 
    
    GaussianKernel = np.array([[1,2,1],[2,4,2],[1,2,1]])
    img_result3 = ndimage.convolve(img_22, GaussianKernel, mode='nearest')#*31212001) % 33292801 # 31212001
    img_result3 = img_result3 * 31212001 % 33292801
        
    return img_result3

#---------------------------------------- Sobel --------------------------------------#
def Sobel(img_afterGaussian):   
    global x,y
    x = (cv2.Sobel(img_afterGaussian, cv2.CV_64F, 1, 0))%33292801
    y = (cv2.Sobel(img_afterGaussian, cv2.CV_64F, 0, 1))%33292801
    x = x.astype('int64')
    y = y.astype('int64')
    
    img_sobel =  (x**2 + y**2)%33292801
    
    return img_sobel 
    
#------------------------------------  Main ----------------------------------#
if __name__ == '__main__':
    print("----------- Local -----------")
    img_original = cv2.imread('D:/Master_2019_2021/Muti_Party_Computation/python/image/hello_3.bmp',0)  # GRAY    
    img_size = np.shape (img_original)
    
    
    plt.figure(0)
    plt.imshow(img_original, cmap='Greys_r') # ????????????
    plt.axis('off') # ??????????????????  
 
    
    print("Distributon")
    party1, party2, party3, party4, party5, party6, party7 = Distribution(img_original) 
 
    # img_size = np.shape (img_party1_)
    print("Gaussian")
    img1_G= Guassian(party1) 
    img2_G= Guassian(party2) 
    img3_G= Guassian(party3) 
    img4_G= Guassian(party4) 
    img5_G= Guassian(party5) 
    img6_G= Guassian(party6)   
    img7_G= Guassian(party7)
    
    print("Sobel")
    img1= Sobel(img1_G) 
    img2= Sobel(img2_G) 
    img3= Sobel(img3_G) 
    img4= Sobel(img4_G) 
    img5= Sobel(img5_G) 
    img6= Sobel(img6_G) 
    img7= Sobel(img7_G) 
    

    np.save('Party1', img1)
    np.save('Party2', img2)
    np.save('Party3', img3)
    np.save('Party4', img4)
    np.save('Party5', img5)
    np.save('Party6', img6)       
    np.save('Party7', img7)     

