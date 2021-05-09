import matplotlib.pyplot as plt # plt 用於顯示圖片
import matplotlib.image as mpimg # mpimg 用於讀取圖片
import numpy as np
import cv2

import psutil
import functools
from threading import Timer

import time
from datetime import datetime

# ------------------------------------Subroutine------------------------------------
def hello():
    global cpu , time_axis   
    cpu.append(psutil.cpu_percent(interval=0.4))
    time_axis.append(datetime.utcnow().strftime("%M:%S"))
    
class RepeatingTimer(Timer): 
    def run(self):
        while not self.finished.is_set():
            self.function(*self.args, **self.kwargs)
            self.finished.wait(self.interval)

#----------------------------------- Gaussian Filter ---------------------------------#
def Guassian(img_22): 
    
    img_size = np.shape (img)
    row_0 = np.zeros([1,img_size[1]])                                              
    column_0 = np.zeros([1,img_size[0]+2])
        
    Gaussian_filter = np.array([[1,2,1],[2,4,2],[1,2,1]])
    img_result = np.zeros([img_size[0], img_size[1]])    
    img_result3 = np.zeros([img_size[0], img_size[1]])
       
    img_result2 = []
    for p in range(3):
        for q in range(3): 
            for i in range (img_size[0]-2):
                for j in range(img_size[1]-2):
                    img_result[i][j] = img[p+i][q+j]*Gaussian_filter[p][q]   
            img_result2.append(np.array(img_result))
    
    for i in range (img_size[0]):
        for j in range(img_size[1]):
            temp = 0
            for k in range (9):
                temp += img_result2[k][i][j]
            img_result3[i][j] = int(temp/16)  
            
    return np.array(img_result3)

#---------------------------------------- Sobel --------------------------------------#    
def Sobel(img_afterGaussian):
    # temp_array = np.insert(img_Gray, 0, values=row_0 , axis=0)
    # temp_array = np.insert(temp_array, img_size[0]+1, values=row_0 , axis=0)
    # temp_array = np.insert(temp_array, 0, values=column_0 , axis=1)
    # img_Gray3 = np.insert(temp_array, img_size[1]+1, values=column_0 , axis=1)
    
    Sobel_filter = [[[-1,0,1],[-2,0,2],[-1,0,1]],[[-1,-2,-1],[0,0,0],[1,2,1]]]               
    sobel_result = np.zeros([img_size[0], img_size[1]]) 
    sobel_result3 = np.zeros([img_size[0], img_size[1]])
    img_sobel = np.zeros([img_size[0], img_size[1]], dtype='uint8')
    
    sobel_result4 = [] 
    for direction in range (2): 
        sobel_result2 = []
        for p in range(3):
            for q in range(3): 
                for i in range (img_size[0]-2):
                    for j in range(img_size[1]-2):
                        sobel_result[i][j] = img_afterGaussian[p+i][q+j]*Sobel_filter[direction][p][q]   
                sobel_result2.append(np.array(sobel_result))
                
        for i in range (img_size[0]):
            for j in range(img_size[1]):
                temp = 0
                for k in range (9):
                    temp += sobel_result2[k][i][j]
                sobel_result3[i][j] = int(temp) 
        sobel_result4.append(np.array(sobel_result3))   
    
    for i in range (img_size[0]):
        for j in range(img_size[1]):   
            img_sobel[i][j] = (sobel_result4[0][i][j]**2 + sobel_result4[1][i][j]**2)**(1/2)
            
    return img_sobel
                
        
#------------------------------------  Main ----------------------------------#
if __name__ == '__main__':
    cpu = []
    time_axis = []
    img = cv2.imread('D:/Master_2019_2021/Muti_Party_Computation/python/image/Lenna_.bmp',0)  # GRAY 
    img_size = np.shape (img)
    
    t_cpu = RepeatingTimer(1.0, hello)
    t_cpu.start()
    time.sleep(3)     
    
    print("Gaussian")
    gs = datetime.utcnow().strftime("%M:%S") 
    img_G = Guassian(img) 
    ge = datetime.utcnow().strftime("%M:%S") 
    
    print("Sobel")
    ss = datetime.utcnow().strftime("%M:%S") 
    img_S = Sobel(img_G)
    se = datetime.utcnow().strftime("%M:%S") 
            

    time.sleep(3)
    t_cpu.cancel() 
    
    plt.figure(20)
    plt.plot(time_axis,cpu, label='CPU')
    plt.ylabel('CPU%')
    plt.ylim(0,100)
    plt.xticks(rotation=45)
    plt.show()     