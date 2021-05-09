import matplotlib.pyplot as plt # plt 用於顯示圖片
import matplotlib.image as mpimg # mpimg 用於讀取圖片
import numpy as np
import time
import cv2

start_time = time.time()

#----------------------------------- Gaussian Filter ---------------------------------#
def Guassian():
    global img_result4
    
    img_result4 = []
    for channel in range (3):   
        temp_array = np.insert(img[:,:,channel], 0, values=row_0 , axis=0)
        temp_array = np.insert(temp_array, img_size[0]+1, values=row_0 , axis=0)
        temp_array = np.insert(temp_array, 0, values=column_0 , axis=1)
        img_Gray2 = np.insert(temp_array, img_size[1]+1, values=column_0 , axis=1)
        
        Gaussian_filter = np.array([[1,2,1],[2,4,2],[1,2,1]])
        img_result = np.zeros([img_size[0], img_size[1]])    
        img_result3 = np.zeros([img_size[0], img_size[1]])
           
        img_result2 = []
        for p in range(3):
            for q in range(3): 
                for i in range (img_size[0]):
                    for j in range(img_size[1]):
                        img_result[i][j] = img_Gray2[p+i][q+j]*Gaussian_filter[p][q]   
                img_result2.append(np.array(img_result))
        
        for i in range (img_size[0]):
            for j in range(img_size[1]):
                temp = 0
                for k in range (9):
                    temp += img_result2[k][i][j]
                temp2 = temp %251
                for s in range (1000):
                    if (251*(s+1) + temp2)%16 == 0:
                        break
                img_result3[i][j] = int((251*(s+1) + temp2)/16) 
        img_result4.append(np.array(img_result3))

#------------------------------------- RGB to Gray -----------------------------------#
def RGB2Gray():
    global img_Gray
    
    img_Gray = np.zeros([img_size[0], img_size[1]], dtype='uint8')
    for i in range (img_size[0]):
        for j in range (img_size[1]):
            img_Gray[i][j] = int(((img_result4[0][i][j])*299 + (img_result4[1][i][j])*587 + (img_result4[2][i][j])*114 + 500) / 1000)

#---------------------------------------- Sobel --------------------------------------#
def Sobel():   
    global img_sobel
    
    img_Gray_3d = cv2.cvtColor(img_Gray, cv2.COLOR_GRAY2BGR)
    x = cv2.Sobel(img_Gray_3d, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img_Gray_3d, cv2.CV_16S, 0, 1)
    
    absX = cv2.convertScaleAbs(x)# 轉回uint8
    absY = cv2.convertScaleAbs(y)
    
    img_sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

#------------------------------------ Binarization ----------------------------------# 
def Binarization(): 
    global Binarization_result
    
    ret,Binarization_result = cv2.threshold(img_sobel,64,255,cv2.THRESH_BINARY)
            
#------------------------------------  Closing ----------------------------------# 
def Closing():
    global closing_result 
    kernel = np.ones((3,3), np.uint8)
    dilation = cv2.dilate(Binarization_result, kernel, iterations = 1)
    closing_result = cv2.erode(dilation, kernel, iterations = 1)
    # closing_result = cv2.erode(closing_result, kernel, iterations = 1)
    # closing_result = cv2.erode(closing_result, kernel, iterations = 1)
    # closing_result = cv2.erode(closing_result, kernel, iterations = 1)
    
#------------------------------------  Main ----------------------------------#     
img = mpimg.imread('D:/CHIAUYAUN/Muti_Party_Computation/python/image/test01.jpg')   
img_size = np.shape (img)
row_0 = np.zeros([1,img_size[1]])                                              
column_0 = np.zeros([1,img_size[0]+2])  
          
Guassian()  
RGB2Gray()  
Sobel()
Binarization() 
Closing()
             
end_time = time.time()            
print("Time : ",round(end_time-start_time, 2),"sec")            
            
plt.figure(0)
plt.imshow(img) # 顯示圖片
plt.axis('off') # 不顯示座標軸
plt.figure(1)
plt.imshow(img_Gray, cmap='Greys_r') # 顯示圖片
plt.axis('off') # 不顯示座標軸
plt.figure(2)
plt.imshow(img_sobel, cmap='Greys_r')
plt.axis('off') # 不顯示座標軸
plt.figure(3)
plt.imshow(Binarization_result, cmap='Greys_r')
plt.axis('off') # 不顯示座標軸
plt.figure(4)
plt.imshow(closing_result , cmap='Greys_r')
plt.axis('off') # 不顯示座標軸
plt.show()
