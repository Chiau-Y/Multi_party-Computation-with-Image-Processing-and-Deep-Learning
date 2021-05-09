import matplotlib.pyplot as plt # plt 用於顯示圖片
import matplotlib.image as mpimg # mpimg 用於讀取圖片
import numpy as np

#----------------------------------- Gaussian Filter ---------------------------------#
img_result4 = []
img = mpimg.imread('D:/CHIAUYAUN/Muti_Party_Computation/python/4.jpg')   
img_size = np.shape (img)
row_0 = np.zeros([1,img_size[1]])                                              
column_0 = np.zeros([1,img_size[0]+2])
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
            img_result3[i][j] = int(temp/16)  
    img_result4.append(np.array(img_result3))
    
#------------------------------------- RGB to Gray -----------------------------------#
img_Gray = np.zeros([img_size[0], img_size[1]], dtype='uint8')
for i in range (img_size[0]):
    for j in range (img_size[1]):
        img_Gray[i][j] = int(((img_result4[0][i][j])*299 + (img_result4[1][i][j])*587 + (img_result4[2][i][j])*114 + 500) / 1000)

#---------------------------------------- Sobel --------------------------------------#    
temp_array = np.insert(img_Gray, 0, values=row_0 , axis=0)
temp_array = np.insert(temp_array, img_size[0]+1, values=row_0 , axis=0)
temp_array = np.insert(temp_array, 0, values=column_0 , axis=1)
img_Gray3 = np.insert(temp_array, img_size[1]+1, values=column_0 , axis=1)

Sobel_filter = [[[-1,0,1],[-2,0,2],[-1,0,1]],[[-1,-2,-1],[0,0,0],[1,2,1]]]               
sobel_result = np.zeros([img_size[0], img_size[1]]) 
sobel_result3 = np.zeros([img_size[0], img_size[1]])
img_sobel = np.zeros([img_size[0], img_size[1]], dtype='uint8')

sobel_result4 = [] 
for direction in range (2): 
    sobel_result2 = []
    for p in range(3):
        for q in range(3): 
            for i in range (img_size[0]):
                for j in range(img_size[1]):
                    sobel_result[i][j] = img_Gray3[p+i][q+j]*Sobel_filter[direction][p][q]   
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
                
#------------------------------------ Binarization ----------------------------------# 
Binarization_result = np.zeros([img_size[0], img_size[1]], dtype='uint8')
for i in range (img_size[0]):
    for j in range(img_size[1]):
        if img_sobel[i][j] > 127 :
            Binarization_result[i][j] = 255
        else:
            Binarization_result[i][j] = 0
            
            
plt.figure(0)
plt.imshow(img) # 顯示圖片
plt.axis('off') # 不顯示座標軸
plt.figure(1)
plt.imshow(img_Gray, cmap='Greys_r') # 顯示圖片
plt.axis('off') # 不顯示座標軸
plt.figure(2)
plt.imshow(sobel_result4[0], cmap='Greys_r')
plt.axis('off') # 不顯示座標軸
plt.figure(3)
plt.imshow(sobel_result4[1], cmap='Greys_r')
plt.axis('off') # 不顯示座標軸
plt.figure(4)
plt.imshow(Binarization_result, cmap='Greys_r')
plt.axis('off') # 不顯示座標軸
plt.show()
