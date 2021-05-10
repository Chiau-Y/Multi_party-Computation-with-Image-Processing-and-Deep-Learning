import cv2
import time
import random
import os.path
import numpy as np
from datetime import datetime
from collections import Counter
import matplotlib.pyplot as plt
# from scipy import misc, ndimage
import Upload_File, Download_File
from tensorflow.keras import datasets, layers, models

start_time = time.time()
#------------------------------------ Distribution ----------------------------------# 
def Distribution(img):
        
    img_party_result = np.zeros((n, img.shape[0], img.shape[1], 3))
    
    img_R = img[:,:,0]
    img_G = img[:,:,1]
    img_B = img[:,:,2]   
    
    for i in range (img.shape[0]):       
        for j in range(img.shape[1]):      
            random_num = random.uniform(-50,50)
            for k in range  (int(-n/2), int((n/2)+1)):
                if (k != 0 and k < 0):
                    img_party_result[k+int(n/2)][i][j][0] = (img_R[i][j] + random_num * (k))/255.0    
                    img_party_result[k+int(n/2)][i][j][1] = (img_G[i][j] + random_num * (k))/255.0  
                    img_party_result[k+int(n/2)][i][j][2] = (img_B[i][j] + random_num * (k))/255.0  
                if (k != 0 and k > 0):
                    img_party_result[k+int(n/2)-1][i][j][0] = (img_R[i][j] + random_num * (k))/255.0   
                    img_party_result[k+int(n/2)-1][i][j][1] = (img_G[i][j] + random_num * (k))/255.0  
                    img_party_result[k+int(n/2)-1][i][j][2] = (img_B[i][j] + random_num * (k))/255.0                   
                    
    return img_party_result

#------------------------------------ Reconstruction ----------------------------------# 
def Reconstruction(share,c):

    total = 0
    for i in range (int(n/2)):
        total += share[i+int(n/2)]*c[i]
        total += share[i]*c[int(n/2)-1-i]

    return total

#------------------------------------ Softmax ----------------------------------#
def softmax(x):
    """Compute the softmax of vector x."""
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    
    return softmax_x  

#------------------------------------ Lagrange Interpolation ----------------------------------# 
def LagrangeInterpolation(n):
    ans_coefficient = []
    for i in range(1,n+1):
        temp = 1
        ans_temp = 1
        for j in range(1,n+1):
            if (-i != -j):
                temp = -(-j)/(-(i-j))
                ans_temp *= temp
        ans_coefficient.append(ans_temp)
        
    return ans_coefficient

#------------------------------------  Main ----------------------------------#
if __name__ == '__main__':
    error_party_check = {}
    Flag_Pseudo = False
    t, n, num = 1, 6, 100

    print("=========== Local ===========\n")
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    
    #pretreatment
    x_test_normalize = x_test.astype('float32')/255.0 
    
    testing_num = random.randint(0,num)
    
    img_size = np.shape (x_test[testing_num])
    
    plt.figure(0)
    plt.imshow(x_test[testing_num]) # 顯示圖片
    plt.axis('off') # 不顯示座標軸  
    
    print("----------- Distributon -----------\n")
    img_party = Distribution(x_test[testing_num]) 
    # # print("----------- Save .npy -----------")
    for k in range(n):
        np.save('Party'+str(k+1)+'_dp', img_party[k])

    print("Upload the file to Cloud")
    time_now_utc = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")

    for num in range (n):   
        print("")
        print("Upload the file, Party{}...".format(num+1),end='\n')
        Upload_File.main(is_update_file_function=bool(True),
                          update_drive_service_folder_name = None,
                          update_drive_service_name='Party'+str(num+1)+'_dp.npy', 
                          update_file_path=os.getcwd() + './',
                          cred_file='credentials_peng01.json',
                          token_file='token_peng01.pickle') 
        
    print("\n=========== Cloud Computing ===========",end='\n')   
    while(not Download_File.main(is_download_file_function=bool(False), 
                                  drive_service_folder_name=None, 
                                  download_drive_service_name='Party'+str(n)+'_dp_Complete.npy', 
                                  download_file_path=os.getcwd() + '/download/',
                                  cred_download_file='credentials_peng01.json',
                                  token_download_file='token_download_peng01.pickle',
                                  time_now=time_now_utc)):
        pass
    
    print("\n=========== Local ===========",end='\n\n') 
    print("Download the file to Cloud") 
    for num in range (n):   
        print("")
        print("Download the file, Party{}...".format(num+1))
        Download_File.main(is_download_file_function=bool(True), 
                            drive_service_folder_name=None, 
                            download_drive_service_name='Party'+str(num+1)+'_dp_Complete.npy', 
                            download_file_path=os.getcwd() + '/download/',
                            cred_download_file='credentials_peng01.json',
                            token_download_file='token_download_peng01.pickle',
                            time_now=time_now_utc)
    

    # print("Load .npy \n\n")
    probabilities_party = []
    for k in range(n):
        probabilities_party.append(np.load('./download/Party'+str(k+1)+'_dp_Complete.npy'))   
    
    print("\n----------- Reconstruction -----------\n")
    largrange =  LagrangeInterpolation(int(n/2))
    
    prob_rec = Reconstruction(np.array(probabilities_party),largrange)
    
    prob_rec_softmax = softmax(prob_rec)
    
    predicted_label_rec = np.argmax(prob_rec_softmax) # 找最大的機率
    
    # show image
    class_name = ['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck'] 
          
    print("The Prediction with MPC :  {:.0f}%  for  {}".format(100*np.max(prob_rec_softmax),
                                                               class_name[predicted_label_rec]))    
    
    probabilities_array = prob_rec_softmax.reshape(10) 
    plt.figure(24,figsize=(20,8))
    plt.grid(False)
    plt.xticks(range(10), class_name, fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim([0, 1])
    barlist = plt.bar(range(10), probabilities_array)
    barlist[predicted_label_rec].set_color('b')
    barlist[y_test[testing_num,0]].set_color('r')
    plt.show()
    
end_time = time.time()     
print("")        
print("總共花費時間 : ",round(end_time-start_time, 2),"sec")       
          

