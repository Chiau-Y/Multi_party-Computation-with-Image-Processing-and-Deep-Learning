import cv2
import time
import random
import os.path
import numpy as np
from datetime import datetime
from fractions import Fraction
from collections import Counter
import matplotlib.pyplot as plt
from scipy import misc, ndimage
import Upload_File, Download_File, ErrorCorrection

start_time = time.time()
#------------------------------------ Distribution ----------------------------------# 
def Distribution(img):
        
    img_party_result = np.zeros((n,img.shape[0], img.shape[1]))
    
    for i in range (img_size[0]):       
        for j in range(img_size[1]):
            random_num = random.randint(0,100)
            img_party_result[0][i][j] = (img[i][j] + random_num * 1)*16 % prime_num
            img_party_result[1][i][j] = (img[i][j] + random_num * 2)*16 % prime_num
            img_party_result[2][i][j] = (img[i][j] + random_num * 3)*16 % prime_num
            img_party_result[3][i][j] = (img[i][j] + random_num * 4)*16 % prime_num
            img_party_result[4][i][j] = (img[i][j] + random_num * 5)*16 % prime_num
            img_party_result[5][i][j] = (img[i][j] + random_num * 6)*16 % prime_num
            img_party_result[6][i][j] = (img[i][j] + random_num * 7)*16 % prime_num
                      
    return img_party_result

#------------------------------------ Reconstruction ----------------------------------# 
def Construction(share,c):

    img_result = np.zeros([img_size[0], img_size[1]], dtype='uint32') 
    
    for i in range (img_size[0]):
        for j in range(img_size[1]):   
            img_result[i][j] = (share[0][i][j]*c[0] + share[1][i][j]*c[1] + share[2][i][j]*c[2])%prime_num

    return img_result

#------------------------------------ Lagrange Interpolation ----------------------------------# 
def LagrangeInterpolation(party):
    ans_coefficient = []
    for i in party:
        temp = 1
        ans_temp = 1
        for j in party:
            if (i != j):
                temp = -j/(i-j)
                ans_temp *= temp
        if ans_temp%1 != 0 :
            num_float = Fraction(str(ans_temp))
            ans_temp = ((num_float.numerator%prime_num)*(ErrorCorrection.Modinv(num_float.denominator,prime_num)))%prime_num
        ans_coefficient.append(ans_temp)
        
    return ans_coefficient

#------------------------------------  Main ----------------------------------#
if __name__ == '__main__':
    error_party_check = {}
    Flag_Pseudo = False
    t, n, prime_num = 2, 7, 33292801 

    print("=========== Local ===========\n")
    img_original = cv2.imread('D:/Master_2019_2021/Muti_Party_Computation/python/image/cameraman_2.bmp',0)  # GRAY    
    img_size = np.shape (img_original)
    
    plt.figure(0)
    plt.imshow(img_original, cmap='Greys_r') # 顯示圖片
    plt.axis('off') # 不顯示座標軸  
    
    print("----------- Distributon -----------\n")
    img_party = Distribution(img_original) 
    # print("----------- Save .npy -----------")
    np.save('Party1', img_party[0])
    np.save('Party2', img_party[1])
    np.save('Party3', img_party[2]) 
    np.save('Party4', img_party[3]) 
    np.save('Party5', img_party[4]) 
    np.save('Party6', img_party[5]) 
    np.save('Party7', img_party[6]) 
    
    print("Upload the file to Cloud")
    time_now_utc = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")

    for num in range (7):   
        print("")
        print("Upload the file, Party{} ...".format(num+1),end='\n')
        Upload_File.main(is_update_file_function=bool(True),
                          update_drive_service_folder_name = None,
                          update_drive_service_name='Party'+str(num+1)+'.npy', 
                          update_file_path=os.getcwd() + './',
                          cred_file='credentials_peng01.json',
                          token_file='token_peng01.pickle') 
        
    print("\n=========== Cloud Computing ===========",end='\n')   
    while(not Download_File.main(is_download_file_function=bool(False), 
                                  drive_service_folder_name=None, 
                                  download_drive_service_name='Party7_Complete.npy', 
                                  download_file_path=os.getcwd() + '/download/',
                                  cred_download_file='credentials_peng01.json',
                                  token_download_file='token_download_peng01.pickle',
                                  time_now=time_now_utc)):
        pass
    
    
    print("\n=========== Local ===========",end='\n') 
    print("Download the file to Cloud") 
    for num in range (7):   
        print("")
        print("Download the file, Party{} ...".format(num+1))
        Download_File.main(is_download_file_function=bool(True), 
                            drive_service_folder_name=None, 
                            download_drive_service_name='Party'+str(num+1)+'_Complete.npy', 
                            download_file_path=os.getcwd() + '/download/',
                            cred_download_file='credentials_peng01.json',
                            token_download_file='token_download_peng01.pickle',
                            time_now=time_now_utc)
    

    # print("Load .npy \n\n")
    img_party1_S = np.load('./download/Party1_Complete.npy')
    img_party2_S = np.load('./download/Party2_Complete.npy')
    img_party3_S = np.load('./download/Party3_Complete.npy')   
    img_party4_S = np.load('./download/Party4_Complete.npy')
    img_party5_S = np.load('./download/Party5_Complete.npy')
    img_party6_S = np.load('./download/Party6_Complete.npy')  
    img_party7_S = np.load('./download/Party7_Complete.npy')
    party1to7 = [img_party1_S, img_party2_S, img_party3_S, img_party4_S, img_party5_S, img_party6_S, img_party7_S]
    
    img_size = np.shape(img_party1_S)
    print("\nError Detection\n")   
    
    k = int((n-t-1)/2)
    print("The maximum of the number of error : ",k)                            # 錯誤數量最大值(k_max)      
    
    for z in range(100):
        i = random.randint(0,img_size[0]-1)
        j = random.randint(0,img_size[1]-1)
        shares = []
        for num in range(7):            
            shares.append(party1to7[num][i][j])
        
        matrix_A = ErrorCorrection.Matrix_A(np.array(shares),k,t,n)                                          # 矩陣A
        
        matrix_b = ErrorCorrection.Matrix_b(np.array(shares),k,n)                                            # 矩陣b
        
        if t+2*k+1 == n:
            if round(np.linalg.det(matrix_A)) < 100:    
                matrix_x = ErrorCorrection.Matrix_x_Singular(matrix_A,matrix_b,k,t,n,prime_num)     
            else:
                matrix_x = ErrorCorrection.Matrix_x(matrix_A,matrix_b,k,t,n,prime_num)  
        else:
            matrix_x = ErrorCorrection.Matrix_x_Pseudo(matrix_A,matrix_b)                           # 解答(矩陣x)
            Flag_Pseudo = True
        
        error_party_temp, fun_e = ErrorCorrection.Solution(matrix_x,k,n,prime_num)                        # 錯誤的party   

        error_party_temp.sort()
        error_dict = Counter(error_party_temp)
        for party, count in error_dict.items():
            if error_party_check.get(party) == None:
                error_party_check[party] = count
            else:
                error_party_check[party] += 1 
                
    error_party = []    
    print("\n")
    for party, num in error_party_check.items():
        print("{} : {} ".format(party,num))
        if num >=100*0.3:
            error_party.append(party)
    
    if (not error_party) or (len(error_party) == 1 and error_party[0] == 0):
        print("\nThere is no error in shares.\n")
    else:
        error_party.sort()
        print("\nThe error party is : ", error_party, end='\n\n')     
    
    print("----------- Reconstruction -----------")
    coefficient, rec_party = [], []
    count = 1
    while(len(coefficient) < 3):
        if count not in error_party:
           coefficient.append(count)  
           rec_party.append(party1to7[count-1])
        count +=1
    print("\nThe reconstructed parties are : ", coefficient, end='\n')     
    largrange =  LagrangeInterpolation(coefficient)

    Image = Construction(rec_party,largrange)
          
    plt.figure(4)
    plt.imshow((Image**(1/2)/16), cmap='Greys_r') # 顯示圖片
    plt.axis('off') # 不顯示座標軸
    

end_time = time.time()     
print("")        
print("總共花費時間 : ",round(end_time-start_time, 2),"sec")       
          

