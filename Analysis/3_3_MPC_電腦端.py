import cv2
import time
import random
import os.path
import numpy as np
from datetime import datetime
from scipy import misc, ndimage
import matplotlib.pyplot as plt
import Upload_File, Download_File

import psutil
import functools
from threading import Timer


start_time = time.time()
# ------------------------------------Subroutine------------------------------------
def hello():
    global cpu , time_axis   
    cpu.append(psutil.cpu_percent(interval=0.1))
    time_axis.append(datetime.utcnow().strftime("%M:%S"))
    
class RepeatingTimer(Timer): 
    def run(self):
        while not self.finished.is_set():
            self.function(*self.args, **self.kwargs)
            self.finished.wait(self.interval)

#------------------------------------ Distribution ----------------------------------# 
def Distribution(img):
        
    img_party_result = np.zeros((6,img.shape[0], img.shape[1]))
    
    for i in range (img_size[0]):       
        for j in range(img_size[1]):
            random_num = random.randint(0,100000)
            img_party_result[0][i][j] = (img[i][j] + random_num * 1)*16 % 33292801
            img_party_result[1][i][j] = (img[i][j] + random_num * 2)*16 % 33292801
            img_party_result[2][i][j] = (img[i][j] + random_num * 3)*16 % 33292801
            img_party_result[3][i][j] = (img[i][j] + random_num * 4)*16 % 33292801
            img_party_result[4][i][j] = (img[i][j] + random_num * 5)*16 % 33292801
            img_party_result[5][i][j] = (img[i][j] + random_num * 6)*16 % 33292801
                      
    return img_party_result

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
    cpu = []
    time_axis = []
    t_cpu = RepeatingTimer(1, hello)
    t_cpu.start()
    time.sleep(3)
    # print("=========== Local ===========")
    # img_original = cv2.imread('D:/Master_2019_2021/Muti_Party_Computation/python/image/Lenna_.bmp',0)  # GRAY    
    # img_size = np.shape (img_original)
    
    # plt.figure(0)
    # plt.imshow(img_original, cmap='Greys_r') # 顯示圖片
    # plt.axis('off') # 不顯示座標軸  
    
    # print("Distributon")
    # img_party = Distribution(img_original) 
    # print("Save .npy",end='')
    # np.save('Party1', img_party[0])
    # np.save('Party2', img_party[1])
    # np.save('Party3', img_party[2]) 
    # np.save('Party4', img_party[3]) 
    # np.save('Party5', img_party[4]) 
    # np.save('Party6', img_party[5]) 
    
    # # print("----------- Upload the file to Cloud -----------")
    cred_token_party = ['peng01','peng03','peng05','chen03','chen16','chen21']
    time_now_utc = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
    us = datetime.utcnow().strftime("%M:%S") 
    for num in range (1):   
        # print("Upload the file to Party{}".format(num+1),end='\n')
        Upload_File.main(is_update_file_function=bool(True),
                          update_drive_service_folder_name = None,
                          update_drive_service_name='Party'+str(num+1)+'.npy', 
                          update_file_path=os.getcwd() + './',
                          cred_file='credentials_'+cred_token_party[num]+'.json',
                          token_file='token_'+cred_token_party[num]+'.pickle')
    ue = datetime.utcnow().strftime("%M:%S")    
    # print("=========== Cloud Computing ===========",end='\n')   
    
    # while(not Download_File.main(is_download_file_function=bool(False), 
    #                               drive_service_folder_name=None, 
    #                               download_drive_service_name='Party1_Complete.npy', 
    #                               download_file_path=os.getcwd() + '/download/',
    #                               cred_download_file='credentials_'+cred_token_party[0]+'.json',
    #                               token_download_file='token_download_'+cred_token_party[0]+'.pickle',
    #                               time_now=time_now_utc)):
    #     pass
    
    time.sleep(15)
    
    # print("=========== Local ===========",end='\n') 
    # print("----------- Download the file to Cloud -----------")
    ds = datetime.utcnow().strftime("%M:%S")    
    for num in range (1):   
        # print("")
        # print("Download the file from Party{}".format(num+1))
        Download_File.main(is_download_file_function=bool(True), 
                            drive_service_folder_name=None, 
                            download_drive_service_name='Party'+str(1)+'_Complete.npy', 
                            download_file_path=os.getcwd() + '/download/',
                            cred_download_file='credentials_peng01.json',
                            token_download_file='token_download_peng01.pickle',
                            time_now=time_now_utc)
    
    de = datetime.utcnow().strftime("%M:%S")    

    # print("Load .npy")
    # img_party1_S = np.load('./download/Party1_Complete.npy')
    # img_party2_S = np.load('./download/Party2_Complete.npy')
    # img_party3_S = np.load('./download/Party3_Complete.npy')       
    
    # print("Reconstruction")
    # img_size = np.shape (img_party1_S)
    # Image12, Image23, Image13, Image = Construction(img_party1_S,img_party2_S,img_party3_S)
          
    # plt.figure(4)
    # plt.imshow((Image12**(1/2)/16), cmap='Greys_r') # 顯示圖片
    # plt.axis('off') # 不顯示座標軸
    # plt.figure(5)
    # plt.imshow((Image23**(1/2)/16), cmap='Greys_r') # 顯示圖片
    # plt.axis('off') # 不顯示座標軸
    # plt.figure(6)
    # plt.imshow((Image13**(1/2)/16), cmap='Greys_r')
    # plt.axis('off') # 不顯示座標軸
    # plt.figure(7)   
    # plt.imshow((Image**(1/2)/16), cmap='Greys_r')
    # plt.axis('off') # 不顯示座標軸 
 

end_time = time.time()     
# print("")        
# print("總共花費時間 : ",round(end_time-start_time, 2),"sec")       
     
time.sleep(3)
t_cpu.cancel() 

plt.figure(20)
plt.plot(time_axis,cpu, label='CPU')
plt.ylabel('CPU%')
plt.ylim(0,100)
plt.xticks(rotation=45)
plt.show()        

# plt.figure(8)
# plt.imshow(np.uint8(img_party3), cmap='Greys_r') # 顯示圖片
# plt.axis('off') # 不顯示座標軸

# data = []
# data.append(cpu)
# data.append(time_axis)
# np.save('CP_cpu', data)