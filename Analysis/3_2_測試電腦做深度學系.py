import cv2
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

import psutil
import functools
from threading import Timer

import time
from datetime import datetime

random_state = 0
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
#------------------------------------ Distribution ----------------------------------# 
def Distribution(img):
    
    global img_R, img_G, img_B, imgg
    imgg = img
    img_party_result = np.zeros((n,img.shape[0], img.shape[1],3))
    shares = np.zeros(n)
    
    img_R = img[:,:,0]
    img_G = img[:,:,1]
    img_B = img[:,:,2]      
    
    for i in range (img.shape[0]):       
        for j in range(img.shape[1]):      
            # while(img_party_result[n-1][i][j]==0):  
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
            # if (shares[n-1] <= 255 and shares[n-1] >= 0 and shares[0] <= 255 and shares[0] >= 0): 
            #     for z in range (n):
            #         img_party_result[k][i][j] = shares[z] 
                        
    return img_party_result
    
#------------------------------------ Reconstruction ----------------------------------# 
def Reconstruction(share):

    c_p = np.array([0.75, -0.3, 0.05]) # 4,-6, 4, -1  # 3, -3, 1
    c_n = np.array([0.05, -0.3, 0.75]) # 4,-6, 4, -1  # 3, -3, 1

    total = 0
    for i in range (int(n/2)):
        total += share[i+int(n/2)]*c_p[i]
        total += share[i]*c_n[i]

    return total

#------------------------------------ Softmax ----------------------------------#
def softmax(x):
    """Compute the softmax of vector x."""
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x 

#------------------------------------ Voting ----------------------------------#
def Voting(s1,s2):
   
    vote = (s1+s2)/2
       
    return vote
    
#------------------------------------  Main ----------------------------------#
if __name__ == '__main__': 
    
    num = 100
    n = 6   
    
    cpu = []
    time_axis = []
    
    # show image
    class_name = ['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']

    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    
    #pretreatment
    x_test_normalize = x_test.astype('float32')/255.0      
    # x_test_normalize = x_test 
    
    testing_num = 100  
    
    # print("---------------- Distributon ----------------")
    party = Distribution(x_test[testing_num]) 
    
    t_cpu = RepeatingTimer(1.0, hello)
    t_cpu.start()
    time.sleep(3)   
    model = tf.keras.models.load_model("./model/ResNet18_b10_e13_lr0005_08")#, custom_objects={'softplus':tf.keras.activations.softplus})
    
    # predict the same model but no softmax
    model2 = tf.keras.Model(inputs=model.input, outputs=model.get_layer(index=-1).output)    
    # --------------------------------- Predict --------------------------------- #
    probabilities = model2.predict(x_test_normalize[testing_num].reshape(1,32,32,3))
    probabilities_ = softmax(probabilities)
        
    probabilities_party = []
    probabilities_p_softmax = []
    for k in range (n):
        probabilities_party.append(model2.predict(party[k].reshape(1,32,32,3)))
        
    time.sleep(3)
    t_cpu.cancel() 
    
    plt.figure(20)
    plt.plot(time_axis,cpu, label='CPU')
    plt.ylabel('CPU%')
    plt.ylim(0,100)
    plt.xticks(rotation=45)
    plt.show()   
    
    # data = []
    # data.append(cpu)
    # data.append(time_axis)
    # np.save('DP_cpu', data)