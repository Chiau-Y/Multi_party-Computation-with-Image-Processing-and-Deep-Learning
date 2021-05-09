import cv2
import time
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.datasets import mnist
from tensorflow.keras import datasets, layers, models

random_state = 0
#------------------------------------ Distribution ----------------------------------# 
def Distribution(img):

    img_party_result = np.zeros((n,img.shape[0], img.shape[1]))
    shares = np.zeros(n)
        
    
    for i in range (img.shape[0]):       
        for j in range(img.shape[1]):      
            # while(img_party_result[n-1][i][j]==0):  
            random_num = random.uniform(-10,10)
            for k in range  (int(-n/2), int((n/2)+1)):
                if (k != 0 and k < 0):
                    img_party_result[k+int(n/2)][i][j] = (img[i][j] + random_num * (k))/255.0   
                if (k != 0 and k > 0):
                    img_party_result[k+int(n/2)-1][i][j] = (img[i][j] + random_num * (k))/255.0  
            # if (shares[n-1] <= 255 and shares[n-1] >= 0 and shares[0] <= 255 and shares[0] >= 0): 
            #     for z in range (n):
            #         img_party_result[k][i][j] = shares[z] 
                        
    return img_party_result
    
#------------------------------------ Reconstruction ----------------------------------# 
def Reconstruction(share):

    c_p = np.array([0.8, -0.4, 0.1143, -0.0143]) # 0.6667, -0.1667 # 0.75, -0.3, 0.05 # 0.8, -0.4, 0.1143, -0.0143
    c_n = np.array([-0.0143, 0.1143, -0.4, 0.8]) # -0.1667, 0.6667 # 0.05, -0.3, 0.75 # -0.0143, 0.1143, -0.4, 0.8

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


# ------------------------------------ Main ---------------------------------- #
if __name__ == '__main__':
    #setting
    learning_rate_set = 0.001
    batch_size_set = 50
    epochs_set = 10
    
    num = 100
    
    n = 8
    
    # show image
    class_name = ['0', '1', '2', '3', '4','5', '6', '7', '8', '9']
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train_padded = np.pad(x_train[:,], ((0,0),(2,2), (2,2)), 'constant')
    
    x_train_reshaped = x_train_padded.reshape((60000, 32, 32, 1))
    
    #pretreatment
    x_train_normalize = x_train_reshaped.astype('float32')/255.0
    y_train_label = y_train
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    
    x_train_normalize_ = x_train_normalize[0:60000-num]
    y_train_ = y_train[0:60000-num]    
    
    model = tf.keras.models.load_model("./model/my_model_cross_entropy_batch1000.h5")
    
    # predict the same model but no softmax
    model2 = tf.keras.Model(inputs=model.input, outputs=model.get_layer(index=7).output)
             
    accuracy, acc_p, acc_n, acc = 0, 0, 0, 0
    h = 0       
    for testing_num in range (60000-num,60000):   
        
        if (testing_num%10 == 0):   
            print("")
            print("testing {} times".format(testing_num-60000+num+1), end ='')
        if (testing_num%2 == 0):
            print('.', end = '')
                
        # print("---------------- Distributon ----------------")
        party = Distribution(x_train_reshaped[testing_num].reshape(32,32)) 
                
        # --------------------------------- Predict --------------------------------- #
        probabilities = model2.predict(x_train_normalize[testing_num].reshape(1,32,32,1))
        probabilities_ = softmax(probabilities)
            
        probabilities_party = []
        probabilities_p_softmax = []
        for k in range (n):
            probabilities_party.append(model2.predict(party[k].reshape(1,32,32,1)))
            probabilities_p_softmax.append(softmax(probabilities_party[k]/probabilities_party[k].sum()))
        
        # print("---------------- Reconstruction ----------------") 
        prob_rec = Reconstruction(np.array(probabilities_party))
        
        prob_rec_softmax = softmax(prob_rec)
           
        predicted_label = np.argmax(probabilities_)
        predicted_label_rec = np.argmax(prob_rec_softmax)
        
        if (y_train_label[testing_num] == np.argmax(prob_rec_softmax)):
            accuracy+=1
        if (y_train_label[testing_num] == np.argmax(probabilities_)):
            acc+=1
    
    print("")
    print("\nAcuuracy with MPC : {}".format(accuracy/num))
    print("Acuuracy without MPC : {}".format(acc/num))