import cv2
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

random_state = 0
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
            random_num = random.uniform(-10,10)
            for k in range  (int(-n/2), int((n/2)+1)):
                if (k != 0 and k < 0):
                    img_party_result[k+int(n/2)][i][j][0] = img_R[i][j] + random_num * (k)   
                    img_party_result[k+int(n/2)][i][j][1] = img_G[i][j] + random_num * (k) 
                    img_party_result[k+int(n/2)][i][j][2] = img_B[i][j] + random_num * (k) 
                if (k != 0 and k > 0):
                    img_party_result[k+int(n/2)-1][i][j][0] = img_R[i][j] + random_num * (k)  
                    img_party_result[k+int(n/2)-1][i][j][1] = img_G[i][j] + random_num * (k) 
                    img_party_result[k+int(n/2)-1][i][j][2] = img_B[i][j] + random_num * (k) 
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

#------------------------------------ Voting ----------------------------------#
def Voting(s1,s2):
   
    vote = (s1+s2)/2
       
    return vote
    
#------------------------------------  Main ----------------------------------#
if __name__ == '__main__': 
    
    num = 100
    n = 8  
    
    # show image
    class_name = ['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']

    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    
    #pretreatment
    # x_test_normalize = x_train.astype('float32')/255.0      
    x_test_normalize = x_test 
    
    model = tf.keras.models.load_model("./model/VGG16_b10_e15_0001_09_77.h5")
    
    # predict the same model but no softmax
    model2 = tf.keras.Model(inputs=model.input, outputs=model.get_layer(index=-1).output)
    
    accuracy, acc = 0, 0
    h = 0       
    for testing_num in range (0,num): 
      
        if (testing_num%10 == 0):   
            print("")
            print("testing {} times".format(testing_num+1), end ='')
        if (testing_num%2 == 0):
            print('.', end = '')  
            
        # print("---------------- Distributon ----------------")
        party = Distribution(x_test_normalize[testing_num]) 
        
        # print("------------------- Predict -------------------")
        probabilities = model2.predict(x_test_normalize[testing_num].reshape(1,32,32,3))
        probabilities_ = softmax(probabilities)
            
        probabilities_party = []
        probabilities_p_softmax = []
        for k in range (n):
            probabilities_party.append(model2.predict(party[k].reshape(1,32,32,3)))
            probabilities_p_softmax.append(softmax(probabilities_party[k]/probabilities_party[k].sum()))
        
        # print("---------------- Reconstruction ----------------") 
        prob_rec = Reconstruction(np.array(probabilities_party))
        
        prob_rec_softmax = softmax(prob_rec)
           
        predicted_label = np.argmax(probabilities_)
        predicted_label_rec = np.argmax(prob_rec_softmax)
        
        if (y_test[testing_num,0] == np.argmax(prob_rec_softmax)):
            accuracy+=1
        if (y_test[testing_num,0] == np.argmax(probabilities_)):
            acc+=1
    
    print("")
    print("\nAcuuracy with MPC : {}".format(accuracy/num))
    print("Acuuracy without MPC : {}".format(acc/num))
 