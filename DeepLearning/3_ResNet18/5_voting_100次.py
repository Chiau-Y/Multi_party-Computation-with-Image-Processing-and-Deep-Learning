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
    global total

    c_p = np.array([5.0, -10.0, 10.0, -5.0, 1.0]) # 4,-6, 4, -1  # 3, -3, 1
    c_n = np.array([1.0, -5.0, 10.0, -10.0, 5.0]) # -1, 4, -6, 4  # 1, -3, 3
    
    total_positive, total_negative = 0, 0
    for i in range (int(n/2)):
        total_positive += share[i+int(n/2)]*c_p[i]
        total_negative += share[i]*c_n[i]

    return total_positive, total_negative

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
    n = 10
    
    # show image
    class_name = ['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']

    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    
    #pretreatment
    x_test_normalize = x_test.astype('float32')/255.0      
    
    model = tf.keras.models.load_model("./model/ResNet18_b10_e13_lr0005_08")
    
    # predict the same model but no softmax
    model2 = tf.keras.Model(inputs=model.input, outputs=model.get_layer(index=-1).output)
    
    accuracy, acc_p, acc_n, acc = 0, 0, 0, 0
    h = 0       
    for testing_num in range (10000-num,10000):   
        
        if (testing_num%10 == 0):   
            print("")
            print("testing {} times".format(testing_num-10000+num+1), end ='')
        if (testing_num%2 == 0):
            print('.', end = '')
                
        # print("---------------- Distributon ----------------")
        party = Distribution(x_test_normalize[testing_num].reshape(32,32,3)) 
                
        # --------------------------------- Predict --------------------------------- #
        probabilities = model2.predict(x_test_normalize[testing_num].reshape(1,32,32,3))
        probabilities_ = softmax(probabilities)
        
        probabilities_party = []
        for k in range (n):
            probabilities_party.append(model2.predict(party[k].reshape(1,32,32,3)))
        
        # print("---------------- Reconstruction ----------------") 
        prob_p, prob_n = Reconstruction(np.array(probabilities_party))
        
        prob_p_softmax = softmax(prob_p)
        prob_n_softmax  = softmax(prob_n)
        
        probabilities_softmax = Voting(prob_p, prob_n)
        prob_softmax  = softmax(probabilities_softmax)
           
        if (y_test[testing_num,0] == np.argmax(prob_softmax)):
            accuracy+=1
        if (y_test[testing_num,0]== np.argmax(prob_p_softmax)):
            acc_p+=1
        if (y_test[testing_num,0] == np.argmax(prob_n_softmax)):
            acc_n+=1
        if (y_test[testing_num,0] == np.argmax(probabilities_)):
            acc+=1
    
    print("")
    print("\nAcuuracy of Voting : {}".format(accuracy/num))
    print("Acuuracy of Positive Party : {}".format(acc_p/num))
    print("Acuuracy of Negative Party : {}".format(acc_n/num))
    print("Acuuracy without MPC : {}".format(acc/num))
 