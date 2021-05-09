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
    global total

    c_p = np.array([5, -10, 10, -5, 1]) # 4,-6, 4, -1  # 3, -3, 1
    c_n = np.array([1, -5, 10, -10, 5]) # -1, 4, -6, 4  # 1, -3, 3
    
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
    # x_test_normalize = x_train.astype('float32')/255.0      
    x_test_normalize = x_test 
    
    model = tf.keras.models.load_model("VGG16_b40_e7.h5")
    
    # predict the same model but no softmax
    model2 = tf.keras.Model(inputs=model.input, outputs=model.get_layer(index=-1).output)
    
    testing_num = random.randint(10000-num+1,10000-1)  
    
    # print("---------------- Distributon ----------------")
    party = Distribution(x_test_normalize[testing_num]) 
    
    # --------------------------------- Predict --------------------------------- #
    probabilities = model2.predict(x_test_normalize[testing_num].reshape(1,32,32,3))
    probabilities_ = softmax(probabilities)
        
    probabilities_party = []
    probabilities_p_softmax = []
    for k in range (n):
        probabilities_party.append(model2.predict(party[k].reshape(1,32,32,3)))
        probabilities_p_softmax.append(softmax(probabilities_party[k]/probabilities_party[k].sum()))
        
    # for i, prob in enumerate(probabilities_p_softmax):
    #     plt.figure(i)
    #     if i >= 5:
    #         plt.title("No.{} party".format(i-4)) 
    #     else:
    #         plt.title("No.{} party".format(i-5)) 
    #     plt.grid(False)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.imshow(party[i], cmap='binary')
    #     predicted_label = np.argmax(prob)
    #     plt.xlabel("{} {:2.0f}% ".format(class_name[predicted_label], 100*np.max(prob)))       
    
    # print("---------------- Reconstruction ----------------") 
    prob_p, prob_n = Reconstruction(np.array(probabilities_party))
    
    prob_p_softmax = softmax(prob_p)
    prob_n_softmax  = softmax(prob_n)
    
    probabilities_softmax = Voting(prob_p, prob_n)
    prob_softmax  = softmax(probabilities_softmax)
    
    predicted_label = np.argmax(probabilities_)
    predicted_label_p = np.argmax(prob_p_softmax)
    predicted_label_n = np.argmax(prob_n_softmax)
    predicted_label_v = np.argmax(prob_softmax)
    
    print("The Prediction of No.1 ~ {} party :  {:.0f}%  for  {}".format(int(n/2),
                                                                     100*np.max(prob_p_softmax),class_name[predicted_label_p]))
    print("The Prediction of No.{} ~ -1 party :  {:.0f}%  for  {}".format(int(-n/2),
                                                                      100*np.max(prob_n_softmax),class_name[predicted_label_n]))
    print("The Prediction of all party with voting :  {:.0f}%  for  {}".format(100*np.max(prob_softmax),
                                                                           class_name[predicted_label_v]))
    print("The Prediction without MPC :  {:.0f}%   for {}".format(100*np.max(probabilities_),
                                                              class_name[predicted_label]))
    
    print("")
    print("It is a/an {}.".format(class_name[y_test[testing_num,0]]))
   
    # plt.figure(11)
    # plt.title("The Prediction without MPC")
    # plt.grid(False)
    # plt.xticks([])
    # plt.yticks([])
    # plt.imshow(x_test_normalize[testing_num])
    # plt.xlabel("{} {:2.0f}% ".format(class_name[predicted_label], 100*np.max(probabilities_)))
    
    # plt.figure(12)
    # plt.title("The Prediction with MPC")
    # plt.grid(False)
    # plt.xticks([])
    # plt.yticks([])
    # plt.imshow(x_test_normalize[testing_num])
    # plt.xlabel("{} {:2.0f}% ".format(class_name[predicted_label_v], 100*np.max(prob_softmax)))
    
    # # estimate this test image
    # probabilities_array = probabilities_.reshape(10) 
    # plt.figure(24)
    # plt.grid(False)
    # plt.xticks(range(10), class_name, fontsize=15, rotation=45)
    # plt.ylim([0, 1])
    # plt.bar(range(10), probabilities_array)
    # plt.show()  
    


    probabilities_party = []
    for k in range (n):
        probabilities_party.append(model2.predict(party[k].reshape(1,32,32,3)))
        
    x = np.arange(int(-n/2),int(n/2)+1)
    x_ = np.delete(x, int(n/2))
    
    for i in range (10):
        plt.figure(i+13)
        plt.title("The Probability of {} ".format(i+1)) 
        y_value = np.array(probabilities_party)[:,0,i]
        plt.plot(x_,y_value,'b.')
        plt.plot(0,probabilities[0][i],'r.')