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
    
    # show image
    class_name = ['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']

    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    
    #pretreatment
    x_test_normalize = x_test.astype('float32')/255.0      
    # x_test_normalize = x_test 
    
    model = tf.keras.models.load_model("./model/ResNet18_b10_e13_lr0005_08")#, custom_objects={'softplus':tf.keras.activations.softplus})
    
    # predict the same model but no softmax
    model2 = tf.keras.Model(inputs=model.input, outputs=model.get_layer(index=-1).output)
    
    testing_num = random.randint(0,num)  
    
    # print("---------------- Distributon ----------------")
    party = Distribution(x_test[testing_num]) 
    
    # --------------------------------- Predict --------------------------------- #
    probabilities = model2.predict(x_test_normalize[testing_num].reshape(1,32,32,3))
    probabilities_ = softmax(probabilities)
        
    probabilities_party = []
    probabilities_p_softmax = []
    for k in range (n):
        probabilities_party.append(model2.predict(party[k].reshape(1,32,32,3)))
        probabilities_p_softmax.append(softmax(probabilities_party[k]/probabilities_party[k].sum()))
        
    for i, prob in enumerate(probabilities_p_softmax):
        plt.figure(i)
        if i >= int(n/2):
            plt.title("No.{} party".format(i-int(n/2)+1)) 
        else:
            plt.title("No.{} party".format(i-int(n/2))) 
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(cv2.convertScaleAbs(party[i]*255))
        predicted_label = np.argmax(prob)
        plt.xlabel("{} {:2.0f}% ".format(class_name[predicted_label], 100*np.max(prob)))     
    
    # print("---------------- Reconstruction ----------------") 
    prob_rec = Reconstruction(np.array(probabilities_party))
    
    prob_rec_softmax = softmax(prob_rec)
       
    predicted_label = np.argmax(probabilities_)
    predicted_label_rec = np.argmax(prob_rec_softmax)
    
    print("The Prediction with MPC :  {:.0f}%  for  {}".format(100*np.max(prob_rec_softmax),
                                                                class_name[predicted_label_rec]))
    print("The Prediction without MPC :  {:.0f}%   for {}".format(100*np.max(probabilities_),
                                                              class_name[predicted_label]))
    
    print("")
    print("It is a/an {}.".format(class_name[y_test[testing_num,0]]))
   
    plt.figure(11)
    plt.title("The Prediction without MPC")
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_test_normalize[testing_num])
    plt.xlabel("{} {:2.0f}% ".format(class_name[predicted_label], 100*np.max(probabilities_)))
    
    # # plt.figure(12)
    # # plt.title("The Prediction with MPC")
    # # plt.grid(False)
    # # plt.xticks([])
    # # plt.yticks([])
    # # plt.imshow(x_test_normalize[testing_num])
    # # plt.xlabel("{} {:2.0f}% ".format(class_name[predicted_label_v], 100*np.max(prob_softmax)))
    
    # estimate this test image
    probabilities_array = probabilities_.reshape(10) 
    plt.figure(24,figsize=(20,8))
    plt.grid(False)
    plt.xticks(range(10), class_name, fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim([0, 1])
    barlist = plt.bar(range(10), probabilities_array)
    barlist[predicted_label].set_color('r')
    plt.show()
    
        
    # x = np.arange(int(-n/2),int(n/2)+1)
    # x_ = np.delete(x, int(n/2))
    
    # for i in range (10):
    #     plt.figure(i+13)
    #     plt.title("The Probability of {} ".format(i+1)) 
    #     y_value = np.array(probabilities_party)[:,0,i]
    #     plt.plot(x_,y_value,'b.')
    #     plt.plot(0,probabilities[0][i],'r.')