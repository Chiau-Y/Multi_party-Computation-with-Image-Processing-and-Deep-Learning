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
            while(img_party_result[n-1][i][j]==0):  
                random_num = random.uniform(-10,10)
                for k in range  (int(-n/2), int((n/2)+1)):
                    if (k != 0 and k < 0):
                        img_party_result[k+int(n/2)][i][j] = img[i][j] + random_num * (k)   
                    if (k != 0 and k > 0):
                        img_party_result[k+int(n/2)-1][i][j] = img[i][j] + random_num * (k)  
                # if (shares[n-1] <= 255 and shares[n-1] >= 0 and shares[0] <= 255 and shares[0] >= 0): 
                #     for z in range (n):
                #         img_party_result[k][i][j] = shares[z] 
                        
    return img_party_result

#------------------------------------ LeNet-5, training ----------------------------------# 
def LeNet5():
    global x_train_normalize, y_train, model
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train_padded = np.pad(x_train[:,], ((0,0),(2,2), (2,2)), 'constant')
    x_test_padded = np.pad(x_test[:,], ((0,0),(2,2), (2,2)), 'constant')
    
    x_train_reshaped = x_train_padded.reshape((60000, 32, 32, 1))
    x_test_reshaped = x_test_padded.reshape((10000, 32, 32, 1))
    
    #pretreatment
    x_train_normalize = x_train_reshaped.astype('float32')/255.0
    x_test_normalize = x_test_reshaped.astype('float32')/255.0
    
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    x_train_normalize_ = x_train_normalize[0:60000-num]
    y_train_ = y_train[0:60000-num]
    
    model= models.Sequential()
    model.add(layers.Conv2D(filters=6,kernel_size=(5,5),input_shape=(32,32,1),activation='softplus'))
    model.add(layers.AveragePooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(filters=16, kernel_size=(5,5),activation='softplus'))
    model.add(layers.AveragePooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(120,activation='softplus'))
    model.add(layers.Dense(84,activation='softplus'))
    model.add(layers.Dense(10))
    model.add(layers.Activation('softmax'))
    
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate_set),
                  loss = 'mean_squared_error', metrics = ['accuracy'])
    
    # mean_squared_error/sparse_categorical_crossentropy/categorical_crossentropy
    
    history = model.fit(x = x_train_normalize_, y = y_train_, validation_data = 
                        (x_test_normalize, y_test),epochs = epochs_set, batch_size = batch_size_set)
    
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

# ------------------------------------ Main ---------------------------------- #
if __name__ == '__main__':
    #setting
    learning_rate_set = 0.001
    batch_size_set = 1000
    epochs_set = 5
    
    num = 100
    
    n = 10
    
    # show image
    class_name = ['0', '1', '2', '3', '4','5', '6', '7', '8', '9']
    
    # LeNet5()
    
    # model.save("my_model_cross_entropy_01.h5")
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train_padded = np.pad(x_train[:,], ((0,0),(2,2), (2,2)), 'constant')
    
    x_train_reshaped = x_train_padded.reshape((60000, 32, 32, 1))
    
    #pretreatment
    x_train_normalize = x_train_reshaped.astype('float32')/255.0
    
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    
    x_train_normalize_ = x_train_normalize[0:60000-num]
    y_train_ = y_train[0:60000-num]       
    
    model = tf.keras.models.load_model("./model/my_model_cross_entropy_batch1000.h5")
    
    # predict the same model but no softmax
    model2 = tf.keras.Model(inputs=model.input, outputs=model.get_layer(index=7).output)
    
    testing_num = random.randint(60000-num+1,60000-1)          
    # print("---------------- Distributon ----------------")
    party = Distribution(x_train_normalize[testing_num].reshape(32,32)) 
             
    # --------------------------------- Predict --------------------------------- #
    probabilities = model2.predict(x_train_normalize[testing_num].reshape(1,32,32,1))
    probabilities_ = softmax(probabilities)
        
    probabilities_party = []
    probabilities_p_softmax = []
    for k in range (n):
        probabilities_party.append(model2.predict(party[k].reshape(1,32,32,1)))
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
    print("It is {}.".format(np.argmax(y_train[testing_num])))

    # plt.figure(11)
    # plt.title("The Prediction without MPC")
    # plt.grid(False)
    # plt.xticks([])
    # plt.yticks([])
    # plt.imshow(x_train_normalize[testing_num][:,:,0], cmap='binary')
    # plt.xlabel("{} {:2.0f}% ".format(class_name[predicted_label], 100*np.max(probabilities_)))
    
    # plt.figure(12)
    # plt.title("The Prediction with MPC")
    # plt.grid(False)
    # plt.xticks([])
    # plt.yticks([])
    # plt.imshow(x_train_normalize[testing_num][:,:,0], cmap='binary')
    # plt.xlabel("{} {:2.0f}% ".format(class_name[predicted_label_v], 100*np.max(prob_softmax)))
    
    probabilities_party = []
    for k in range (n):
        probabilities_party.append(model2.predict(party[k].reshape(1,32,32,1)))
        
    x = np.arange(int(-n/2),int(n/2)+1)
    x_ = np.delete(x, int(n/2))
    
    # for i in range (10):
    #     plt.figure(i+13)
    #     plt.title("The Probability of {} ".format(i+1)) 
    #     y_value = np.array(probabilities_party)[:,0,i]
    #     plt.plot(x_,y_value,'b.')
    #     plt.plot(0,probabilities[0][i],'r.')