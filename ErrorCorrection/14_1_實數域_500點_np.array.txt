import cv2
import time
import math
import random
import numpy as np
from sympy import * 
from collections import Counter
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial as P

start_time = time.time()
# ------------------------------------Subroutine------------------------------------
def Check_n(n,t):                                                            # 設定party個數                   
    if n < 2*t or n <= 0:
        print("You need more parties")
        check = 1
    else:
        check = 0
    return check

def Matrix_A(shares_err,k,t,n):                                             
    matrix_A1, matrix_A2 = [], []
    share = shares_err
    k1 = k
    for i in range (n):
        matrix_A1.append(share[i])
        for j in range (k-1):
            power = 1
            for z in range (j+1):
                power *= (i+1)
            matrix_A1.append(share[i]*power)   
    for i in range (n):
        matrix_A2.append(-1)
        num = 0
        for j in range (t+k):
            power = 1
            for z in range (j+1):
                power *= (i+1)
            matrix_A2.append((-1)*power) 
    if k == 0:
        k1 += 1
   
    return np.append(np.array(matrix_A1).reshape(n,k1), np.array(matrix_A2).reshape(n, t+k+1), axis=1) 

def Matrix_b(share,k,n):                                                   
    matrix_b1 = []
    for i in range (1,n+1):
        power = 1
        for j in range (k):
            power *= i            
        matrix_b1.append(share[i-1]*power*(-1))      
    return np.array(matrix_b1).reshape(n,1)
    
def Matrix_x(matrix1,matrix2):                                             # 矩陣求解
    # print("\n===== Square and Non-Singular Matrix=====")
    Augmented = Matrix(np.hstack((matrix1,matrix2)))
    A_inv = np.linalg.inv(matrix1)
    ans = A_inv.dot(matrix2)
    return ans

def Matrix_x_Pseudo(matrix1,matrix2):                                      # 非方正矩陣求解
    # print("\n===== Non-square Matrix =====")
    A_inv = np.linalg.pinv(np.mat(matrix1))
    ans = np.dot(A_inv,matrix2)   
    return np.array(ans)

def Matrix_x_Singular(matrix1,matrix2):                                  # 奇異矩陣求解
    # print("\n===== Singular Matrix =====")
    Augmented_matrix = Matrix(np.hstack((matrix1,matrix2))).applyfunc(nsimplify)  
    matrix_rref = np.array(Augmented_matrix.rref()[0])
    ans = np.zeros([t+2*k+1,1])
    size = matrix_rref.shape
    for i in range(size[0]-1,-1,-1):
        try:           
            row = list(matrix_rref[i]).index(1)
            total = 0
            for j in range(row+1,size[1]-1):
                total += matrix_rref[i][j] * ans[j][0]  # 通解t設為零要加* ans[j][0]，設為1不用加
            ans[row][0] = matrix_rref[i][size[1]-1] - total
        except:
            pass        
    return ans 

def Solution(matrix,flag):                                               # 多項式求解(error)
    solve1, solution2 = [], []
    solve1.append(1)
    for i in range (k):
        solve1.append(matrix[(k-1)-i][0])
    solve2 = np.poly1d(np.array(solve1))
    solution1 = solve2.roots  
    # print(solution1)
    for i in range (k):
        if flag :    # Non-square Matrix 
            if int(solution1[i].real) > 0 and solution1[i] <= n and isinstance(solution1[i], (int, float)): # 會有複數
                solution2.append(int(round(solution1[i])))
        else:
            if isinstance(solution1[i], (int, float)):
                if solution1[i] > 0 and int(solution1[i]) <= n :#and abs(math.fmod(round(solution1[i],0),1)) < 1e-6:
                    solution2.append(int(round(solution1[i])))
    return solution2, solve2


#------------------------------------ Image to Array ----------------------------------#
def Image2Array(img_party):  
    
    img_gray = np.zeros([img_size[0], img_size[1]], dtype='float32')
    
    for i in range (img_size[0]):       
        for j in range(img_size[1]): 
            img_gray[i][j] = int(img_party[:,:,0][i][j]) + int(img_party[:,:,1][i][j]) + int(img_party[:,:,2][i][j])
    
    return img_gray 

# ------------------------------------Main program------------------------------------
check_n, Flag_Pseudo = True, False
matrix_A1, matrix_A2, matrix_b1, matrix_x1 = [], [], [], []
answer, solve1, solution2, solve3, solution3 = [], [], [], [], []
error_party_check = {}
t, n = 1, 6                                                         # 次方

print("\n-------------------Start-------------------")        

# party1 = cv2.imread('./parties/ImageProcessing/party1_real.bmp')  
# party2 = cv2.imread('./parties/ImageProcessing/party2_real.bmp')
# party3 = cv2.imread('./parties/ImageProcessing/party3_real.bmp')
# party4 = cv2.imread('./parties/ImageProcessing/party4_real.bmp')
# party5 = cv2.imread('./parties/ImageProcessing/party5_real.bmp')
# party6 = cv2.imread('./parties/ImageProcessing/party6_real.bmp')   

party1 = np.load('./parties/ImageProcessing/party1_real.npy') 
party2 = np.load('./parties/ImageProcessing/party2_real.npy') 
party3 = np.load('./parties/ImageProcessing/party3_real.npy') 
party4 = np.load('./parties/ImageProcessing/party4_real.npy') 
party5 = np.load('./parties/ImageProcessing/party5_real.npy') 
party6 = np.load('./parties/ImageProcessing/party6_real.npy')  

# print("Image to Array")
img_size = np.shape(party1)
# party1_gray = Image2Array(party1)  
# party2_gray = Image2Array(party2) 
# party3_gray = Image2Array(party3) 
# party4_gray = Image2Array(party4) 
# party5_gray = Image2Array(party5) 
# party6_gray = Image2Array(party6)  

print("Error Correction\n")

while (check_n):
    check_n = Check_n(n,t)                                                  # party數(n)
print("There are {} parties.".format(n))                                  

k = int((n-t-1)/2)
print("\nThe maximum of the number of error : ",k)                            # 錯誤數量最大值(k_max)  

for z in range(500):
    i = random.randint(0,img_size[0]-1)
    j = random.randint(0,img_size[1]-1)
 
    shares = []
    shares.append(party1[i][j])
    shares.append(party2[i][j])
    shares.append(party3[i][j])
    shares.append(party4[i][j])
    shares.append(party5[i][j])
    shares.append(party6[i][j])

    matrix_A = Matrix_A(np.array(shares),k,t,n)                                          # 矩陣A
    
    matrix_b = Matrix_b(np.array(shares),k,n)                                            # 矩陣b
    
    if t+2*k+1 == n:
        if abs(np.linalg.det(matrix_A)) < 1e-06:    
            matrix_x = Matrix_x_Singular(matrix_A,matrix_b)    
        else:
            matrix_x = Matrix_x(matrix_A,matrix_b)  
    else:
        matrix_x = Matrix_x_Pseudo(matrix_A,matrix_b)                           # 解答(矩陣x)
        Flag_Pseudo = True
    
    error_party_temp, fun_e = Solution(matrix_x, Flag_Pseudo)                        # 錯誤的party
    
    error_party_temp.sort()
    error_dict = Counter(error_party_temp)
    for party, count in error_dict.items():
        if party != 0:
            if error_party_check.get(party) == None:
                error_party_check[party] = count
            else:
                error_party_check[party] += 1

error_party = []    
print("\n")
for party, num in error_party_check.items():
    print("{} : {} ".format(party,num))
    if num >= 500*0.4:
        error_party.append(party)

if (not error_party) or (len(error_party) == 1 and error_party[0] == 0):
    print("\nThere is no error in shares.\n")
else:
    print("\nThe error party is : ", error_party)
    
end_time = time.time()     
print("")        
print("Time : ",round(end_time-start_time, 2),"sec")    
print("\n-------------------END-------------------\n")
