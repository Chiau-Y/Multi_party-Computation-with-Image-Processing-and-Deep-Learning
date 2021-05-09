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

def Matrix_x_Singular(matrix1,matrix2,k,t):                                  # 奇異矩陣求解
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

def Solution(matrix,flag,k,n):                                               # 多項式求解(error)
    solve1, solution2 = [], []
    solve1.append(1)
    for i in range (k):
        solve1.append(matrix[(k-1)-i][0])
    solve2 = np.poly1d(np.array(solve1))
    solution1 = solve2.roots  
    for i in range (k):
        if flag :    # Non-square Matrix 
            if int(solution1[i].real) > 0 and solution1[i] <= n and isinstance(solution1[i], (int, float)): # 會有複數
                solution2.append(int(round(solution1[i])))
        else:
            if isinstance(solution1[i], (int, float)):
                if solution1[i] > 0 and int(solution1[i]) <= n :#and abs(math.fmod(round(solution1[i],0),1)) < 1e-6:
                    solution2.append(int(round(solution1[i])))
    return solution2, solve2