import cv2
import functools
import numpy as np
from sympy import * 
from itertools import product
from fractions import Fraction
from collections import Counter
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial as P

# ------------------------------------Subroutine------------------------------------
def Egcd(a, b):
    if a == 0:
        return (b, 0, 1)
    else:
        g, y, x = Egcd(b % a, a)
        return (g, x - (b // a) * y, y)

def Modinv(a, m):   # 反模數
    g, x, y = Egcd(a, m)
    if g != 1:
        raise Exception('modular inverse does not exist')
    else:
        return x % m 
 
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
    
def Matrix_x(matrix1,matrix2,k,t,n,prime_num):                                             # 矩陣求解
    # print("\n===== Square and Non-Singular Matrix=====")
    Augmented = Matrix(np.hstack((matrix1,matrix2)), dtype='float64')
    matrix_rref_temp = np.array(Augmented.rref()[0]).astype(str)
    matrix_rref = np.array(Matrix(Augmented.rref()[0]).applyfunc(nsimplify)) 
    
    for i in range(n):
        num_float = Fraction(matrix_rref_temp[i][t+2*k+1])
        if num_float.denominator != 1:
            matrix_rref[i][t+2*k+1] = ((num_float.numerator%prime_num)*Modinv(num_float.denominator,prime_num))%prime_num
            
    size = matrix_rref.shape
    ans = np.zeros([t+2*k+1,1])
    for i in range(size[0]-1,-1,-1):
        try:           
            row = list(matrix_rref[i]).index(1)
            total = 0
            for j in range(row+1,size[1]-1):
                total += matrix_rref[i][j] * ans[j][0]   # 通解t設為零要加* ans[j][0]，設為1不用加
            ans[row][0] = matrix_rref[i][size[1]-1] - total
        except:
            pass      
    return ans   

def Matrix_x_Pseudo(matrix1,matrix2):                                      # 非方正矩陣求解
    # print("\n===== Non-square Matrix =====")
    A_T = np.array(np.transpose(matrix1))
    
    A_star_temp = np.dot(A_T,matrix1)
    A_star_inv_temp = np.linalg.inv(A_star_temp)
    A_star_det_temp = (round(np.linalg.det(A_star_temp))) % prime_num
    A_star_adj = np.around((A_star_inv_temp*round(np.linalg.det(A_star_temp))))
    A_star_det_inv = Modinv(A_star_det_temp,prime_num)
    A_star_inv = (A_star_adj*A_star_det_inv) % prime_num

    A_star = np.dot(A_star_inv,A_T)
    ans = np.dot(A_star,matrix2) % prime_num   
    
    return np.array(ans)

def Matrix_x_Singular(matrix1,matrix2,k,t,n,prime_num):                                  # 奇異矩陣求解
    # print("\n===== Singular Matrix =====")
    Augmented = Matrix(np.hstack((matrix1,matrix2)), dtype='float64')  
    matrix_rref_temp = np.array(Augmented.rref()[0]).astype(str)
    matrix_rref = np.array(Matrix(Augmented.rref()[0]).applyfunc(nsimplify))  

    for i in range(n):
        for j in range(t+2*k+2):
            num_float = Fraction(matrix_rref_temp[i][j])
            # print(num_float.denominator)
            if num_float.denominator != 1:
                matrix_rref[i][j] = (num_float.numerator*Modinv(num_float.denominator,prime_num))%prime_num
            
    size = matrix_rref.shape
    answer = []
    
    for z in range(2):
        if z == 0:
            ans = np.zeros([t+2*k+1,1])
            num = 0
        else:
            ans = np.ones([t+2*k+1,1])
            num = 1
        
        for i in range(size[0]-1,-1,-1):
            try:           
                row = list(matrix_rref[i]).index(1)
                total = 0
                for j in range(row+1,size[1]-1):
                    total += matrix_rref[i][j] * (num or ans[j][0])   # 通解t設為零要加* ans[j][0]，設為1不用加
                ans[row][0] = matrix_rref[i][size[1]-1] - total
            except:
                pass      
        answer.append(ans)
        
    return answer

def solution_errorposition(a):
    b = (a.roots).real.tolist()
    ans1 = all((round(x,10))%1 == 0 for x in b) # 確認每個元素是不是整數，將所有bool做and,要等於True
    ans2 = any(x > 0 for x in b) # 確認每個元素是不是，將所有bool做and,要等於True
    return (not (ans1 and ans2)), b

def solution_errorposition_singular(a_s):
    b_s = (a_s.roots).real.tolist()
    ans = any((round(x,10))%1 == 0 for x in b_s) # 確認每個元素是不是整數，將所有bool做and,要等於True
    return (not ans), b_s

def Solution(matrix,k,n,prime_num):                                                      # 多項式求解(error)
    solution2 = []
    prime_set = [0,-prime_num,prime_num,-prime_num*2,prime_num*2,-prime_num*3,prime_num*3] 
    if (k==1):
        mod_set = list(product(prime_set,prime_set,prime_set)) # 所有有可能的set(排列組合)
    elif (k==2):
        mod_set = list(product(prime_set,prime_set,prime_set,prime_set)) # 所有有可能的set(排列組合)
    elif (k==3):
        mod_set = list(product(prime_set,prime_set,prime_set,prime_set,prime_set)) # 所有有可能的set(排列組合)
    
    if len(matrix) == 2 : # 奇異矩陣
        solve2, temp_ans2 = [], []            
        for j in range(2):
            solve1 = [1]
            for i in range (2):
                solve1.append(matrix[j][(2-1)-i][0])
            solve2.append(np.poly1d(np.array(solve1)))  # mod
    
        for j in range(2):
            count = 0
            start_flag = True 
            res = []
            while(start_flag and (count < pow(len(prime_set),len(solve2[j])+1))):
                solve3 =  solve2[j] + mod_set[count]
                start_flag, temp_ans = solution_errorposition_singular(solve3) 
                count+=1     
            for i in temp_ans:  # 移除重複的
                if (i not in res) and (round(i,10))%1 == 0 and round(i,10) != 0 and round(i,10) <= n and  round(i,10) > 0: 
                    res.append(round(i,10))  
            temp_ans2.append(res)    
                         
        temp_ans2 = temp_ans2[0]+temp_ans2[1] # 合併
        result = Counter(temp_ans2) # dictionary
        result_list = result.most_common() # dictionary to list
        
        for j in range(len(result_list)):
            if result_list[j][1] > 1: # 找重複的答案
                solution2.append(int(result_list[j][0]%prime_num))  # mod
            else:
                pass    
    else:         # 非奇異矩陣
        solve1 = [1]     
        for i in range (k):
            solve1.append(matrix[(k-1)-i][0])
        solve2 = np.poly1d(np.array(solve1))

        start_flag = True       
        count = 0          
        while(start_flag and (count < pow(len(prime_set),len(solve2)+1))):
            solve3 =  solve2 + mod_set[count]          
            start_flag, solution1 = solution_errorposition(solve3)  
            # print(solution1)
            count+=1
        for i in range (k):
            if solution1[i] > 0 and solution1[i] <= n:
                solution2.append(int(round(solution1[i])))
    return solution2, solve2
