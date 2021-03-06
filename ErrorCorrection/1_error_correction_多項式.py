from numpy.polynomial import Polynomial as P
from sympy import * 
import numpy as np
import random


# ------------------------------------Subroutine------------------------------------
def Check_n(n,t):                                                            # 設定party個數                   
    if n < 2*t or n <= 0:
        print("You need more parties")
        check = 1
    else:
        check = 0
    return check

def Fun_sender(t):                                                           # 創造多項式
    a = -1
    for i in range (t):
        # fun_s.append(random.randint(-10,10))
        fun_s.append(-1)
    return fun_s

def Share(n,t,f_coeff,s):                                                    # 發送shares
    for i in range (n):
        num = 0
        for j in range (t):
            power = 1
            for z in range (j+1):
                power *= (i+1)
            num += f_coeff[j]*power
        share.append(num+s)
    return share

def Error(num_e,shares):                                                     # shares被竄改
    error = shares
    while (num_e):
        position = int(input("The position of the error of the party : "))
        error[position-1] = int(input("The error is : "))
        num_e -= 1
    return error

def Matrix_A(shares_err,k,t,n):                                              # A矩陣
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

def Matrix_b(shares_err,k,n):                                                # b矩陣
    for i in range (n):
        power = 1
        for j in range (k):
            power *= (i+1)
        matrix_b1.append(share[i]*power*(-1))      
    return np.array(matrix_b1).reshape(n,1)
    
def Matrix_x(matrix1,matrix2):                                             # 矩陣求解
    print("===== Square and Non-Singular Matrix=====")
    Augmented = Matrix(np.hstack((matrix1,matrix2)))
    matrix_rref = np.array(Augmented.rref()[0])
    print("rref : \n",matrix_rref)
    
    A_inv = np.linalg.inv(matrix1)
    ans = A_inv.dot(matrix2)
    return ans

def Matrix_x_Pseudo(matrix1,matrix2):                                      # 非方正矩陣求解
    print("===== Non-square Matrix =====")
    A_inv = np.linalg.pinv(np.mat(matrix1))
    ans = np.dot(A_inv,matrix2)   
    return ans

def Matrix_x_Singular(n,matrix1,matrix2):                                  # 奇異矩陣求解
    print("===== Singular Matrix =====")
    Augmented = Matrix(np.hstack((matrix1,matrix2)))
    matrix_rref = np.array(Augmented.rref()[0])
    # print("rref : \n",matrix_rref)
    count = 0
    for i in range (n):        
        if max(matrix_rref[n-(i+1)]) == 0:                                   # 判斷有幾列整列皆是0
            count += 1 
    for i in range (n):  
        position_2 = -1    
        sum, p = 0, 1
        for j in range (t+2*k+1): 
            if position_2 == -1:                                             # 判斷有沒有找到第一個1在哪個位置
                if matrix_rref[n-(i+1)][j] == 1:                             # 判斷1在哪個位置
                    position_2 = j                                           # 要解的未知數的位置
                    if position_2 == (t+2*k):
                        answer.append(matrix_rref[n-(i+1)][j+1])
                        for z in range (count):                              # 通解的未知數全部設0，0的個數count決定
                            answer.append(0)  
                        count = 0      
            else:               
                sum += -1*matrix_rref[n-(i+1)][j]*answer[(t+2*k+1)-(j+1)]    # (j+1)+(x+1)-1 = t+2*k+1，從頭或從尾開始算
                p += 1
        if count == 0 and position_2 != (t+2*k):
            sum += matrix_rref[n-(i+1)][t+2*k+1]
            answer.append(sum)  
    return np.array(answer).reshape(t+2*k+1,1)

def Reshape(matrix):                                                     # 重新排列矩陣
    for i in range (t+2*k+1):
        matrix_x1.append(matrix[t+2*k+1-(i+1)][0])
    return np.array(matrix_x1).reshape(t+2*k+1,1)   

def Solution(matrix):                                                      # 多項式求解(error)
    solve1.append(1)
    for i in range (k):
        solve1.append(matrix[(k-1)-i][0])
    solve2 = np.poly1d(np.array(solve1))
    solution1 = solve2.roots
    for i in range (k):
        if solution1[i] > 0 and solution1[i] <= n:
            solution2.append(int(round(solution1[i])))
    return solution2, solve1

def Orinignal_fun(matrix1,matrix2,matrix3,matrix4):
    for i in range (t+k+1):
        solve3.append(matrix_x[((t+2*k+1)-1)-i][0]) 
    solve4 = list(P(np.array(solve3))//P(np.array(matrix4)))
    solve5 = np.poly1d(solve4)
    for i in range (num_e):
        matrix3[matrix2[i]-1] = int(round(solve5(matrix2[i])))
    return matrix3, solve5

# ------------------------------------Main program------------------------------------
check_n = 1
fun_s, share, error  = [], [], []
matrix_A1, matrix_A2, matrix_b1, matrix_x1 = [], [], [], []
answer, solve1, solution2, solve3, solution3 = [], [], [], [], []

print("\n-------------------Start-------------------")                

s = int(input("Secret : "))                                                 # 秘密(s0) 
t = int(input("The degree of f(x) : "))                                     # 次方(t)

while (check_n):
    n = int(input("The number of parties (n > 2t) : "))
    check_n = Check_n(n,t)                                                  # party數(n)

k = int((n-t-1)/2)
print("The maximum of the number of error : ",k)                            # 錯誤數量最大值(k_max)          

f_coeff = Fun_sender(t)                                                     # 多項式係數
print("The Coefficient of f(x) = ", f_coeff)

shares = Share(n,t,f_coeff,s)                                               # shares
print("The shares are : ",shares)

num_e = int(input("\nHow many errors happen : "))                           # 錯誤的數量
shares_err = Error(num_e,shares)                                            # 更改過的shares
print("\nThe shares are redefined : ",shares_err)

matrix_A = Matrix_A(shares_err,k,t,n)                                       # 矩陣A
print("\nMatrix A : \n",matrix_A)

matrix_b = Matrix_b(shares_err,k,n)                                         # 矩陣b
print("Matrix b : \n",matrix_b)

if t+2*k+1 == n:
    if np.linalg.det(matrix_A) < 1e-06:    
        matrix_x_1 = Matrix_x_Singular(n,matrix_A,matrix_b)   
        matrix_x = Reshape(matrix_x_1)
    else:
        matrix_x = Matrix_x(matrix_A,matrix_b)  
else:
    matrix_x = Matrix_x_Pseudo(matrix_A,matrix_b)                           # 解答(矩陣x)
print("Matrix x : \n",matrix_x)

error_party, fun_e = Solution(matrix_x)                                     # 錯誤的party
error_party.sort()
if len(error_party) == 0:
    print("\nThere is no error in shares.")
    f_coeff.append(s) 
    print("The f(x) is : ",np.poly1d(np.array(f_coeff)))
else:
    print("\nThe error party is : ", error_party)
    shares_corret, fun_org = Orinignal_fun(matrix_x,error_party,shares_err,fun_e)
    print("\nThe correct shares are : ",shares_corret)
    print("The f(x) is : ",fun_org)

print("\n-------------------END-------------------\n")