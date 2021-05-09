import cv2
import time
import psutil
import random
import functools
import numpy as np
from sympy import * 
from threading import Timer
from itertools import product
from collections import Counter
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial as P

start_time = time.time()
# ------------------------------------Subroutine------------------------------------
def hello():
    global cpu
    
    cpu.append(psutil.cpu_percent(interval=0.4))

    
    # print ("hello, world")
    
class RepeatingTimer(Timer): 
    def run(self):
        while not self.finished.is_set():
            self.function(*self.args, **self.kwargs)
            self.finished.wait(self.interval)

def Check_n(n,t):                                                            # 設定party個數                   
    if n < 2*t or n <= 0:
        print("You need more parties")
        check = 1
    else:
        check = 0
    return check

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
    
def Matrix_x(matrix1,matrix2):                                             # 矩陣求解
    # print("\n===== Square and Non-Singular Matrix=====")
    A_inv_temp = np.linalg.inv(matrix1)
    A_det_temp = (round(np.linalg.det(matrix1)))%prime_num
    A_adj = np.around((A_inv_temp*round(np.linalg.det(matrix1))))
    
    A_det_inv = Modinv(A_det_temp,prime_num)
    A_inv = (A_adj*A_det_inv)%prime_num
    ans = A_inv.dot(matrix2)
    return ans % prime_num

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

def Matrix_x_Singular(matrix1,matrix2):                                  # 奇異矩陣求解
    # print("\n===== Singular Matrix =====")
    Augmented = Matrix(np.hstack((matrix1,matrix2)))
    matrix_rref = np.array(Augmented.rref()[0])
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
    ans1 = all(x%1 == 0 for x in b) # 確認每個元素是不是整數，將所有bool做and,要等於True
    ans2 = all(x > 0 for x in b) # 確認每個元素是不是，將所有bool做and,要等於True
    return (not (ans1 and ans2)), b

def solution_errorposition_singular(a_s):
    b_s = (a_s.roots).real.tolist()
    ans = any((round(x,10))%1 == 0 for x in b_s) # 確認每個元素是不是整數，將所有bool做and,要等於True
    return (not ans), b_s

def Solution(matrix):                                                      # 多項式求解(error)
    prime_set = [0,-prime_num,prime_num,-prime_num*2,prime_num*2] 
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
                if (i not in res) and (round(i,10))%1 == 0 and round(i,10) != 0: 
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
            count+=1
        
        for i in range (k):
            if solution1[i] > 0 and solution1[i] <= n:
                solution2.append(int(round(solution1[i])))
    return solution2, solve2

def Orinignal_fun(matrix1,matrix4): # matrix_x,fun_e
    answer, solve3 = [], []
    variable = [a,b,c,d]   
    if len(matrix1) == 2 : # 奇異矩陣
        for i in range (t+k+1):                              # h(x)
            solve3.append(matrix1[0][((t+2*k+1)-1)-i][0]) 
        fun = matrix4[0]
    else:
        for i in range (t+k+1):                              # h(x)
            solve3.append(matrix1[((t+2*k+1)-1)-i][0]) 
        fun = matrix4
    h = np.poly1d(solve3)

    if (t==1):
        coefficient = [a,b]
    elif (t==2):
        coefficient = [a,b,c]
    elif (t==3):
        coefficient = [a,b,c,d]

    for i in range (t+1):
        unknown = np.poly1d(coefficient)  # 原式
        h_Prime = unknown*fun 
        ans = solve(h_Prime[len(h_Prime)-i]-h[len(h)-i],variable[i])
        try:
            answer.append(ans[0]%prime_num)
            coefficient[i] = answer[i]
        except: 
            answer.append(0)  # 原式就是0+0x，此時b會算不出來，因此用except
    solve5 = np.poly1d(answer)

    return solve5

# ------------------------------------Main program------------------------------------
cpu = []
t_cpu = RepeatingTimer(1, hello)
t_cpu.start()

time.sleep(3)

check_n, Flag_Pseudo = True, False
matrix_A1, matrix_A2, matrix_b1, matrix_x1 = [], [], [], []
answer, solve1, solution2, solve3, solution3 = [], [], [], [], []
error_party_check = {}
t, n, prime_num = 1, 6, 33292801                                # 次方

a = Symbol('a')    # 解原式
b = Symbol('b')
c = Symbol('c')
d = Symbol('d')

print("\n-------------------Start-------------------")     
   
party1 = np.load('./parties/test/Party1.npy') 
party2 = np.load('./parties/test/Party2_original.npy') 
party3 = np.load('./parties/test/Party3.npy') 
party4 = np.load('./parties/test/Party4.npy') 
party5 = np.load('./parties/test/Party5.npy') 
party6 = np.load('./parties/test/Party6.npy')  

# plt.figure(1)
# plt.imshow(party1, cmap='gray')
# plt.axis('off') # 不顯示座標軸 
# # plt.savefig("./parties/result/party1_z.png",bbox_inches='tight',pad_inches = 0) 
# plt.figure(2)
# plt.imshow(party2, cmap='gray')
# plt.axis('off') # 不顯示座標軸
# # plt.savefig("./parties/result/party2_z.png",bbox_inches='tight',pad_inches = 0) 
# plt.figure(3)
# plt.imshow(party3, cmap='gray')
# plt.axis('off') # 不顯示座標軸  
# # plt.savefig("./parties/result/party3_z.png",bbox_inches='tight',pad_inches = 0) 
# plt.figure(4)
# plt.imshow(party4, cmap='gray')
# plt.axis('off') # 不顯示座標軸 
# # plt.savefig("./parties/result/party4_z.png",bbox_inches='tight',pad_inches = 0) 
# plt.figure(5)
# plt.imshow(party5, cmap='gray')
# plt.axis('off') # 不顯示座標軸
# # plt.savefig("./parties/result/party5_z.png",bbox_inches='tight',pad_inches = 0) 
# plt.figure(6)
# plt.imshow(party6, cmap='gray')
# plt.axis('off') # 不顯示座標軸 
# # plt.savefig("./parties/result/party6_z.png",bbox_inches='tight',pad_inches = 0) 

img_size = np.shape(party1)
party_new = [party1,party2,party3,party4,party5,party6]

print("Error Correction\n")

while (check_n):
    check_n = Check_n(n,t)                                                  # party數(n)
print("There are {} parties.".format(n))                                  

k = int((n-t-1)/2)
print("\nThe maximum of the number of error : ",k)                            # 錯誤數量最大值(k_max)  

for i in range (img_size[0]): 
    if i%100 == 0 and i != 0:
        print(" {}%".format(round((i/img_size[1])*100)))         
    if i%5 == 0:
        print('=', end = '')
    for j in range(img_size[1]): 
        shares = []
        shares.append(int(party1[i][j]))
        shares.append(int(party2[i][j]))
        shares.append(int(party3[i][j]))
        shares.append(int(party4[i][j]))
        shares.append(int(party5[i][j]))
        shares.append(int(party6[i][j]))
    
        matrix_A = Matrix_A(np.array(shares),k,t,n)                                          # 矩陣A
        
        matrix_b = Matrix_b(np.array(shares),k,n)                                            # 矩陣b
        
        if t+2*k+1 == n:
            if round(np.linalg.det(matrix_A)) == 0:    
                matrix_x = Matrix_x_Singular(matrix_A,matrix_b)     
            else:
                matrix_x = Matrix_x(matrix_A,matrix_b)  
        else:
            matrix_x = Matrix_x_Pseudo(matrix_A,matrix_b)                           # 解答(矩陣x)
            Flag_Pseudo = True
            
        error_party_temp, fun_e = Solution(matrix_x)                        # 錯誤的party   
        error_party_temp.sort()
        error_dict = Counter(error_party_temp)
        for party, count in error_dict.items():
            if error_party_check.get(party) == None:
                error_party_check[party] = count
            else:
                error_party_check[party] += 1
                
        if len(error_party_temp) == 0:            
            pass
        else:
            fun_org = Orinignal_fun(matrix_x,fun_e)
            for p in error_party_temp:
                party_new[p-1][i][j] =  int(round(fun_org(p))) % prime_num
 
error_party = []    
print("\n")
for party, num in error_party_check.items():
    print("{} : {} ".format(party,num))
    if num >=img_size[0]*img_size[1]*0.4:
        error_party.append(party)

if (not error_party) or (len(error_party) == 1 and error_party[0] == 0):
    print("\nThere is no error in shares.\n")
else:
    error_party.sort()
    print("\nThe error party is : ", error_party)
    
# plt.figure(11)
# plt.imshow(party_new[0], cmap='gray')
# plt.axis('off') # 不顯示座標軸 
# plt.figure(12)
# plt.imshow(party_new[1], cmap='gray')
# plt.axis('off') # 不顯示座標軸
# np.save('Party2_r_z', party_new[1])
# # plt.savefig("./parties/result/party2_r_z.png",bbox_inches='tight',pad_inches = 0) 
# plt.figure(13)
# plt.imshow(party_new[2], cmap='gray')
# plt.axis('off') # 不顯示座標軸  
# plt.figure(14)
# plt.imshow(party_new[3], cmap='gray')
# plt.axis('off') # 不顯示座標軸 
# plt.figure(15)
# plt.imshow(party_new[4], cmap='gray')
# plt.axis('off') # 不顯示座標軸
# plt.figure(16)
# plt.imshow(party_new[5], cmap='gray')
# plt.axis('off') # 不顯示座標軸        
# np.save('Party6_r_z', party_new[5])
# # plt.savefig("./parties/result/party6_r_z.png",bbox_inches='tight',pad_inches = 0) 
end_time = time.time()   

print("")        
print("Time : ",round(end_time-start_time, 2),"sec")    
print("\n-------------------END-------------------\n")

time.sleep(3)
t_cpu.cancel() 

plt.figure(20)
plt.plot(cpu, label='CPU')
plt.ylabel('CPU%')
plt.ylim(0,100)
plt.show()

