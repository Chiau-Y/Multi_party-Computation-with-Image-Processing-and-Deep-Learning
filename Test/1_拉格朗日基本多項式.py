
n = 6
coefficient = []

for i in range (1, n + 1):
    temp = 1
    ans_temp = 1
    for j in range (1, n + 1):
        if (i != j):
            temp = -j/(i-j)
            ans_temp *= temp
    coefficient.append(ans_temp)
    
print(coefficient)

# # -------------------------------------------------- #
# n = 2
# coefficient = []

# for i in range (-n, n+1):
#     if (i != 0):
#         temp = 1
#         ans_temp = 1
#         for j in range (-n, n+1):
#             if (i != j and j != 0):
#                 temp = -j/(i-j)
#                 ans_temp *= temp
#         coefficient.append(round(ans_temp,4))
    
# print(coefficient)


# b = np.flip(a)
