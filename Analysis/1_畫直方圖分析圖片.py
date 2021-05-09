import cv2
import numpy as np
import matplotlib.pyplot as plt

# 讀取圖檔
img = cv2.imread('DMaster_2019_2021Muti_Party_ComputationErrorCorrectionpartiesresultzImage1to6_r.png',0)

# 計算直方圖每個 bin 的數值
hist = cv2.calcHist([img], [0], None, [256], [0, 256])

# 畫出直方圖
plt.hist(img.ravel(), 256, [0, 256])
# plt.bar(range(1,257), hist)
plt.show()

