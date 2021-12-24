#***********************************************************************#
#***********************************************************************#
#   注意在AutoIteration中更改衍射恢复图像的保存路径


import cv2
import numpy as np
import matplotlib.pyplot as plt
from Autolteration import AutoIteration

img = cv2.imread('D:\\Diffraction recovery(Temp)\\475nm.jpg')

print(img.shape[:])

img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img = img.astype(np.float64)
inputdata = img
# backdata = np.mean(np.mean(inputdata, 0))
backdata = 1

Lamda = np.float64(470)

Z = np.arange(((Lamda*(10**(-6)))*1000), 1 ,(Lamda*(10**(-6))*5))     # 衍射恢复初版细胞使用数据
# Z = np.arange(((Lamda*(10**(-6)))*1276), 0.9 ,(Lamda*(10**(-6))*21))

Theta = 90
PixelSize = 2.2  # um
Scale = 4
IterativeTimes = 15
arph = 0
# z_best,Diffimage1,Diffimage2 = AutoIteration(inputdata, backdata, Z, Lamda, Theta, PixelSize, Scale, IterativeTimes, arph)
z_best,Diffimage1 = AutoIteration(inputdata, backdata, Z, Lamda, Theta, PixelSize, Scale, IterativeTimes, arph)

print('Z_best:',z_best)
# cv2.imshow('Diffimae1',Diffimage1/255)      # z_best+Lamda*1/4
# cv2.imshow('Diffimae2',Diffimage2/255)      # z_best+Lamda*3/4
# cv2.waitKey(2)
plt.show()