import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from DisDiffraction import DisDiffraction
from IterationFct import IterationFCT

def AutoIteration(inputdata, backdata, Z, Lamda, Theta, PixelSize, Scale, IterativeTimes, arph):
    Lamda = Lamda * 10 ** (-6)
    Theta = Theta * math.pi / 180
    InputImage = inputdata.astype(np.float64)
    z_list = []
    q_eval_list = []

    for z in Z:
        Diffimae,Diffimae_equ = DisDiffraction(InputImage, backdata, z, Lamda, Theta, PixelSize, Scale, 0 , 0)
        z_list.append(z)
        q_eval = IterationFCT(Diffimae,'METH1')
        q_eval_list.append(q_eval)

        cv2.imwrite('D:\\Diffraction recovery(Temp)\\data3\\{}{}.jpg'.format(round(z, 4), 'nm'), Diffimae_equ)

    finall_index = q_eval_list.index(max(q_eval_list))
    z_best = z_list[finall_index]    

    # 绘图 并 标记出坐标点
    plt.figure('z & q_eval'),plt.title('z_list & q_eval_list')
    plt.plot(z_list, q_eval_list, z_best,max(q_eval_list),'ko', 0.75, 0.0020078, 'ko')     # k是黑色; o是圆圈
            # plt.plot(x, y, format_string, **kwargs)
            #       format_string: 控制曲线的格式字符串，可选
            #       **kwargs     : 第二组或更多(x,y,format_string)，可画多条曲线

    show_max = '(' + str(round(z_best,4)) + ')'     # round( , 4)  保留4位有效数字
    plt.annotate('(0.7500)', xy=(0.75,0.0020078), xytext=(0.75,0.0020078), weight = 'heavy')
    plt.annotate(show_max, xy=(z_best,max(q_eval_list)), xytext=(z_best,max(q_eval_list)), weight = 'heavy')
            # 该函数用以在图上标注文字
            # plt.annotate(s='str', xy=(x,y), xytext=(l1,l2) , ...)
            #               s     ：为注释文本内容
            #               xy    ：为被注释的坐标点
            #               xytext：为注释文字的坐标位置
            #               textcoords = 'offset points': 相对于被注释点xy的偏移量（单位是点）None:被注释的坐标点xy为参考 (默认值)
            #               weight: 设置字体线型

    Diffimage1 = DisDiffraction(InputImage, backdata, z_best+Lamda*1/4, Lamda, Theta, PixelSize, Scale, IterativeTimes, arph)
    cv2.imshow('Diffimae1',Diffimage1)      # z_best+Lamda*1/4
    
    Diffimage2 = DisDiffraction(InputImage, backdata, z_best+Lamda*3/4, Lamda, Theta, PixelSize, Scale, IterativeTimes, arph)
    cv2.imshow('Diffimae2',Diffimage2)      # z_best+Lamda*3/4
    cv2.waitKey(2)

    return z_best,Diffimage1