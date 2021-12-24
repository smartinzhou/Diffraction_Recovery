import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

def DisDiffraction(inputdata, backdata, Z, Lamda, Theta, PixelSize, Scale, IterativeTimes, arph):
        r, c = inputdata.shape[0:2]
        r = Scale * r
        c = Scale * c
        pixel1_size = PixelSize * np.sin(Theta)
        I1 = cv2.resize(inputdata/backdata,(c, r),interpolation=cv2.INTER_CUBIC)
        
        Lox = c*(pixel1_size*(10**(-3))/Scale)
        Loy = r*(pixel1_size*(10**(-3))/Scale)

        fx = np.linspace(-c/2/Lox, c/2/Lox, c)
        fy = np.linspace(-r/2/Loy, r/2/Loy, r)
        [fx, fy] = np.meshgrid(fx, fy)

        k = 2*(math.pi)/Lamda
        Hz_p = np.exp( (1j*k*Z) * (np.sqrt(1-np.power(Lamda*fx, 2) - np.power(Lamda*fy, 2))) )
        Hz_n = np.exp( (-1j*k*Z) * (np.sqrt(1-np.power(Lamda*fx, 2) - np.power(Lamda*fy, 2))) )


        test = np.sqrt(1-np.power(Lamda*fx, 2) - np.power(Lamda*fy, 2))

        Seat = np.argwhere((np.power(Lamda*fx, 2)+np.power(Lamda*fy, 2)) > 1)           # np.argwhere():查找按元素分组的非零数组元素的索引
        for i in range(len(Seat)):
                Hz_n.itemset(((Seat[i])[0],(Seat[i])[1]), 0)

        U1_abs = abs(np.sqrt(I1))
        Ainfft = np.fft.fftshift(np.fft.fft2(U1_abs))
        Aoutfft = Ainfft * (Hz_n)
        U0_amp = np.fft.ifft2(np.fft.ifftshift(Aoutfft))
        U0_amp_angle = np.angle(U0_amp)
        U0_amp_abs = abs(U0_amp)

        for i in range(IterativeTimes):
                U0_amp_abs = -np.log(U0_amp_abs)
                
                Seat1 = np.argwhere(U0_amp_abs < 0)
                for i in range(len(Seat1)):
                        U0_amp_abs.itemset(((Seat1[i])[0],(Seat1[i])[1]), 0)
                
                U0_amp_abs = np.exp(- U0_amp_abs)
                U0_new = U0_amp_abs * (np.exp(1j*U0_amp_angle))

                FFTU0_call = np.fft.fftshift(np.fft.fft2(U0_new))
                U11_amp = np.fft.ifft2(np.fft.ifftshift(FFTU0_call * Hz_p))
                Phase_retrival = np.angle(U11_amp)
                U11_cal = (arph * abs(U11_amp) + (1-arph) * abs(U1_abs)) * (np.exp(1j*Phase_retrival))
                FFTU1_cal = np.fft.fftshift(np.fft.fft2(U11_cal))
                U0_amp = np.fft.ifft2(np.fft.ifftshift(FFTU1_cal * Hz_n))
                U0_amp_angle = np.angle(U0_amp)
                U0_amp_abs = abs(U0_amp)

                plt.figure(132), plt.imshow( ((np.power(abs(U0_amp - 1), 2)) * backdata), cmap='gray')          # abs--幅值
                plt.figure(133), plt.imshow((np.angle(- U0_amp)), cmap='gray')          # angle--相位
                
        Diffimae = np.power(abs(U0_amp-1), 2) * backdata
        Diffimae = Diffimae.astype(np.uint8)
# ***************************************直方图均衡化******************************************** #       
        Diffimae_equ = cv2.equalizeHist(Diffimae)
        cv2.imshow('Diffimae_temp',Diffimae)
        cv2.waitKey(2)

        return Diffimae,Diffimae_equ