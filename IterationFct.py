import math
import numpy as np


def IterationFCT(inputdata, Mode):
    H, W = inputdata.shape[0:2]
    if Mode == 'METH1':     # 小波变换判焦函数
        inputdata2 = inputdata[0:math.floor(H/4)*4,0:math.floor(W/4)*4]
        Diffimae = inputdata2.astype(np.float64)
        
        row, col = Diffimae.shape[0:2]
        odd = Diffimae[:, 0:col-1:2]
        even = Diffimae[:, 1:col:2]

        #  行变换
        H1 = np.zeros((row,int(col/2)))            # 创建H1 L1矩阵，后面替换矩阵的元素
        L1 = np.ones((row,int(col/2)))
        H1[:,[0]] = odd[:,[0]] - even[:,[0]]
        H1[:, 1:int(col/2)] = odd[:,1:int(col/2)] - (even[:,0:int(col/2-1)] + even[:,1:int(col/2)])/2
        L1[:,0:int(col/2-1)] = even[:,0:int(col/2-1)] + (H1[:,0:int(col/2-1)] + H1[:,1:int(col/2)] +2 )/4
        L1[:,int(col/2-1)] = odd[:,-1] + 1/2*(even[:,-1] + 1)
        trans_first = np.concatenate([L1[:,0:int(col/2)],H1[:,0:int(col/2)]],axis=1)        # 横向拼接
        #  行变换结束

        #  列变换
        odd2 = trans_first[0:row-1:2, :]
        even2 = trans_first[1:row:2, :]
        H11 = np.zeros((int(row/2),col))
        L11 = np.ones((int(row/2),col))
        H11[[0], :] = odd2[[0], :] - even2[[0], :]
        H11[1:int(row/2), :] = odd2[1:int(row/2), :] - (even2[0:int(row/2-1), :] + even2[1:int(row/2), :])/2    # 高频
        L11[0:int(row/2-1), :] = even2[0:int(row/2-1), :] + (H11[0:int(row/2-1), :] + H11[1:int(row/2), :] + 2)/4    # 低频
        L11[int(row/2-1), :] = odd2[-1, :] + 1/2*(even2[-1, :] + 1)
        dim_first = np.concatenate([L11[0:int(row/2), :],H11[0:int(row/2), :]],axis=0)      # 纵向拼接
        #  一维变换结束

        #  第二次---行变换
        odd3 = dim_first[:, 0:int(col/2-1):2]
        even3 = dim_first[:, 1:int(col/2):2]
        H2 = np.zeros((row,int(col/4)))
        L2 = np.ones((row,int(col/4)))
        H2[:, [0]] = odd3[:, [0]] - even3[:, [0]]
        H2[:, 1:int(col/4)] = odd3[:, 1:int(col/4)] - (even3[:, 0:int(col/4-1)] + even3[:, 1:int(col/4)])/2
        L2[:, 0:int(col/4-1)] = even3[:, 0:int(col/4-1)] + (H2[:,0:int(col/4-1)] + H2[:, 1:int(col/4)] + 2)/4
        L2[:, int(col/4-1)] = odd3[:, -1] + 1/2*(even3[:, -1] + 1)
        trans_second = np.concatenate([L2[:,0:int(col/2)],H2[:,0:int(col/2)]],axis=1)       # 横向拼接
        #  二维行变换结束

        #  列变换
        odd4 = trans_second[0:int(row/2-1):2, :]
        even4 = trans_second[1:int(row/2):2, :]
        H22 = np.zeros((int(row/4),int(col/2)))
        L22 = np.ones((int(row/4),int(col/2)))
        H22[[0], :] = odd4[[0], :] - even4[[0], :]
        H22[1:int(row/4), :] = odd4[1:int(row/4), :] - (even4[0:int(row/4-1), :] + even4[1:int(row/4), :])/2
        L22[0:int(row/4-1), :] = even4[0:int(row/4-1), :] + (H22[0:int((row/4-1)), :] + H22[1:int(row/4), :] + 2)/4
        L22[int(row/4-1), :] = odd4[-1, :] + 1/2*(even4[-1, :] + 1)
        #  二维变换结束

        Lx, Ly = L22.shape[:]
        LL2 = L22[:, 0:int(Ly/2)]           # 二维低频
        HL2 = L22[:, int(Ly/2):int(Ly)]     # 二维高频1
        LH2 = H22[:, 0:int(Ly/2)]           # 二维高频2
        HH2 =  H22[:, int(Ly/2):int(Ly)]    # 二维高频3

        ELL2 = np.sum(np.sum(np.power(LL2,2),axis=0))       #先对矩阵列求和，再对矩阵行求和
        EHL2 = np.sum(np.sum(np.power(HL2,2),axis=0))
        ELH2 = np.sum(np.sum(np.power(LH2,2),axis=0))
        EHH2 = np.sum(np.sum(np.power(HH2,2),axis=0))

        q_eval = (EHL2 + ELH2 + EHH2)/ELL2
        return q_eval


    elif Mode == 'METH2':       # 对比度判焦函数
        g = inputdata
        r, c = g.shape[:]
        dg1 = g[1:r-1, 0:c-2] - g[1:r-1, 1:c-1]
        dg2 = g[1:r-1, 2:c] - g[1:r-1,1:c-1]
        dg3 = g[0:r-2, 2:c] - g[1:r-1,1:c-1]
        dg4 = g[2:r, 2:c] - g[1:r-1,1:c-1]

        q_eval = np.sum(np.sum((np.power(dg1,2) + np.power(dg2,2) + np.power(dg3,2) + np.power(dg4,2)),axis=0))/((r-2)*(c-2))
        return q_eval


    elif Mode == 'METH3':       # Tenengrad判焦
        I = inputdata
        M, N = I.shape[:]
        # 利用Sobel算子gx,gy与图像做卷积,提取图像水平方向和垂直方向的梯度值
        GX = 0      # 图像水平方向的梯度值
        GY = 0      # 图像垂直方向的梯度值
        FI = 0      #  变量，暂时存储图像清晰度值
        for x in range(1,M-1):
            for y in range(1,N-1):
                GX = I[x-1,y+1] + 2*I[x,y+1] + I[x+1,y+1] - I[x-1,y-1] - 2*I[x,y-1] -I [x+1,y-1]
                GY = I[x+1,y-1] + 2*I[x+1,y] + I[x+1,y+1] -I[x-1,y-1] -2*I[x-1,y] - I[x-1,y+1]
                SXY = abs(np.sqrt(GX*GX + GY*GY))       # 某一点的梯度值
                FI = FI + SXY*SXY                       # Tenengrad值定义
                
                q_eval = FI/(M*N)
                return q_eval


    else:               # Laplacin判焦函数
        I =inputdata
        M, N = I.shape[:]
        FI = 0
        for x in range(1,M-1):
            for y in range(1,N-1):
                Lxy = -4*I[x,y] + I[x,y+1] + I[x,y-1] + I[x+1,y] + I[x-1,y]
                q_eval = FI + Lxy*Lxy
                return q_eval