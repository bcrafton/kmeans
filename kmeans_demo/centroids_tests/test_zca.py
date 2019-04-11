
import numpy as np
from whiten import whiten

x_train = np.load('x_train.npy')

'''
x_step = 4
y_step = 4
z_step = 32
for x in range(0, 16, 2):
    for y in range(0, 16, 2):
        for z in range(0, 128, 16):
            
            print (x, y, z)
            
            x1 = x
            x2 = min(x + x_step, 16)
            
            y1 = y
            y2 = min(y + y_step, 16)

            z1 = z
            z2 = min(z + y_step, 128)
            
            white = whiten(X=x_train[:, x1:x2, y1:y2, z1:z2], method='zca')
            white = np.reshape(white, (50000, x2-x1, y2-y1, z2-z1))
            x_train[:, x1:x2, y1:y2, z1:z2] = white
'''

ksize = (4, 4, 32)
ssize = (2, 2, 16)

def zca_approx(data, ksize, ssize):
    N, H, W, C = np.shape(data)
    KX, KY, KZ = ksize
    SX, SY, SZ = ssize

    for sx in range(0, KX, SX):
        for sy in range(0, KY, SY):
            for sz in range(0, KZ, SZ):
            
                for x in range(sx, H+sx, KX):
                    for y in range(sy, W+sy, KY):
                        for z in range(sz, C+sz, KZ):
                            
                            print (x, y, z)

                            x1 = x
                            x2 = min(x + KX, H)
                            
                            y1 = y
                            y2 = min(y + KY, W)

                            z1 = z
                            z2 = min(z + KZ, C)
                
                            white = whiten(X=x_train[:, x1:x2, y1:y2, z1:z2], method='zca')
                            white = np.reshape(white, (N, x2-x1, y2-y1, z2-z1))
                            x_train[:, x1:x2, y1:y2, z1:z2] = white
                            
zca_approx(x_train, ksize, ssize)

                            
