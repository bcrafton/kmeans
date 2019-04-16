
import numpy as np
from sklearn.decomposition import PCA
from whiten import whiten
from sklearn import svm
import matplotlib.pyplot as plt
from scipy import io
import tensorflow as tf
import conv_utils 

np.random.seed(0)

###########################################
PATCH_NUM = 500000
TRAIN_EXAMPLES = 50000
TEST_EXAMPLES = 10000

H = 32
W = 32
C = 3

FH = 7
FW = 7
FC = 3

TRAIN_SHAPE = (TRAIN_EXAMPLES, H, W, C)
TEST_SHAPE = (TEST_EXAMPLES, H, W, C)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

if np.shape(x_train) != TRAIN_SHAPE:
    x_train = np.transpose(x_train, (0, 2, 3, 1))
    
if np.shape(x_test) != TEST_SHAPE:
    x_test = np.transpose(x_test, (0, 2, 3, 1))

###########################################

def get_patches(X, patch_shape, patch_num):
    PH, PW, PC = patch_shape
    patches = np.zeros(shape=(patch_num, PH, PW, PC))
    for ii in range(patch_num):
        idx = ii % TRAIN_EXAMPLES
        h = np.random.randint(H - PH)
        w = np.random.randint(W - PW)
        # c = np.random.randint(C - PC) # we are taking all the channels
        patch = X[idx, h:h+PH, w:w+PW, :]
        patches[ii] = patch
    
    return patches
    
def factors(x):
    l = [] 
    for i in range(1, x + 1):
        if x % i == 0:
            l.append(i)
    
    mid = int(len(l) / 2)
    
    if (len(l) % 2 == 1):
        return [l[mid], l[mid]]
    else:
        return l[mid-1:mid+1]
    
def viz(name, filters):
    fh, fw, fin, fout = np.shape(filters)
    filters = filters.T
    assert(np.shape(filters) == (fout, fin, fw, fh))
    [nrows, ncols] = factors(fin * fout)
    filters = np.reshape(filters, (nrows, ncols, fw, fh))

    for ii in range(nrows):
        for jj in range(ncols):
            if jj == 0:
                row = filters[ii][jj]
            else:
                row = np.concatenate((row, filters[ii][jj]), axis=1)
                
        if ii == 0:
            img = row
        else:
            img = np.concatenate((img, row), axis=0)

    plt.imsave(name, img, cmap='gray')
###########################################
patches = get_patches(X=x_train, patch_shape=(FH, FW, FC), patch_num=PATCH_NUM)
patches = np.reshape(patches, (PATCH_NUM, FH * FW * FC))

mean = np.mean(patches, axis=1, keepdims=True)
assert(np.shape(mean) == (PATCH_NUM, 1))
std = np.std(patches, axis=1, ddof=1, keepdims=True)
scale = std + 1.
patches = patches - mean
patches = patches / scale

[D, V] = np.linalg.eig(patches.T @ patches)
# print (np.shape(D), np.shape(V))

filters = V
filters = np.reshape(filters, (FH * FW * FC, FH, FW, FC))
filters = np.transpose(filters, (1, 2, 3, 0))

# print (V.T @ V) # = Identity.
# print (V @ V.T) # = Identity.

viz('filters.png', filters)

















