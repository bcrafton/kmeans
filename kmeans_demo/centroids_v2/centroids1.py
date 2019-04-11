
import numpy as np
# from scipy.cluster.vq import whiten
from sklearn.decomposition import PCA
from whiten import whiten
from sklearn import svm
import matplotlib.pyplot as plt
from scipy import io
import tensorflow as tf
import conv_utils 

np.random.seed(0)

###########################################
TRAIN_EXAMPLES = 50000
TEST_EXAMPLES = 10000
H = 32
W = 32
C = 3

TRAIN_SHAPE = (TRAIN_EXAMPLES, H, W, C)
TEST_SHAPE = (TEST_EXAMPLES, H, W, C)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

if np.shape(x_train) != TRAIN_SHAPE:
    x_train = np.transpose(x_train, (0, 2, 3, 1))
    
if np.shape(x_test) != TEST_SHAPE:
    x_test = np.transpose(x_test, (0, 2, 3, 1))

###########################################

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
    # nrows = fout
    # ncols = fin
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

###########################################

def kmeans(patches, patch_shape, patch_num, centroid_num, iterations):
    BATCH_SIZE = 1000

    pixel_num = np.prod(patch_shape)
    patches = np.reshape(patches, (patch_num, pixel_num))
    centroids = np.random.normal(loc=0., scale=0.1, size=(centroid_num, pixel_num))

    for itr in range(iterations):
        summation = np.zeros(shape=(centroid_num, pixel_num))
        counts = np.zeros(shape=(centroid_num))
        c2 = 0.5 * np.sum(centroids ** 2, axis=1, keepdims=True)
    
        for ii in range(0, patch_num, BATCH_SIZE):
            assert(ii + BATCH_SIZE <= patch_num)
            batch = patches[ii:ii+BATCH_SIZE]

            val = np.dot(centroids, batch.T) - c2
            labels = np.argmax(val, axis=0)
            val = np.max(val, axis=0)
            
            s = np.zeros(shape=(BATCH_SIZE, centroid_num))
            s[range(BATCH_SIZE), labels] = 1
            
            summation = summation + np.dot(s.T, batch)
            counts = counts + np.sum(s, axis=0)

        print (np.std(centroids))

        idx = np.where(counts != 0)
        nidx = np.where(counts == 0)
        _counts = 1. * np.reshape(counts, (-1, 1))
        centroids[idx] = summation[idx] / _counts[idx]
        centroids[nidx] = 0.
        
    return centroids
    
###########################################

x_train = np.reshape(x_train, (TRAIN_EXAMPLES, H, W, C))
mean = np.mean(x_train, axis=(0,1,2), keepdims=True)
std = np.std(x_train, axis=(0,1,2), ddof=1, keepdims=True)
scale = std + 1.
x_train = x_train - mean
x_train = x_train / scale

x_train = whiten(X=x_train, method='zca')
x_train = np.reshape(x_train, (TRAIN_EXAMPLES, H, W, C))

###########################################

filters = np.zeros(shape=(96, 5, 5, 3))
for ii in range(6):
    patches = get_patches(X=x_train, patch_shape=(5, 5, 3), patch_num=400000)
    patches = np.reshape(patches, (400000, 5*5*3))

    centroids = kmeans(patches=patches, patch_shape=(5, 5, 3), patch_num=400000, centroid_num=16, iterations=25)
    filters[ii*16:(ii+1)*16] = np.reshape(centroids, (16, 5, 5, 3))

'''
patches = get_patches(X=x_train, patch_shape=(5, 5, 3), patch_num=400000)
patches = np.reshape(patches, (400000, 5*5*3))
centroids = kmeans(patches=patches, patch_shape=(5, 5, 3), patch_num=400000, centroid_num=96, iterations=25)
filters = np.reshape(centroids, (96, 5, 5, 3))
'''

filters = np.transpose(filters, (1, 2, 3, 0))
viz('filters1', filters)
np.save('filters1', {'conv1': filters})










