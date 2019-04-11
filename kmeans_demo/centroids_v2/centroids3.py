
import numpy as np
# from scipy.cluster.vq import whiten
from whiten import whiten
# from sklearn import svm
import matplotlib.pyplot as plt
from scipy import io
import tensorflow as tf
import conv_utils 

np.random.seed(0)

###########################################

TRAIN_EXAMPLES = 50000
TEST_EXAMPLES = 10000
H = 16
W = 16
C = 96

x_train = np.load('x_train.npy')
# x_test = np.load('x_test.npy')

TRAIN_SHAPE = (TRAIN_EXAMPLES, H, W, C)
TEST_SHAPE = (TEST_EXAMPLES, H, W, C)

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
    # nrows = 768
    # ncols = 16
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

def get_patches(X, patch_shape, patch_start, patch_num):
    PH, PW, PC = patch_shape
    patches = np.zeros(shape=(patch_num, PH, PW, PC))
    for ii in range(patch_num):
        idx = (patch_start + ii) % TRAIN_EXAMPLES
        h = np.random.randint(H - PH)
        w = np.random.randint(W - PW)
        # c = np.random.randint(C - PC) # we are taking all the channels
        patch = X[idx, h:h+PH, w:w+PW, :]
        patches[ii] = patch
    
    # print (np.shape(patches))
        
    return patches

###########################################

def kmeans(X, patch_shape, patch_num, centroid_num, iterations):
    BATCH_SIZE = 1000

    pixel_num = np.prod(patch_shape)
    centroids = np.random.normal(loc=0., scale=0.1, size=(centroid_num, pixel_num))

    for itr in range(iterations):
        summation = np.zeros(shape=(centroid_num, pixel_num))
        counts = np.zeros(shape=(centroid_num))
        c2 = 0.5 * np.sum(centroids ** 2, axis=1, keepdims=True)
    
        for ii in range(0, patch_num, BATCH_SIZE):
            patches = get_patches(X, patch_shape, ii, BATCH_SIZE)
            patches = np.reshape(patches, (BATCH_SIZE, -1))
            batch = patches

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

def kmeans(X, patch_shape, patch_num, centroid_num, iterations):
    BATCH_SIZE = 1000

    pixel_num = np.prod(patch_shape)
    centroids = np.random.normal(loc=0., scale=0.1, size=(centroid_num, pixel_num))

    for itr in range(iterations):
        summation = np.zeros(shape=(centroid_num, pixel_num))
        counts = np.zeros(shape=(centroid_num))
        c2 = 0.5 * np.sum(centroids ** 2, axis=1, keepdims=True)

        for ii in range(0, patch_num, BATCH_SIZE):
            patches = get_patches(X, patch_shape, ii, BATCH_SIZE)
            patches = np.reshape(patches, (BATCH_SIZE, -1))
            batch = patches

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
# which axis do we use ? 
mean = np.mean(x_train, axis=(0, 1, 2), keepdims=True)
std = np.std(x_train, axis=(0, 1, 2), ddof=1, keepdims=True)
scale = std + 1.
x_train = x_train - mean
x_train = x_train / scale

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
                            
                            x1 = x
                            x2 = x + KX

                            y1 = y
                            y2 = y + KY

                            z1 = z
                            z2 = z + KZ

                            if (x2 > H or y2 > W or z2 > C):
                                continue

                            print (x, y, z)
                
                            white = whiten(X=x_train[:, x1:x2, y1:y2, z1:z2], method='zca')
                            white = np.reshape(white, (N, x2-x1, y2-y1, z2-z1))
                            x_train[:, x1:x2, y1:y2, z1:z2] = white

x_train = zca_approx(x_train, (8, 8, 48), (8, 8, 12))

###########################################

centroids = kmeans(X=x_train, patch_shape=(5, 5, 96), patch_num=400000, centroid_num=128, iterations=100)
filters = np.reshape(centroids, (128, 5, 5, 96))
filters = np.transpose(filters, (1, 2, 3, 0))
viz('filters2', filters)
np.save('filters2', {'conv2': filters})

###########################################








