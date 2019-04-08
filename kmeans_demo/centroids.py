
import numpy as np
# from scipy.cluster.vq import whiten
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


'''
img = x_train[1, :, :, :]
img = np.pad(img, ((5, 5), (5, 5), (0,0)), 'constant', constant_values=255.)
plt.imsave('test.jpg', img, cmap="gray")
assert(False)
'''

'''
x_train1 = io.loadmat('./CIFAR10/data_batch_1.mat')['data']
x_train2 = io.loadmat('./CIFAR10/data_batch_2.mat')['data']
x_train3 = io.loadmat('./CIFAR10/data_batch_3.mat')['data']
x_train4 = io.loadmat('./CIFAR10/data_batch_4.mat')['data']
x_train5 = io.loadmat('./CIFAR10/data_batch_5.mat')['data']
x_train = np.concatenate((x_train1, x_train2, x_train3, x_train4, x_train5), axis=0)
x_train = x_train.astype(float)
x_train = np.reshape(x_train, (TRAIN_SHAPE))
'''
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
    filters = np.transpose(filters, (3, 0, 1, 2))
    
    [nrows, ncols] = factors(fout)
    filters = np.reshape(filters, (nrows, ncols, fh, fw, fin))

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
            
    plt.imsave(name, img)
    

# verfiy this function works by diffing with TF function.
# verify padding works by vizing image. 
def conv2d(inputs, filters, strides, padding):
    N, h, w, c = np.shape(inputs)
    fh, fw, fin, fout = np.shape(filters)
    assert(fin == c)
    pad_h, pad_w = conv_utils.get_pad(padding, np.array([fh, fw]))
    stride_row, stride_col = strides
    # inputs = np.pad(inputs, [[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]])
    # TODO verify this works by vizing a CIFAR10 image.
    inputs = np.pad(inputs, ((0,0), (pad_h, pad_h), (pad_w, pad_w), (0,0)), 'constant')

    output_row = conv_utils.conv_output_length(h, fh, padding, strides[0])
    output_col = conv_utils.conv_output_length(w, fw, padding, strides[1])
    output_shape = (output_row, output_col, fout)
    filter_shape = (output_row * output_col, fh * fw * fin, fout)

    xs = []
    for i in range(output_row):
        for j in range(output_col):
            slice_row = slice(i * stride_row, i * stride_row + fh)
            slice_col = slice(j * stride_col, j * stride_col + fw)
            xs.append(np.reshape(inputs[:, slice_row, slice_col, :], (1, N, fh * fw * fin)))

    x_aggregate = np.concatenate(xs, axis=0)
    # print (np.shape(x_aggregate), np.shape(filters))
    _filters = np.reshape(filters, (fh*fw*fin, fout))
    output = np.matmul(x_aggregate, _filters)
    output = np.reshape(output, (output_row, output_col, N, fout))
    output = np.transpose(output, (2, 0, 1, 3))
    return output

'''
def conv2d(inputs, kernel, strides, output_shape):
    N = np.shape(inputs)[0]

    stride_row, stride_col = strides
    output_row, output_col = output_shape
    kernel_shape = np.shape(kernel)
    feature_dim = kernel_shape[1]
    kernel_size = kernel_shape[2]

    xs = []
    for i in range(output_row):
        for j in range(output_col):
            slice_row = slice(i * stride_row, i * stride_row + kernel_size[0])
            slice_col = slice(j * stride_col, j * stride_col + kernel_size[1])
            xs.append(np.reshape(inputs[:, slice_row, slice_col, :], (1, N, feature_dim)))

    x_aggregate = np.concat(xs, axis=0)
    # output = tf.keras.backend.batch_dot(x_aggregate, kernel)
    output = np.matmul(x_aggregate, kernel)
    output = np.reshape(output, (output_row, output_col, N, filters))
    output = np.transpose(output, (2, 0, 1, 3))
    return output
'''

###########################################

# patches = 400000, 108
# so that menas to me 6*6*3 = 108

# so first we must extract random patches apparently.
# literally looks like they tear through CIFAR10 and take out the patches
# so i guess we do the same thing

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

### CHECK SIZING
### IF WE WERE SMART WE WOULD MAKE CHECKS FOR THE SIZE OF THESE THINGS ...

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

patches = get_patches(X=x_train, patch_shape=(6, 6, 3), patch_num=400000)
patches = np.reshape(patches, (400000, 6*6*3))

# patches = io.loadmat('patches.mat')['patches']

mean = np.mean(patches, axis=1, keepdims=True)
patches = patches - mean
# sqrt(var(patches,[],2)+10)
var = np.var(patches, axis=1, ddof=1, keepdims=True)
scale = np.sqrt(var + 10.)
patches = patches / scale

print (np.std(patches, ddof=1))
print ('mean', np.std(mean, ddof=1), np.shape(mean))
print ('var', np.std(var, ddof=1), np.shape(var))

cov = np.cov(patches.T)
mean = np.mean(patches, axis=0, keepdims=True)
[D, V] = np.linalg.eig(cov)

# P = V * diag(sqrt(1./(diag(D) + 0.1))) * V'; 
a = np.diag(np.sqrt(1./(D + 0.1)))
b = np.dot(V, a)
c = np.dot(b, V.T)
P = c
patches = np.dot(patches - mean, P)

print (np.std(patches))

# patches = whiten(X=patches, method='zca')
# patches = io.loadmat('patches.mat')['patches']

###########################################

centroids = kmeans(patches=patches, patch_shape=(6, 6, 3), patch_num=400000, centroid_num=128, iterations=10)
filters = np.reshape(centroids, (128, 6, 6, 3))
filters = np.transpose(filters, (1, 2, 3, 0))

features = np.zeros(shape=(TRAIN_EXAMPLES, H, W, 128))
for ii in range(0, TRAIN_EXAMPLES, 100):
    print (ii)
    
    start = ii
    stop = ii + 100
    _features = conv2d(inputs=x_train[start:stop], filters=filters, strides=[1, 1], padding='same')
    features[start:stop] = _features

print (np.shape(features))
np.save('features', features)

'''
# if we dump it and do show_centroids in matlab it works fine.
dic = {'centroids': centroids}
io.savemat('centroids.mat', dic)
'''
'''
centroids = np.reshape(centroids, (1600, 6, 6, 3))
centroids = np.transpose(centroids, (1, 2, 3, 0))
viz('centroids', centroids)
'''
'''
clf = svm.SVC(gamma='scale')
clf.fit(X, y) 
'''









