
import numpy as np
import tensorflow as tf

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
        patch = X[idx, h + PH, w + PW, :]
        patches[ii] = patch
        
    return patches

###########################################

### CHECK SIZING
### IF WE WERE SMART WE WOULD MAKE CHECKS FOR THE SIZE OF THESE THINGS ...

def kmeans(patches, patch_shape, patch_num, centroid_num, iterations):
    BATCH_SIZE = 1000

    pixel_num = np.prod(patch_shape)
    patches = np.reshape(patches, (patch_num, pixel_num))
    centroids = np.random.uniform(low=0., high=0.1, size=(centroid_num, pixel_num))
    
    summation = np.zeros(shape=(centroid_num, pixel_num))
    counts = np.zeros(shape=(centroid_num))
    
    for ii in range(0, patch_num, BATCH_SIZE):
        assert(ii + BATCH_SIZE <= patch_num)
        batch = patches[ii:ii+BATCH_SIZE]
        
        # attempt 1
        # val = np.dot(centroids, batch.T)
        # labels = np.argsort(val)
        # print (np.shape(centroids), np.shape(batch.T))
        # print (np.shape(val), np.shape(labels))
        
        # attempt 2 
        val = np.dot(centroids, batch.T)
        labels = np.argmax(val, axis=0)
        val = np.max(val, axis=0)
        # print (np.shape(centroids), np.shape(batch.T))
        # print (np.shape(val), np.shape(labels))
        # print (labels)
        
        # S = sparse(1:m,labels,1,m,k,m); % labels as indicator matrix
        s = np.zeros(shape=(BATCH_SIZE, centroid_num))
        s[range(BATCH_SIZE), labels] = val
        # print (np.shape(s))
        # s = (1000, 1600)
        # centroids = (1600, 108)
        # summation = (1600, 108)
        
        # summation = summation + S'*X(i:lastIndex,:);
        # counts = counts + sum(S,1)';
        summation = summation + np.dot(s.T, batch)
        counts = counts + np.sum(s, axis=0)
        print (np.shape(counts))
        
    return centroids
    
###########################################

patches = get_patches(X=x_train, patch_shape=(6, 6, 3), patch_num=100000)

centroids = kmeans(patches=patches, patch_shape=(6, 6, 3), patch_num=100000, centroid_num=1600, iterations=1)

print (centroids)


'''
def run_kmeans(x, k, iterations):

    # x2 = sum(X.^2,2);
    # centroids = randn(k,size(X,2))*0.1;%X(randsample(size(X,1), k), :);
    
    num = np.shape(X)
'''





