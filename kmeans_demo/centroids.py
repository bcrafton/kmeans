
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

patches = get_patches(x_train, (6, 6, 3), 100)
print (np.shape(patches))

'''
def run_kmeans(x, k, iterations):

    # x2 = sum(X.^2,2);
    # centroids = randn(k,size(X,2))*0.1;%X(randsample(size(X,1), k), :);
    
    num = np.shape(X)
'''
