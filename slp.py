from cProfile import label
from multiprocessing import reduction
import numpy as np
import matplotlib.pyplot as plt
from functions import montage

TRAINING_FILE = 'data_batch_1'
VALIDATION_FILE = 'data_batch_2'
TEST_FILE = 'test_batch'
K = 10
D = 3072
MU = 0
SIGMA = 0.01


def LoadBatch(filename):
    """ Copied from the dataset website"""
    import pickle
    with open('Datasets/'+filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    X = np.transpose(dict[b'data'])
    y = dict[b'labels']
    Y = create_one_hot(y)
    return X, Y, y


def create_one_hot(y):
    Y = np.zeros((K, len(y)))
    for image, label in enumerate(y):
        Y[label][image] = 1
    return Y


def compute_mean(X):
    mean_X = np.zeros(len(X))
    for dimension, row in enumerate(X):
        mean_X[dimension] = np.mean(row)
    return mean_X


def compute_std(X):
    std_X = np.zeros(len(X))
    for dimension, row in enumerate(X):
        std_X[dimension] = np.std(row)
    return std_X


def normalize(X, mean, std):
    X = X-np.vstack(mean)
    X = X / np.vstack(std)
    return X


def deprecated_normalize_X(X):
    norm_X = np.zeros((X.shape[0], X.shape[1]))
    for dimension, row in enumerate(X):
        norm_X[dimension] = np.linalg.norm(row)
    return norm_X


def init_weights(mu, sigma):
    W = np.random.normal(mu, sigma, size=(K, D))
    return W


def init_bias(mu, sigma):
    b = np.random.normal(mu, sigma, size=K)
    return b

# (X: d*n, W:K*d, b:K*1, p:K*n)


def Evaluate_Classifier(X, W, b):
    # s = Wx+b
    # p = SOFTMAX(s)
    # SOFTMAX(s) = exp(s)/(1^T)exp(s)
    P = 0

    return P


# load training, validation, and test data
X_training, Y_training, y_training = LoadBatch(TRAINING_FILE)
X_validation, Y_validation, y_validation = LoadBatch(VALIDATION_FILE)
X_test, Y_test, y_test = LoadBatch(TEST_FILE)

# show example images
# montage(X_training)


# compute mean and standard deviation
mean_training = compute_mean(X_training)
std_training = compute_std(X_training)

# normalize training, validation, and test data with mean and std from train-set
X_norm_train = normalize(X_training, mean_training, std_training)
X_norm_validation = normalize(X_validation, mean_training, std_training)
X_norm_test = normalize(X_test, mean_training, std_training)

# initialize weights and bias
W = init_weights(MU, SIGMA)
b = init_bias(MU, SIGMA)
print(W.shape)
print(b.shape)

# norm_X = deprecated_normalize_X(X_training)
# print(norm_X.shape)
# print(norm_X)
