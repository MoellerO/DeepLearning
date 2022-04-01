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
LAMBDA = 0.01


def load_batch(filename):
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
    b = np.random.normal(mu, sigma, size=K)[np.newaxis]
    return np.transpose(b)


def evaluate_classifier(X, W, b):
    # (X: d*n, W:K*d, b:K*1, p:K*n)
    batch_size = X.shape[1]
    ones_batch_size = np.ones(batch_size)[np.newaxis]
    # s = Wx+b
    s = np.matmul(W, X) + np.matmul(b, ones_batch_size)

    # P = SOFTMAX(s)
    # SOFTMAX(s) = exp(s)/(1^T)exp(s)
    # denominator = (1^T)exp(s)
    denominator = np.matmul(np.ones(K), np.exp(s))
    P = np.exp(s)/denominator
    return P


def compute_cost(X, Y_one_hot, W, b, Lambda):
    # (X: d*n, Y:k*n ,W:K*d, b:K*1, lamda: scalar)
    D = X.shape[1]
    P = evaluate_classifier(X, W, b)
    Y_t = np.transpose(Y_one_hot)
    l_cross = (-1) * np.matmul(Y_t, np.log(P))
    # J =  1/D * Sum(l_cross(x,y,W,b)+ lambda * Sum(W_{i,j}^2))
    J = 1/D * np.trace(l_cross) + Lambda * np.square(W).sum()
    return J


# load training, validation, and test data
X_training, Y_training, y_training = load_batch(TRAINING_FILE)
X_validation, Y_validation, y_validation = load_batch(VALIDATION_FILE)
X_test, Y_test, y_test = load_batch(TEST_FILE)

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

# Evaluate: softmax(WX+p)
evaluate_classifier(X_norm_train, W, b)
compute_cost(X_norm_train, Y_training, W, b, LAMBDA)
