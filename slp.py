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
LAMBDA = 0
LEARNING_RATE = 0.1
N_BATCH = 100
N_EPOCHS = 40

# lambda 0, eta 0.1:
#   Epoch: 38
#   Loss: 5.863541698629551
#   Accuracy: 0.3688
#   Epoch: 40
#   Loss: 3.391550342234606
#   Accuracy: 0.481

# lambda 0, eta 0.001:
#   Epoch: 40
#   Loss: 1.6128101071014034
#   Accuracy: 0.4555

# lambda 0.1, eta 0.001:
#   Epoch: 40
#   Loss: 1.6462751621219471
#   Accuracy: 0.4471

# lambda 1, eta 0.001:
#   Epoch: 40
#   Loss: 1.8025967352797148
#   Accuracy: 0.3987


np.random.seed(0)

# From canvas page


def montage(W):
    """ Display the image for each label in W """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 5)
    for i in range(2):
        for j in range(5):
            im = W[i*5+j, :].reshape(32, 32, 3, order='F')
            sim = (im-np.min(im[:]))/(np.max(im[:])-np.min(im[:]))
            sim = sim.transpose(1, 0, 2)
            ax[i][j].imshow(sim, interpolation='nearest')
            ax[i][j].set_title("y="+str(5*i+j))
            ax[i][j].axis('off')
    plt.show()


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


def compute_accuracy(X, y, W, b):
    P = evaluate_classifier(X, W, b)
    predicted_labels = np.argmax(P, axis=0)
    number_correct_predictions = np.sum(predicted_labels == y)
    accuracy = number_correct_predictions / P.shape[1]
    return accuracy


def compute_gradients(X, Y, P, W, Lambda):
    # (X: d*n, Y:k*n, P:K*N, W:K*d,  lamda: scalar)
    n = X.shape[1]
    G_batch = -(Y-P)
    G_W = 1/n * np.matmul(G_batch, np.transpose(X)) + 2*Lambda*W
    ones_b = np.transpose(np.ones(n)[np.newaxis])
    G_b = 1/n * np.matmul(G_batch, ones_b)
    return G_W, G_b

# From canvas page


def ComputeGradsNum(X, Y, P, W, b, lamda, h):
    """ Converted from matlab code """
    no = W.shape[0]
    d = X.shape[0]

    grad_W = np.zeros(W.shape)
    grad_b = np.zeros((no, 1))

    c = compute_cost(X, Y, W, b, lamda)

    for i in range(len(b)):
        b_try = np.array(b)
        b_try[i] += h
        c2 = compute_cost(X, Y, W, b_try, lamda)
        grad_b[i] = (c2-c) / h

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_try = np.array(W)
            W_try[i, j] += h
            c2 = compute_cost(X, Y, W_try, b, lamda)
            grad_W[i, j] = (c2-c) / h

    return grad_W, grad_b


def check_gradients(X, Y, P, W, b, Lambda, h):
    # analytical gradients
    ana_w, ana_b = compute_gradients(
        X[:, 1:20], Y[:, 1:20], P[:, 1:20], W, Lambda)

    # numerical gradients
    num_w, num_b = ComputeGradsNum(
        X[:, 1:20], Y[:, 1:20], P[:, 1:20], W, b, Lambda, 0.000001)

    # relative error
    w_rel_error = np.absolute(
        ana_w-num_w) / np.maximum(0.0000001, np.absolute(ana_w) + np.absolute(num_w))

    b_rel_error = np.absolute(
        ana_b-num_b) / np.maximum(0.0000001, np.absolute(ana_b) + np.absolute(num_b))
    print('W-error:', w_rel_error)  # print for error analysis
    print('W-error max:', w_rel_error.max())
    print('b-error:', b_rel_error)  # print for error analysis
    print('b-error max:', b_rel_error.max())


def shuffle_data(labels, targets_one_hot, targets):
    assert labels.shape[1] == targets_one_hot.shape[1]
    p = np.random.permutation(labels.shape[1])
    return labels[:, p], targets_one_hot[:, p], [targets[i] for i in p]


def mini_batch_gd(X, Y, y, n_batch, eta, n_epochs, W, b, Lambda):
    n = X.shape[1]
    if(n % n_batch != 0):
        print("Data size mismatch batch_size")
        return -1
    W_list = []
    b_list = []
    cost_list = []
    loss_list = []
    accuracy_list = []
    for i in range(n_epochs):
        # shuffle_data
        X, Y, y = shuffle_data(X, Y, y)

        # Store current W and b before manipulating to allow for later visualization
        W_list.append(W)
        b_list.append(b)

        # Compute and print cost and accuracy for babysitting
        # Compute loss
        cost = compute_cost(X, Y, W, b, LAMBDA)
        loss = compute_cost(X, Y, W, b, 0)
        accuracy = compute_accuracy(X, y, W, b)
        cost_list.append(cost)
        loss_list.append(loss)
        accuracy_list.append(accuracy)
        print('Epoch    :', i+1)
        print('Loss    :', loss)
        print('Accuracy:', accuracy)

        for i in range(n//n_batch):
            # create current batch
            i_start = i*n_batch
            i_end = (i+1)*n_batch
            inds = list(range(i_start, i_end))
            Xbatch = X[:, inds]
            Ybatch = Y[:, inds]

            # Evaluate: P = softmax(WX+p)
            P_batch = evaluate_classifier(Xbatch, W, b)

            # Compute gradients
            W_grad, b_grad = compute_gradients(
                Xbatch, Ybatch, P_batch, W, Lambda)

            # Update W and b
            W = W-eta*W_grad
            b = b-eta*b_grad

        # break
    return W_list, b_list, loss_list, cost_list, accuracy_list


def plot_data(targets, patterns):
    class_A_indices = np.where(targets == -1)
    class_B_indices = np.where(targets == 1)
    patterns_A = patterns[class_A_indices]
    patterns_B = patterns[class_B_indices]

    plt.scatter(patterns_A[:, 0], patterns_A[:, 1], s=5)
    plt.scatter(patterns_B[:, 0], patterns_B[:, 1], s=5)

    plt.show()


def plot_overview(W_list, b_list, accuracy_list, loss_list, cost_list,
                  X_norm_validation, Y_validation, y_validation):
    """Plots accuracy and loss
    """

    # compute cost and
    # accuracy for validation
    val_losses = []
    val_costs = []
    val_accs = []
    for epoch in range(len(W_list)):
        current_loss = compute_cost(
            X_norm_validation, Y_validation, W_list[epoch], b_list[epoch], 0)
        current_cost = compute_cost(
            X_norm_validation, Y_validation, W_list[epoch], b_list[epoch], LAMBDA)
        current_accuracy = compute_accuracy(
            X_norm_validation, y_validation, W_list[epoch], b_list[epoch])
        val_losses.append(current_loss)
        val_costs.append(current_cost)
        val_accs.append(current_accuracy)
    figure, axis = plt.subplots(2)
    axis[0].plot(cost_list, label="Cost-Training", color='blue')
    axis[0].plot(loss_list, label="Loss-Training", color='green')
    axis[0].plot(val_costs, label="Cost-Validation", color='orange')
    axis[0].plot(val_losses, label="Loss-Validation", color='red')
    axis[0].legend(loc="upper left")

    axis[1].plot(accuracy_list, label="Accuracy-Training", color='green')
    axis[1].plot(val_accs, label="Accuracy-Validation", color='red')
    axis[1].legend(loc="upper left")

    plt.show()


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

# Compute gradients and check gradients
# check_gradients(X_norm_train, Y_training, P, W, b, LAMBDA, 0.0001)

W_list, b_list, loss_list, cost_list, acc_list = mini_batch_gd(X_norm_train, Y_training, y_training, N_BATCH,
                                                               LEARNING_RATE, N_EPOCHS, W, b, LAMBDA)
plot_overview(W_list, b_list, acc_list, loss_list, cost_list,
              X_norm_validation, Y_validation, y_validation)

montage(W_list[-1])
