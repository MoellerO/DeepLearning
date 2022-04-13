from cProfile import label
from multiprocessing import reduction
import random
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import sys
import pickle


from sympy import false

from functions import *

FILE_NAMES = ['data_batch_1', 'data_batch_2',
              'data_batch_3', 'data_batch_4', 'data_batch_5']
TRAINING_FILE = 'data_batch_1'
VALIDATION_FILE = 'data_batch_2'
TEST_FILE = 'test_batch'
K = 10
D = 3072
MU = 0
SIGMA = 0.1
LAMBDA = 0.01
LEARNING_RATE = 0.1
N_BATCH = 50
N_EPOCHS = 40

GS_LR = [0.1, 0.05, 0.01, 0.001, 0.0001]
GS_LAMBDA = [1, 0.1, 0.01, 0.001, 0.0001]
GS_N = [10, 100, 500, 1000, 7000]


# Comment in for random seed
np.random.seed(100)


def evaluate_classifier_exercise_1(X, W, b):
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


def evaluate_classifier_exercise_2(X, W, b):
    # (X: d*n, W:K*d, b:K*1, p:K*n)
    batch_size = X.shape[1]
    ones_batch_size = np.ones(batch_size)[np.newaxis]
    # s = Wx+b
    s = np.matmul(W, X) + np.matmul(b, ones_batch_size)

    # P = sigmoid(s)
    # sigmoid(s) = exp(s)/exp(s)+1
    P = np.exp(s)/(np.exp(s)+1)
    return P


def compute_cost_split(X, Y_one_hot, W, b, Lambda):
    # (X: d*n, Y:k*n ,W:K*d, b:K*1, lamda: scalar)
    """
    Computes cost/loss by dividing the data into smaller junks and building averages
    """
    X_split = split_matrix(X)
    Y_split = split_matrix(Y_one_hot)
    costs = []
    for i in range(len(X_split)):
        X = X_split[i]
        Y = Y_split[i]
        D = X.shape[1]
        P = evaluate_classifier_exercise_1(X, W, b)
        Y_t = np.transpose(Y)
        l_cross = (-1) * np.matmul(Y_t, np.log(P))
        # J =  1/D * Sum(l_cross(x,y,W,b)+ lambda * Sum(W_{i,j}^2))
        J = 1/D * np.trace(l_cross) + Lambda * np.square(W).sum()
        costs.append(J)
    return np.mean(J)


def compute_cost_no_split_exercise_1(X, Y_one_hot, W, b, Lambda):
    """
    Computes cost/loss on whole data sets
    """
    # (X: d*n, Y:k*n ,W:K*d, b:K*1, lamda: scalar)
    D = X.shape[1]
    P = evaluate_classifier_exercise_1(X, W, b)
    Y_t = np.transpose(Y_one_hot)
    l_cross = (-1) * np.matmul(Y_t, np.log(P))
    # J =  1/D * Sum(l_cross(x,y,W,b)+ lambda * Sum(W_{i,j}^2))
    J = 1/D * np.trace(l_cross) + Lambda * np.square(W).sum()
    return J


def compute_cost_split_exercise_2(X, Y_one_hot, W, b, Lambda):
    # (X: d*n, Y:k*n ,W:K*d, b:K*1, lamda: scalar)
    """
    Computes cost/loss by dividing the data into smaller junks and building averages
    """
    X_split = split_matrix(X)
    Y_split = split_matrix(Y_one_hot)
    costs = []
    for i in range(len(X_split)):
        X = X_split[i]
        Y = Y_split[i]
        D = X.shape[1]
        P = evaluate_classifier_exercise_2(X, W, b)
        Y_t = np.transpose(Y)
        bce = (-1/K) * (np.matmul((1-Y_t), np.log(1-P)) +
                        np.matmul((Y_t), np.log(P)))
        J = 1/D * np.trace(bce) + Lambda * np.square(W).sum()
        costs.append(J)
    return np.mean(J)


def compute_cost_no_split_exercise_2(X, Y_one_hot, W, b, Lambda):
    """
    Computes cost/loss on whole data sets
    """
    # (X: d*n, Y:k*n ,W:K*d, b:K*1, lamda: scalar)
    D = X.shape[1]
    P = evaluate_classifier_exercise_2(X, W, b)
    Y_t = np.transpose(Y_one_hot)
    bce = (-1/K) * (np.matmul((1-Y_t), np.log(1-P)) +
                    np.matmul((Y_t), np.log(P)))
    J = 1/D * np.trace(bce) + Lambda * np.square(W).sum()
    return J


def split_matrix(matrix):
    """Used to solve the problem that local device runs out of memory when trained with to much data
    """
    m1 = matrix[:, :10000]
    m2 = matrix[:, 10001:20000]
    m3 = matrix[:, 20001:30000]
    m4 = matrix[:, 30001:40000]
    m5 = matrix[:, 40001:50000]
    return [m1, m2, m3, m4, m5]


def compute_accuracy_exercise_1(X, y, W, b):
    P = evaluate_classifier_exercise_1(X, W, b)
    predicted_labels = np.argmax(P, axis=0)
    number_correct_predictions = np.sum(predicted_labels == y)
    accuracy = number_correct_predictions / P.shape[1]
    return accuracy


def compute_accuracy_exercise_2(X, y, W, b):
    P = evaluate_classifier_exercise_2(X, W, b)
    predicted_labels = np.argmax(P, axis=0)
    number_correct_predictions = np.sum(predicted_labels == y)
    accuracy = number_correct_predictions / P.shape[1]
    return accuracy


def create_histograms(X, y, W, b):
    P = evaluate_classifier_exercise_2(X, W, b)
    correctly_classified = []
    falsly_classified = []
    predicted_labels = np.argmax(P, axis=0)
    for image, label in enumerate(predicted_labels):
        ground_truth = y[image]
        if(label == y[image]):
            correctly_classified.append(P[ground_truth, image])
        else:
            falsly_classified.append(P[ground_truth, image])

    bins = np.arange(0, 1, 0.025)  # fixed bin size
    plt.xlim([0, 1])

    plt.hist(correctly_classified, bins=bins,
             alpha=0.5, label="correctly classfied")
    plt.hist(falsly_classified, bins=bins, alpha=0.5,
             label="Incorrectly classfied")
    plt.legend(loc='upper right')
    plt.title('Probability for the ground truth class')
    plt.xlabel('Probability (bin size = 0.025)')
    plt.ylabel('count')

    plt.show()


def compute_gradients_exercise_1(X, Y, P, W, Lambda):
    # (X: d*n, Y:k*n, P:K*N, W:K*d,  lamda: scalar)
    n = X.shape[1]
    G_batch = -(Y-P)
    G_W = 1/n * np.matmul(G_batch, np.transpose(X)) + 2*Lambda*W
    ones_b = np.transpose(np.ones(n)[np.newaxis])
    G_b = 1/n * np.matmul(G_batch, ones_b)
    return G_W, G_b


def compute_gradients_exercise_2(X, Y, P, W, Lambda):
    # (X: d*n, Y:k*n, P:K*N, W:K*d,  lamda: scalar)
    n = X.shape[1]
    G_batch = (1/K)*(-Y+P)
    G_W = 1/n * np.matmul(G_batch, np.transpose(X)) + 2*Lambda*W
    ones_b = np.transpose(np.ones(n)[np.newaxis])
    G_b = 1/n * np.matmul(G_batch, ones_b)
    return G_W, G_b


def ComputeGradsNumSlow(X, Y, P, W, b, lamda, h):
    """ Converted from matlab code """
    # From canvas page
    no = W.shape[0]
    d = X.shape[0]

    grad_W = np.zeros(W.shape)
    grad_b = np.zeros((no, 1))

    for i in range(len(b)):
        b_try = np.array(b)
        b_try[i] -= h
        c1 = compute_cost_no_split_exercise_1(X, Y, W, b_try, lamda)

        b_try = np.array(b)
        b_try[i] += h
        c2 = compute_cost_no_split_exercise_1(X, Y, W, b_try, lamda)

        grad_b[i] = (c2-c1) / (2*h)

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_try = np.array(W)
            W_try[i, j] -= h
            c1 = compute_cost_no_split_exercise_1(X, Y, W_try, b, lamda)

            W_try = np.array(W)
            W_try[i, j] += h
            c2 = compute_cost_no_split_exercise_1(X, Y, W_try, b, lamda)

            grad_W[i, j] = (c2-c1) / (2*h)

    return [grad_W, grad_b]


def ComputeGradsNum(X, Y, P, W, b, lamda, h):
    """ Converted from matlab code """
    # From Canvas page
    no = W.shape[0]
    d = X.shape[0]

    grad_W = np.zeros(W.shape)
    grad_b = np.zeros((no, 1))
    print(X.shape)

    c = compute_cost_no_split_exercise_1(X, Y, W, b, lamda)

    for i in range(len(b)):
        b_try = np.array(b)
        b_try[i] += h
        c2 = compute_cost_no_split_exercise_1(X, Y, W, b_try, lamda)
        grad_b[i] = (c2-c) / h

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_try = np.array(W)
            W_try[i, j] += h
            c2 = compute_cost_no_split_exercise_1(X, Y, W_try, b, lamda)
            grad_W[i, j] = (c2-c) / h

    return grad_W, grad_b


def check_gradients(X, Y, Lambda):
    W = init_weights(MU, SIGMA)
    b = init_bias(MU, SIGMA)

    print('[Note] Computing mean and std..')
    mean_training = compute_mean(X)
    std_training = compute_std(X)

    # normalize training, validation, and test data with mean and std from train-set
    print('[Note] Normalizing data..')
    X = normalize(X, mean_training, std_training)

    P = evaluate_classifier_exercise_1(X, W, b)
    # analytical gradients
    # ana_w, ana_b = compute_gradients(
    #     X[:, 0:20], Y[:, 0:20], P[:, 0:20], W, Lambda)
    ana_w, ana_b = compute_gradients_exercise_1(
        X[:, 0:100], Y[:, 0:100], P[:, 0:100], W, Lambda)

    # numerical gradients
    # num_w, num_b = ComputeGradsNum(
    #     X[:, 0:20], Y[:, 0:20], P[:, 0:20], W, b, Lambda, 0.000001)
    num_w, num_b = ComputeGradsNumSlow(
        X[:, 0:100], Y[:, 0:100], P[:, 0:100], W, b, Lambda, 0.000001)

    # relative error
    w_rel_error = np.absolute(
        ana_w-num_w) / np.maximum(0.0000001, np.absolute(ana_w) + np.absolute(num_w))

    b_rel_error = np.absolute(
        ana_b-num_b) / np.maximum(0.0000001, np.absolute(ana_b) + np.absolute(num_b))
    print('W-error:', w_rel_error)  # print for error analysis
    print('W-error max:', w_rel_error.max())
    print('b-error:', b_rel_error)  # print for error analysis
    print('b-error max:', b_rel_error.max())


def mini_batch_gd_exercise_1(X, Y, y, n_batch, eta, n_epochs, W, b, Lambda, grid_search):
    n = X.shape[1]
    if(n % n_batch != 0):
        print("Data size mismatch batch_size", n, n_batch)
        return -1
    W_list = []
    b_list = []
    cost_list = []
    loss_list = []
    accuracy_list = []
    for i in range(n_epochs):
        if(i != 0 and i % 5 == 0):
            eta = eta/10
        shuffle_data
        X, Y, y = shuffle_data(X, Y, y)

        # flip images with 50 % chance
        X = turn_images_fifty_percent(X)
        # Store current W and b before manipulating to allow for later visualization
        W_list.append(W)
        b_list.append(b)

        # Compute and print cost and accuracy for babysitting
        # Compute loss
        if(not grid_search):
            print('[Note] Computing Cost..')
            cost = compute_cost_split(X, Y, W, b, Lambda)
            # cost = compute_cost_no_split_exercise_1(X, Y, W, b, Lambda)
            print('[Note] Computing Loss..')
            loss = compute_cost_split(X, Y, W, b, 0)
            # loss = compute_cost_no_split_exercise_1(X, Y, W, b, 0)
            print('[Note] Computing Acc..')
            accuracy = compute_accuracy_exercise_1(X, y, W, b)
            cost_list.append(cost)
            loss_list.append(loss)
            accuracy_list.append(accuracy)
            print('Loss    :', loss)
            print('Accuracy:', accuracy)
            print('Epoch    :', i+1)

        for i in range(n//n_batch):
            # create current batch
            i_start = i*n_batch
            i_end = (i+1)*n_batch
            inds = list(range(i_start, i_end))
            Xbatch = X[:, inds]
            Ybatch = Y[:, inds]

            # Evaluate: P = softmax(WX+p)
            P_batch = evaluate_classifier_exercise_1(Xbatch, W, b)

            # Compute gradients
            W_grad, b_grad = compute_gradients_exercise_1(
                Xbatch, Ybatch, P_batch, W, Lambda)

            # Update W and b
            W = W-eta*W_grad
            b = b-eta*b_grad
    if(grid_search):
        print('[Note] Computing Cost..')
        cost = compute_cost_split(X, Y, W, b, Lambda)
        print('[Note] Computing Loss..')
        loss = compute_cost_split(X, Y, W, b, 0)
        print('[Note] Computing Acc..')
        accuracy = compute_accuracy_exercise_1(X, y, W, b)
        cost_list.append(cost)
        loss_list.append(loss)
        accuracy_list.append(accuracy)
        print('Loss    :', loss)
        print('Accuracy:', accuracy)
        # break
    return W_list, b_list, loss_list, cost_list, accuracy_list


def mini_batch_gd_exercise_2(X, Y, y, n_batch, eta, n_epochs, W, b, Lambda):
    n = X.shape[1]
    if(n % n_batch != 0):
        print("Data size mismatch batch_size", n, n_batch)
        return -1
    W_list = []
    b_list = []
    cost_list = []
    loss_list = []
    accuracy_list = []
    for i in range(n_epochs):
        if(i != 0 and i % 5 == 0):
            eta = eta/10
        # shuffle_data
        X, Y, y = shuffle_data(X, Y, y)

        # flip images with 50 % chance
        X = turn_images_fifty_percent(X)
        # Store current W and b before manipulating to allow for later visualization
        W_list.append(W)
        b_list.append(b)

        # Compute and print cost and accuracy for babysitting
        # Compute loss
        print('[Note] Computing Cost..')
        # TODO: Update cost function
        cost = compute_cost_split_exercise_2(X, Y, W, b, Lambda)
        # cost = compute_cost_no_split_exercise_2(X, Y, W, b, Lambda)
        print('[Note] Computing Loss..')
        loss = compute_cost_split_exercise_2(X, Y, W, b, 0)
        # loss = compute_cost_no_split_exercise_2(X, Y, W, b, 0)
        print('[Note] Computing Acc..')
        accuracy = compute_accuracy_exercise_2(X, y, W, b)
        cost_list.append(cost)
        loss_list.append(loss)
        accuracy_list.append(accuracy)
        print('Loss    :', loss)
        print('Accuracy:', accuracy)
        print('Epoch    :', i+1)
        for i in range(n//n_batch):
            # create current batch
            i_start = i*n_batch
            i_end = (i+1)*n_batch
            inds = list(range(i_start, i_end))
            Xbatch = X[:, inds]
            Ybatch = Y[:, inds]

            # Evaluate: P = softmax(WX+p)
            P_batch = evaluate_classifier_exercise_2(Xbatch, W, b)

            # Compute gradients
            W_grad, b_grad = compute_gradients_exercise_2(
                Xbatch, Ybatch, P_batch, W, Lambda)

            # Update W and b
            W = W-eta*W_grad
            b = b-eta*b_grad

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
                  X_norm_validation, Y_validation, y_validation, lamda):
    """Plots accuracy and loss
    """

    # compute cost and
    # accuracy for validation
    val_losses = []
    val_costs = []
    val_accs = []
    for epoch in range(len(W_list)):
        current_loss = compute_cost_no_split_exercise_2(
            X_norm_validation, Y_validation, W_list[epoch], b_list[epoch], 0)
        current_cost = compute_cost_no_split_exercise_2(
            X_norm_validation, Y_validation, W_list[epoch], b_list[epoch], lamda)
        current_accuracy = compute_accuracy_exercise_2(
            X_norm_validation, y_validation, W_list[epoch], b_list[epoch])
        val_losses.append(current_loss)
        val_costs.append(current_cost)
        val_accs.append(current_accuracy)
    print("Val-Accuracy:", val_accs[-1])
    print("Val-Loss:", val_losses[-1])
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


def execute_gds_exercise_1(eta, lamda, n, X_training, Y_training, y_training, X_validation, Y_validation, y_validation, X_test, Y_test, y_test, grid_search=False):
    # compute mean and standard deviation
    print('[Note] Computing mean and std..')
    mean_training = compute_mean(X_training)
    std_training = compute_std(X_training)

    # normalize training, validation, and test data with mean and std from train-set
    print('[Note] Normalizing data..')
    X_norm_train = normalize(X_training, mean_training, std_training)
    X_norm_validation = normalize(X_validation, mean_training, std_training)
    X_norm_test = normalize(X_test, mean_training, std_training)

    # initialize weights and bias
    print('[Note] Initializing weights..')
    W = init_weights(MU, SIGMA)
    b = init_bias(MU, SIGMA)

    # Compute gradients and check gradients
    # check_gradients(X_norm_train, Y_training, P, W, b, LAMBDA, 0.0001)
    print('[Note] Starting gradient decent..')
    W_list, b_list, loss_list, cost_list, acc_list = mini_batch_gd_exercise_1(X_norm_train, Y_training, y_training, n,
                                                                              eta, N_EPOCHS, W, b, lamda, grid_search)
    if(not grid_search):
        create_histograms(X_norm_test, y_test, W_list[-1], b_list[-1])
        print('[Note] Plotting..')
        plot_overview(W_list, b_list, acc_list, loss_list, cost_list,
                      X_norm_validation, Y_validation, y_validation, lamda)
        montage(W_list[-1])
    else:
        loss_validation = compute_cost_no_split_exercise_1(
            X_norm_validation, Y_validation, W_list[-1], b_list[-1], 0)
        accuracy_validation = compute_accuracy_exercise_1(
            X_norm_validation, y_validation, W_list[-1], b_list[-1])

        return accuracy_validation, loss_validation, loss_list[-1], cost_list[-1], acc_list[-1]


def execute_gds_exercise_2(eta, lamda, n, X_training, Y_training, y_training, X_validation, Y_validation, y_validation, X_test, Y_test, y_test):
    # compute mean and standard deviation
    print('[Note] Computing mean and std..')
    mean_training = compute_mean(X_training)
    std_training = compute_std(X_training)

    # normalize training, validation, and test data with mean and std from train-set
    print('[Note] Normalizing data..')
    X_norm_train = normalize(X_training, mean_training, std_training)
    X_norm_validation = normalize(X_validation, mean_training, std_training)
    X_norm_test = normalize(X_test, mean_training, std_training)

    # initialize weights and bias
    print('[Note] Initializing weights..')
    W = init_weights(MU, SIGMA)
    b = init_bias(MU, SIGMA)

    # Compute gradients and check gradients
    # check_gradients(X_norm_train, Y_training, P, W, b, LAMBDA, 0.0001)
    print('[Note] Starting gradient decent..')
    W_list, b_list, loss_list, cost_list, acc_list = mini_batch_gd_exercise_2(X_norm_train, Y_training, y_training, n,
                                                                              eta, N_EPOCHS, W, b, lamda)

    create_histograms(X_norm_test, y_test, W_list[-1], b_list[-1])
    print('[Note] Plotting..')
    plot_overview(W_list, b_list, acc_list, loss_list, cost_list,
                  X_norm_validation, Y_validation, y_validation, lamda)
    montage(W_list[-1])


def turn_images_fifty_percent(X):
    n = X.shape[1]
    # computed indexes to swap
    aa = np.array(list(range(32)))
    bb = np.array(list(range(31, -1, -1)))[np.newaxis]
    vv = np.matlib.repmat(32*aa, 32, 1)
    raveld_vv = np.transpose(vv.ravel('F')[np.newaxis])
    ind_flip = raveld_vv + np.matlib.repmat(np.transpose(bb), 32, 1)
    inds_flipped = np.concatenate(
        [ind_flip, 1023+ind_flip, 2047+ind_flip]).ravel()

    for i in range(n):
        if(random.choice([0, 1])):
            X[:, i] = X[inds_flipped, i]

    return X


def grid_search(etas, lamdas, batch_sizes, X_training, Y_training, y_training,
                X_validation, Y_validation, y_validation):
    scores = []
    runs = len(etas)*len(lamdas)*len(batch_sizes)
    i = 1
    for eta in etas:
        for lamda in lamdas:
            for n in batch_sizes:
                print('Run:', i, 'of', runs)
                val_acc, val_loss, tr_loss, tr_cost, tr_acc = execute_gds_exercise_1(eta, lamda, n, X_training, Y_training, y_training,
                                                                                     X_validation, Y_validation, y_validation, True)

                score = [eta, lamda, n, tr_loss, tr_acc, val_loss, val_acc]
                scores.append(score)
                i += 1
    for row in scores:
        print(row)


def handle_gridsearch_results(filename):
    with open(filename) as file:
        lines = [line.rstrip() for line in file]

    for index, line in enumerate(lines):
        lines[index] = line.split()
        lines[index][0] = float(lines[index][0][1:-1])
        lines[index][1] = float(lines[index][1][:-1])
        lines[index][2] = int(lines[index][2][:-1])
        lines[index][3] = np.format_float_positional(
            float(lines[index][3][:-1]), 3)
        lines[index][4] = np.format_float_positional(
            float(lines[index][4][:-1]), 3)
        lines[index][5] = np.format_float_positional(
            float(lines[index][5][:-1]), 3)
        lines[index][6] = np.format_float_positional(
            float(lines[index][6][:-1]), 3)
    arr = np.array(lines)
    # sort after accuracy
    arr = arr[arr[:, 4].argsort()[::-1]]
    # print(arr)
    print(arr.shape)
    print(arr)


# BEGIN: read in and split data
# Comment in for EXERCISE 1:
# load training, validation, and test data
# X_training, Y_training, y_training = load_batch(TRAINING_FILE)
# X_validation, Y_validation, y_validation = load_batch(VALIDATION_FILE)
X_test, Y_test, y_test = load_batch(TEST_FILE)

# Comment in for EXERCISE 2:
print('[Note] Loading data..')
X, Y, y = load_all_batches(FILE_NAMES)
print('[Note] Seperating data..')
X_training = X[:, 0:-1000]
Y_training = Y[:, 0:-1000]
y_training = y[0:-1000]
X_validation = X[:, -1000:]
Y_validation = Y[:, -1000:]
y_validation = y[-1000:]

# END: read in and split data

# Comment in to check for gradients
# check_gradients(X, Y, 0.01)

# Comment in to for image visualization
# montage(np.transpose(X))

# Execute gradient decent
# execute_gds_exercise_1(LEARNING_RATE, LAMBDA, N_BATCH, X_training, Y_training, y_training,
#                        X_validation, Y_validation, y_validation, X_test, Y_test, y_test)
execute_gds_exercise_2(LEARNING_RATE, LAMBDA, N_BATCH, X_training, Y_training, y_training,
                       X_validation, Y_validation, y_validation, X_test, Y_test, y_test)


# Comment in for grid search
# grid_search(GS_LR, GS_LAMBDA, GS_N, X_training, Y_training, y_training,
#             X_validation, Y_validation, y_validation)


# Comment in to load grid search results from text file
# handle_gridsearch_results('results_gridsearch.txt')

# Comment in to visualize flipped images
# montage(np.transpose(X))
# turned = turn_images_fifty_percent(X)
# montage(np.transpose(turned))
