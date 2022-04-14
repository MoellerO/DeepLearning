import numpy as np
import matplotlib.pyplot as plt
from functions import load_batch, preprocess_data, init_parameters, shuffle_data


# Constants
FILE_NAMES = ['data_batch_1', 'data_batch_2',
              'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']
K = 10
D = 3072
M = 50  # size hidden layer
LAMBDA = 0.01
LEARNING_RATE = 0.1
N_BATCH = 50
N_EPOCHS = 10
MU = 0
SIGMA_1 = 1/np.sqrt(D)  # used to init W1
SIGMA_2 = 1/np.sqrt(M)  # used to init W2


def evaluate_classifier(X, W1, W2, b1, b2):
    # (X: d*n, W1:m*d,W2:K*m b1:m*1, b2:K*1, p:K*n)
    batch_size = X.shape[1]
    ones_batch_size = np.ones(batch_size)[np.newaxis]
    # s1 = W1x+b1
    s1 = np.matmul(W1, X) + np.matmul(b1, ones_batch_size)
    h = np.maximum(0, s1)  # TODO
    s2 = np.matmul(W2, h) + np.matmul(b2, ones_batch_size)
    # SOFTMAX(s) = exp(s)/(1^T)exp(s)
    # denominator = (1^T)exp(s)
    denominator = np.matmul(np.ones(K), np.exp(s2))
    P = np.exp(s2)/denominator
    return h, P


def compute_gradients(X, Y, P, h, W1, W2):
    # (X: d*n, Y:k*n, P:K*N, W:K*d,  lamda: scalar, G_batch: k*n, h: m*n)
    n = X.shape[1]
    G_batch = -(Y-P)
    G_W2 = 1/n * np.matmul(G_batch,
                           np.transpose(h))
    ones_b = np.transpose(np.ones(n)[np.newaxis])
    G_b2 = 1/n * np.matmul(G_batch, ones_b)

    G_batch = np.matmul(np.transpose(W2), G_batch)  # m*n
    G_batch = np.multiply(G_batch, np.where(h > 0, 1, 0))
    G_W1 = 1/n * np.matmul(G_batch, np.transpose(X))
    G_b1 = 1/n * np.matmul(G_batch, ones_b)

    return G_W1, G_W2, G_b1, G_b2


def compute_cost(X, Y_one_hot, W1, W2, b1, b2, Lambda):
    """
    Computes cost/loss on whole data sets
    """
    # (X: d*n, Y:k*n ,W1:m*d,W2:K*m b1:m*1, b2:K*1, lamda: scalar)
    D = X.shape[1]
    _, P = evaluate_classifier(X, W1, W2, b1, b2)
    Y_t = np.transpose(Y_one_hot)
    l_cross = (-1) * np.matmul(Y_t, np.log(P))
    # J =  1/D * Sum(l_cross(x,y,W,b)+ lambda * Sum(W_{i,j}^2))
    J = 1/D * np.trace(l_cross) + Lambda * \
        (np.square(W1).sum() + np.square(W2).sum())
    return J


def compute_accuracy(X, y, W1, W2, b1, b2):
    _, P = evaluate_classifier(X, W1, W2, b1, b2)
    predicted_labels = np.argmax(P, axis=0)
    number_correct_predictions = np.sum(predicted_labels == y)
    accuracy = number_correct_predictions / P.shape[1]
    return accuracy


def mini_batch_gd(X, Y, y, n_batch, eta, n_epochs, W1, W2, b1, b2, Lambda):
    n = X.shape[1]
    if(n % n_batch != 0):
        print("Data size mismatch batch_size", n, n_batch)
        return -1
    W1_list = []
    W2_list = []
    b1_list = []
    b2_list = []
    cost_list = []
    loss_list = []
    accuracy_list = []
    for i in range(n_epochs):
        if(i != 0 and i % 5 == 0):
            eta = eta/10
        X, Y, y = shuffle_data(X, Y, y)

        # Store current W and b before manipulating to allow for later visualization
        W1_list.append(W1)
        W2_list.append(W2)
        b1_list.append(b1)
        b2_list.append(b2)

        # Compute and print cost and accuracy for babysitting
        # Compute loss
        print('[Note] Computing Cost..')
        cost = compute_cost(
            X, Y, W1, W2, b1, b2, Lambda)
        # cost = compute_cost_no_split_exercise_1(X, Y, W, b, Lambda)
        print('[Note] Computing Loss..')
        loss = compute_cost(
            X, Y, W1, W2, b1, b2, 0)
        # loss = compute_cost_no_split_exercise_1(X, Y, W, b, 0)
        print('[Note] Computing Acc..')
        accuracy = compute_accuracy(X, y, W1, W2, b1, b2)
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
            h_batch, P_batch = evaluate_classifier(Xbatch, W1, W2, b1, b2)

            # Compute gradients
            W1_grad, W2_grad, b1_grad, b2_grad = compute_gradients(
                Xbatch, Ybatch, P_batch, h_batch, W1, W2)

            # Update W and b
            W1 = W1-eta*W1_grad
            W2 = W2-eta*W2_grad
            b1 = b1-eta*b1_grad
            b2 = b2-eta*b2_grad

    return W1_list, W2_list, b1_list, b2_list, loss_list, cost_list, accuracy_list


def compute_validation_curves(X_norm_validation, Y_validation, y_validation, W1_list, W2_list, b1_list, b2_list, lamda):
    print('[Note] Evaluating Validation Performance..')
    val_losses = []
    val_costs = []
    val_accs = []
    for epoch in range(len(W1_list)):
        current_loss = compute_cost(
            X_norm_validation, Y_validation, W1_list[epoch], W2_list[epoch], b1_list[epoch], b2_list[epoch], 0)
        current_cost = compute_cost(
            X_norm_validation, Y_validation, W1_list[epoch], W2_list[epoch], b1_list[epoch], b2_list[epoch], lamda)
        current_accuracy = compute_accuracy(
            X_norm_validation, y_validation, W1_list[epoch], W2_list[epoch], b1_list[epoch], b2_list[epoch])
        val_losses.append(current_loss)
        val_costs.append(current_cost)
        val_accs.append(current_accuracy)
    print("Val-Accuracy:", val_accs[-1])
    print("Val-Loss:", val_losses[-1])
    return val_losses, val_costs, val_accs


def plot_overview(accuracy_list, loss_list, cost_list, val_losses, val_costs, val_accs):
    """Plots accuracy and loss
    """
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


print('[Note] Loading batches..')
X_training, Y_training, y_training = load_batch(FILE_NAMES[0])
X_validation, Y_validation, y_validation = load_batch(FILE_NAMES[1])
X_test, Y_test, y_test = load_batch(FILE_NAMES[2])

# normalize training, validation, and test data with mean and std from train-set
print('[Note] Normalizing data..')
X_norm_train, X_norm_validation, X_norm_test = preprocess_data(
    X_training, X_validation, X_test)

# initialize weights and bias
print('[Note] Initializing weights..')
W1, W2, b1, b2 = init_parameters(MU, SIGMA_1, SIGMA_2, K, D, M)
print(np.abs(np.sum(W1)))


h, P = evaluate_classifier(X_training, W1, W2, b1, b2)

G_W1, G_W2, G_b1, G_b2 = compute_gradients(
    X_norm_train, Y_training, P, h, W1, W2)

W1_list, W2_list, b1_list, b2_list, loss_list, cost_list, accuracy_list = mini_batch_gd(
    X_norm_train[:, 0:1000], Y_training[:, 0:1000], y_training[0:1000], N_BATCH, LEARNING_RATE, N_EPOCHS, W1, W2, b1, b2, LAMBDA)

val_losses, val_costs, val_accs = compute_validation_curves(
    X_norm_validation, Y_validation, y_validation, W1_list, W2_list, b1_list, b2_list, LAMBDA)

plot_overview(accuracy_list, loss_list, cost_list,
              val_losses, val_costs, val_accs)
