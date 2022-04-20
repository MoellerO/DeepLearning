import numpy as np
import matplotlib.pyplot as plt
import pickle
np.random.seed(0)

# Constants
FILE_NAMES = ['data_batch_1', 'data_batch_2',
              'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']
TEST_FILE = 'test_batch'

K = 10
D = 3072
M = 50  # size hidden layer
LAMBDA_DEFAULT = 0.01
LAMBDA_COARSE_MIN = -5
LAMBDA_COARSE_MAX = -1
LAMBDA_FINE_MIN = -4.5
LAMBDA_FINE_MAX = -3.5
LAMBDA_FINAL = 0.00003
LEARNING_RATE = 0.001
ETA_MIN = 0.00001
ETA_MAX = 0.1
ETA_K = 4  # final is 4
N_BATCH = 100
# Default is 8 (one cycle is 4 when 55k training samples), Final is 32
N_EPOCHS = 16

MU = 0
SIGMA_1 = 1/np.sqrt(D)  # used to init W1
SIGMA_2 = 1/np.sqrt(M)  # used to init W2


def init_weights(mu, sigma, dimensions):
    dim1, dim2 = dimensions[0], dimensions[1]
    W = np.random.normal(mu, sigma, size=(dim1, dim2))
    return W


def init_bias(mu, sigma, dimension):
    b = np.random.normal(mu, sigma, size=dimension)[np.newaxis]
    return np.transpose(b)


def init_parameters(mu, sigma1, sigma2, k, d, m):
    W1 = init_weights(mu, sigma1, [m, d])
    W2 = init_weights(mu, sigma2, [k, m])
    b1 = init_bias(0, 0, m)
    b2 = init_bias(0, 0, k)
    return W1, W2, b1, b2


def shuffle_data(features, targets_one_hot, targets):
    assert features.shape[1] == targets_one_hot.shape[1]
    p = np.random.permutation(features.shape[1])
    return features[:, p], targets_one_hot[:, p], [targets[i] for i in p]


def split_matrix(matrix, pieces):
    """Used to solve the problem that local device runs out of memory when trained with to much data
    """
    len = matrix.shape[1]
    part = len//pieces
    piece_list = []
    for i in range(pieces):
        m = matrix[:, i*part:(i+1)*part]
        piece_list.append(m)
    return piece_list


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


def preprocess_data(training, validation, test):
    mean = compute_mean(training)
    std = compute_std(training)
    norm_train = normalize(training, mean, std)
    norm_validation = normalize(validation, mean, std)
    norm_test = normalize(test, mean, std)
    return norm_train, norm_validation, norm_test


def create_one_hot(y, K):
    Y = np.zeros((K, len(y)))
    for image, label in enumerate(y):
        Y[label][image] = 1
    return Y


def load_all_batches(filenames):
    # For optional part 1a)
    X_list = []
    y_list = []
    for filename in filenames:

        with open('Datasets/'+filename, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        X_batch = np.transpose(dict[b'data'])
        y_batch = dict[b'labels']
        X_list.append(X_batch)
        y_list.append(y_batch)

    X = np.concatenate(X_list, axis=1)
    y = np.concatenate(y_list)
    Y = create_one_hot(y, 10)
    return X, Y, y


def load_batch(filename):
    """ Copied from the dataset website"""
    with open('Datasets/'+filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    X = np.transpose(dict[b'data'])
    y = dict[b'labels']
    Y = create_one_hot(y, 10)
    return X, Y, y


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
    if(X.shape[1] < 10000):
        D = X.shape[1]
        _, P = evaluate_classifier(X, W1, W2, b1, b2)
        Y_t = np.transpose(Y_one_hot)
        l_cross = (-1) * np.matmul(Y_t, np.log(P))
        # J =  1/D * Sum(l_cross(x,y,W,b)+ lambda * Sum(W_{i,j}^2))
        J = 1/D * np.trace(l_cross) + Lambda * \
            (np.square(W1).sum() + np.square(W2).sum())
        return J

    X_split = split_matrix(X, 10)
    Y_split = split_matrix(Y_one_hot, 10)
    costs = []
    for i in range(len(X_split)):
        X = X_split[i]
        Y = Y_split[i]
        D = X.shape[1]
        _, P = evaluate_classifier(X, W1, W2, b1, b2)
        Y_t = np.transpose(Y)
        l_cross = (-1) * np.matmul(Y_t, np.log(P))
        # J =  1/D * Sum(l_cross(x,y,W,b)+ lambda * Sum(W_{i,j}^2))
        J = 1/D * np.trace(l_cross) + Lambda * \
            (np.square(W1).sum() + np.square(W2).sum())
        costs.append(J)
    return np.mean(costs)


def compute_accuracy(X, y, W1, W2, b1, b2):
    _, P = evaluate_classifier(X, W1, W2, b1, b2)
    predicted_labels = np.argmax(P, axis=0)
    number_correct_predictions = np.sum(predicted_labels == y)
    accuracy = number_correct_predictions / P.shape[1]
    return accuracy


def mini_batch_gd(training_set, validation_set, n_batch, eta, n_epochs, W1, W2, b1, b2, Lambda):
    X_train = training_set[0]
    Y_train = training_set[1]
    y_train = training_set[2]
    X_validation = validation_set[0]
    Y_validation = validation_set[1]
    y_validation = validation_set[2]
    n = X_train.shape[1]
    ns = ETA_K*n//n_batch  # stepsize

    if(n % n_batch != 0):
        print("Data size mismatch batch_size", n, n_batch)
        return -1
    W1_list = []
    W2_list = []
    b1_list = []
    b2_list = []
    train_loss_list = []
    train_cost_list = []
    train_accuracy_list = []
    val_loss_list = []
    val_cost_list = []
    val_acc_list = []
    eta_list = []
    for epoch in range(n_epochs):

        X_train, Y_train, y_train = shuffle_data(X_train, Y_train, y_train)

        # Store current W and b before manipulating to allow for later visualization
        W1_list.append(W1)
        W2_list.append(W2)
        b1_list.append(b1)
        b2_list.append(b2)

        # Compute and print cost and accuracy for babysitting
        # Compute loss
        print('[Note] Computing Cost..')
        cost = compute_cost(
            X_train, Y_train, W1, W2, b1, b2, Lambda)
        print('[Note] Computing Loss..')
        loss = compute_cost(
            X_train, Y_train, W1, W2, b1, b2, 0)
        print('[Note] Computing Acc..')
        accuracy = compute_accuracy(X_train, y_train, W1, W2, b1, b2)
        train_cost_list.append(cost)
        train_loss_list.append(loss)
        train_accuracy_list.append(accuracy)
        print('[Note] Computing Validation Performance')
        val_cost = compute_cost(
            X_validation, Y_validation, W1, W2, b1, b2, Lambda)
        val_loss = compute_cost(X_validation, Y_validation, W1, W2, b1, b2, 0)
        val_acc = compute_accuracy(X_validation, y_validation, W1, W2, b1, b2)
        val_cost_list.append(val_cost)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        print('Loss               :', loss)
        print('Train Accuracy     :', accuracy)
        print('Validation Accuracy:', val_acc)
        print('Epoch              :', epoch+1)
        for i in range(n//n_batch):
            iteration = i + (epoch*n//n_batch)
            cycle = np.floor(1+iteration/(2*ns))
            x = np.abs(iteration/ns - 2*cycle + 1)
            eta = ETA_MIN + (ETA_MAX - ETA_MIN)*np.maximum(0, (1-x))
            # print(eta)
            eta_list.append(eta)

            # create current batch
            i_start = i*n_batch
            i_end = (i+1)*n_batch
            inds = list(range(i_start, i_end))
            Xbatch = X_train[:, inds]
            Ybatch = Y_train[:, inds]

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
    return W1_list, W2_list, b1_list, b2_list, train_loss_list, train_cost_list, train_accuracy_list, val_cost_list, val_loss_list, val_acc_list, eta_list


# TODO: Currently deprecedated and not in use
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


def get_datasets(filenames, testfile, val_size=1000):
    """Creates train and validation sets, while 5000 samples are used for validation and the rest for training.
        Furthermore, normalizes the all feature data according to mean and std from the train set.
    """
    print('[Note] Loading batches..')
    X, Y, y = load_all_batches(filenames)
    X_test, Y_test, y_test = load_batch(testfile)
    X_training = X[:, 0:-val_size]
    Y_training = Y[:, 0:-val_size]
    y_training = y[0:-val_size]
    X_validation = X[:, -val_size:]
    Y_validation = Y[:, -val_size:]
    y_validation = y[-val_size:]

    # normalize training, validation, and test data with mean and std from train-set
    print('[Note] Normalizing data..')
    X_norm_train, X_norm_validation, X_norm_test = preprocess_data(
        X_training, X_validation, X_test)

    return [X_norm_train, Y_training, y_training], [X_norm_validation, Y_validation, y_validation], [X_norm_test, Y_test, y_test]


def report_test_performance(X_test, Y_test, y_test, W_list, b_list, lamda):
    W1 = W_list[0]
    W2 = W_list[1]
    b1 = b_list[0]
    b2 = b_list[1]
    cost = compute_cost(X_test, Y_test, W1, W2, b1, b2, lamda)
    loss = compute_cost(X_test, Y_test, W1, W2, b1, b2, 0)
    acc = compute_accuracy(X_test, y_test, W1, W2, b1, b2)
    print("Test Loss:", loss)
    print("Test Cost:", cost)
    print("Test Acc:", acc)


def normal_program_execution(lamda, all_data=True):
    """ Gradient decent using the Constant Parameters defined at the beginning of this file and all available data.
    """
    print('[Note] Loading batches..')
    if all_data == True:
        train_data, val_data, test_data = get_datasets(FILE_NAMES, TEST_FILE)
    else:
        # using 10k trainig samples
        X_training, Y_training, y_training = load_batch(FILE_NAMES[0])
        X_validation, Y_validation, y_validation = load_batch(FILE_NAMES[1])
        X_test, Y_test, y_test = load_batch(FILE_NAMES[2])
        print('[Note] Normalizing data..')
        X_norm_train, X_norm_validation, X_norm_test = preprocess_data(
            X_training, X_validation, X_test)
        train_data = [X_norm_train, Y_training, y_training]
        val_data = [X_norm_validation, Y_validation, y_validation]
        test_data = [X_norm_test, Y_test, y_test]

    # initialize weights and bias
    print('[Note] Initializing weights..')
    W1, W2, b1, b2 = init_parameters(MU, SIGMA_1, SIGMA_2, K, D, M)

    W1_list, W2_list, b1_list, b2_list, train_loss_list, train_cost_list, train_accuracy_list, val_cost_list, val_loss_list, val_acc_list, eta_list = mini_batch_gd(
        [train_data[0], train_data[1], train_data[2]], [val_data[0], val_data[1], val_data[2]], N_BATCH, ETA_MIN, N_EPOCHS, W1, W2, b1, b2, lamda)

    plt.plot(eta_list)

    plot_overview(train_accuracy_list, train_loss_list, train_cost_list,
                  val_loss_list, val_cost_list, val_acc_list)

    report_test_performance(test_data[0], test_data[1], test_data[2], [
                            W1_list[-1], W2_list[-1]], [b1_list[-1], b2_list[-1]], lamda)


def lambda_search(lamda_min, lamda_max):
    print('[Note] Loading batches..')
    train_data, val_data, _ = get_datasets(FILE_NAMES, TEST_FILE)

    step = (lamda_max - lamda_min) / 10

    results = []
    for i in range(10):
        lamda = 10**(lamda_min + i*step)

        W1, W2, b1, b2 = init_parameters(MU, SIGMA_1, SIGMA_2, K, D, M)
        _, _, _, _, _, _, _, _, _, val_acc_list, _ = mini_batch_gd(
            [train_data[0], train_data[1], train_data[2]], [val_data[0], val_data[1], val_data[2]], N_BATCH, ETA_MIN, N_EPOCHS, W1, W2, b1, b2, lamda)
        results.append([round(lamda, 5), round(val_acc_list[-1], 2)])

    results.sort(key=lambda x: x[1], reverse=True)
    print(results)


def ComputeGradsNum(X, Y, W1, W2, b1, b2, lamda, h=1e-6):
    no1 = W1.shape[0]
    no2 = W2.shape[0]
    grad_W1 = np.zeros(W1.shape)
    grad_b1 = np.zeros((no1, 1))
    grad_W2 = np.zeros(W2.shape)
    grad_b2 = np.zeros((no2, 1))
    c = compute_cost(X, Y, W1, W2, b1, b2, lamda)

    for i in range(len(b1)):
        b_try = np.array(b1)
        b_try[i] += h
        c2 = compute_cost(X, Y, W1, W2, b_try, b2, lamda)
        grad_b1[i] = (c2-c) / h
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            W1_try = np.array(W1)
            W1_try[i, j] += h
            c2 = compute_cost(X, Y, W1_try, W2, b1, b2, lamda)
            grad_W1[i, j] = (c2-c) / h

    for i in range(len(b2)):
        b2_try = np.array(b2)
        b2_try[i] += h
        c2 = compute_cost(X, Y, W1, W2, b1, b2_try, lamda)
        grad_b2[i] = (c2-c) / h

    for i in range(W2.shape[0]):
        for j in range(W2.shape[1]):
            W2_try = np.array(W2)
            W2_try[i, j] += h
            c2 = compute_cost(X, Y, W1, W2_try, b1, b2, lamda)
            grad_W2[i, j] = (c2-c) / h

    return [grad_W1, grad_W2, grad_b1, grad_b2]


def check_gradients():
    train, val, _ = get_datasets(FILE_NAMES, TEST_FILE)
    W1, W2, b1, b2 = init_parameters(MU, SIGMA_1, SIGMA_2, K, D, M)

    X = train[0][:, 0:101]
    Y = train[1][:, 0:101]

    h, P = evaluate_classifier(X, W1, W2, b1, b2)

    ana_W1, ana_W2, ana_b1, ana_b2 = compute_gradients(X, Y, P, h, W1, W2)
    num_W1, num_W2, num_b1, num_b2 = ComputeGradsNum(
        X, Y, W1, W2, b1, b2, 0, h=1e-6)

    # relative error
    w1_rel_error = np.absolute(
        ana_W1-num_W1) / np.maximum(0.0000001, np.absolute(ana_W1) + np.absolute(num_W1))
    print("\nW1:")
    print(w1_rel_error)
    print(w1_rel_error.max())

    b1_rel_error = np.absolute(
        ana_b1-num_b1) / np.maximum(0.0000001, np.absolute(ana_b1) + np.absolute(num_b1))
    print("\nB1:")
    print(b1_rel_error)
    print(b1_rel_error.max())

    w2_rel_error = np.absolute(
        ana_W2-num_W2) / np.maximum(0.0000001, np.absolute(ana_W2) + np.absolute(num_W2))
    print("\nW2:")
    print(w2_rel_error)
    print(w2_rel_error.max())

    b2_rel_error = np.absolute(
        ana_b2-num_b2) / np.maximum(0.0000001, np.absolute(ana_b2) + np.absolute(num_b2))
    print("\nB2:")
    print(b2_rel_error)
    print(b2_rel_error.max())


# lambda_search(LAMBDA_COARSE_MIN, LAMBDA_COARSE_MAX)
# lambda_search(LAMBDA_FINE_MIN, LAMBDA_FINE_MAX)
normal_program_execution(LAMBDA_DEFAULT, False)
# normal_program_execution(LAMBDA_DEFAULT)
# normal_program_execution(LAMBDA_FINAL)


# check_gradients()
