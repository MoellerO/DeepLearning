import random
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import pickle

import csv
np.random.seed(10)
random.seed(10)

# TODO: TQDM library

# Constants
FILE_NAMES = ['data_batch_1', 'data_batch_2',
              'data_batch_3', 'data_batch_4', 'data_batch_5']
TEST_FILE = 'test_batch'


def init_weights(mu, sigma, dimensions):
    dim1, dim2 = dimensions[0], dimensions[1]
    W = np.random.normal(mu, sigma, size=(dim1, dim2))
    return W


def init_bias(mu, sigma, dimension):
    b = np.random.normal(mu, sigma, size=dimension)[np.newaxis]
    return np.transpose(b)


def init_parameters(hiddenlayer, number_classes=10, data_dimensions=3072):
    number_layers = len(hiddenlayer)+1
    assert number_layers > 0

    mu = 0  # data mean
    columns = data_dimensions  # columns for the created matrix
    weights = []  # list containing the weight-matrices for all layers
    biases = []  # list containing the bias-vectors for all layers
    for number_nodes in hiddenlayer:
        sigma = np.sqrt(2/columns)
        weight = init_weights(mu, sigma, [number_nodes, columns])
        bias = init_bias(0, 0, number_nodes)
        columns = number_nodes
        weights.append(weight)
        biases.append(bias)
    last_weight = init_weights(mu, sigma, [number_classes, columns])
    last_bias = init_bias(0, 0, number_classes)
    weights.append(last_weight)
    biases.append(last_bias)
    return weights, biases


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


def ComputeGradsNum(X, Y, weights, biases, lamda, h=1e-6):
    c = compute_cost(X, Y, weights, biases, lamda)

    W_gradients = []
    B_gradients = []

    for l, weight in enumerate(weights):
        weights_copy = weights[:]
        grad_W = np.zeros(weight.shape)
        for i in range(weight.shape[0]):
            for j in range(weight.shape[1]):
                W_try = np.array(weight)
                W_try[i, j] += h
                weights_copy[l] = W_try
                c2 = compute_cost(X, Y, weights_copy, biases, lamda)
                grad_W[i, j] = (c2-c) / h
        W_gradients.append(grad_W)

        biases_copy = biases[:]
        grad_b = np.zeros((weight.shape[0], 1))
        for i in range(len(biases_copy[l])):
            b_try = np.array(biases_copy[l])
            b_try[i] += h
            biases_copy[l] = b_try
            c2 = compute_cost(X, Y, weights, biases_copy, lamda)
            grad_b[i] = (c2-c) / h
        B_gradients.append(grad_b)

    return W_gradients, B_gradients


def check_gradients(hyperparams):
    train, val, _ = get_50k_training(FILE_NAMES, TEST_FILE)
    weights, biases = init_parameters(hyperparams['HIDDENLAYERS'])

    X = train[0][:, 0:21]
    Y = train[1][:, 0:21]

    layers, P = evaluate_classifier(X, weights, biases)

    ana_w_gradients, ana_b_gradients = compute_gradients(
        X, Y, P, layers, weights, 0)
    num_w_gradients, num_b_gradients = ComputeGradsNum(
        X, Y, weights, biases, 0, h=1e-6)

    assert len(ana_w_gradients) == len(num_w_gradients)
    for l in range(len(ana_w_gradients)):
        # relative error
        wl_rel_error = np.absolute(
            ana_w_gradients[l]-num_w_gradients[l]) / np.maximum(0.0000001, np.absolute(ana_w_gradients[l]) + np.absolute(num_w_gradients[l]))
        print('\nW', l, ':')
        print(wl_rel_error)
        print(wl_rel_error.max())

    assert len(ana_b_gradients) == len(num_b_gradients)
    for l in range(len(ana_b_gradients)):
        b1_rel_error = np.absolute(
            ana_b_gradients[l]-num_b_gradients[l]) / np.maximum(0.0000001, np.absolute(ana_b_gradients[l]) + np.absolute(num_b_gradients[l]))
        print("\nB", l, ':')
        print(b1_rel_error)
        print(b1_rel_error.max())


def montage(W):
    """ Display the image for each label in W """
    # From canvas page
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


def check_data_augmentation():
    X, Y, y = load_all_batches(FILE_NAMES)

    # montage(np.transpose(X))
    montage(np.transpose(X))

    X, Y, y = augment_data(X, Y, y)

    montage(np.transpose(X))


def evaluate_classifier(X, weights, biases, drop_rate=0):
    # (X: d*n, W1:m*d,W2:K*m b1:m*1, b2:K*1, p:K*n)
    number_classes = 10
    batch_size = X.shape[1]
    ones_batch_size = np.ones(batch_size)[np.newaxis]
    # TODO: CAREFULL WHEN COMPUTING GRADIENTS; MAYBE IGNORE FIRST ELEMENT
    h_storage = [X]
    for l in range(len(weights)-1):
        s = np.matmul(weights[l], h_storage[-1]) + \
            np.matmul(biases[l], ones_batch_size)
        if(drop_rate > 0):
            u = (np.random.rand(*s.shape) < drop_rate) / \
                drop_rate  # dropout mask
            s *= u  # drop !
        h = np.maximum(0, s)
        h_storage.append(h)

    s = np.matmul(weights[-1], h_storage[-1]) + \
        np.matmul(biases[-1], ones_batch_size)
    denominator = np.matmul(np.ones(number_classes), np.exp(s))
    P = np.exp(s)/denominator
    return h_storage, P


def compute_gradients(X, Y, P, layers, weights, lamda):
    # (X: d*n, Y:k*n, P:K*N, W:K*d,  lamda: scalar, G_batch: k*n, h: m*n)
    n = X.shape[1]
    G_batch = -(Y-P)

    assert len(layers) == len(weights)

    layers = layers[::-1]
    weights = weights[::-1]
    G_W_storage = []
    G_b_storage = []
    for l in range(len(layers)):
        G_W = 1/n * np.matmul(G_batch,
                              np.transpose(layers[l]))
        ones_b = np.transpose(np.ones(n)[np.newaxis])
        G_b = 1/n * np.matmul(G_batch, ones_b)
        G_batch = np.matmul(np.transpose(weights[l]), G_batch)  # m*n
        G_batch = np.multiply(G_batch, np.where(layers[l] > 0, 1, 0))

        # apply regularization
        G_W = G_W + 2*lamda*weights[l]
        G_W_storage.insert(0, G_W)
        G_b_storage.insert(0, G_b)

    return G_W_storage, G_b_storage


def compute_cost(X, Y_one_hot, weights, biases, Lambda):
    """
    Computes cost/loss on whole data sets
    """
    # (X: d*n, Y:k*n ,W1:m*d,W2:K*m b1:m*1, b2:K*1, lamda: scalar)
    if(X.shape[1] < 10000):
        D = X.shape[1]
        _, P = evaluate_classifier(X, weights, biases)
        Y_t = np.transpose(Y_one_hot)
        l_cross = (-1) * np.matmul(Y_t, np.log(P))
        # J =  1/D * Sum(l_cross(x,y,W,b)+ lambda * Sum(W_{i,j}^2))
        weight_sum = 0
        for weight in weights:
            weight_sum += np.square(weight).sum()
        J = 1/D * np.trace(l_cross) + Lambda * weight_sum
        return J

    X_split = split_matrix(X, 100)
    Y_split = split_matrix(Y_one_hot, 100)
    costs = []
    for i in range(len(X_split)):
        X = X_split[i]
        Y = Y_split[i]
        D = X.shape[1]
        _, P = evaluate_classifier(X, weights, biases)
        Y_t = np.transpose(Y)
        l_cross = (-1) * np.matmul(Y_t, np.log(P))
        # J =  1/D * Sum(l_cross(x,y,W,b)+ lambda * Sum(W_{i,j}^2))
        weight_sum = 0
        for weight in weights:
            weight_sum += np.square(weight).sum()
        J = 1/D * np.trace(l_cross) + Lambda * weight_sum
        costs.append(J)
    return np.mean(costs)


def compute_accuracy(X, y, weights, biases, dropout=0):
    _, P = evaluate_classifier(X, weights, biases, dropout)
    predicted_labels = np.argmax(P, axis=0)
    number_correct_predictions = np.sum(predicted_labels == y)
    accuracy = number_correct_predictions / P.shape[1]
    return accuracy


def anneal_eta_max(epoch, eta_max, frequency):
    new_eta_max = eta_max
    if(epoch != 0 and epoch % frequency == 0):
        new_eta_max = eta_max/2
    return new_eta_max


def mirror_images(X):
    """_summary_
    turns each images with a fifty percent change
    """
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
        # if(random.choice([0, 1])):
        X[:, i] = X[inds_flipped, i]

    return X


def compute_translations():
    translations = []
    for tx in range(-3, 4):
        for ty in range(-3, 4):
            if(tx != 0 and ty != 0):
                aa = np.array(list(range(32)))

                if (tx >= 0):
                    bb1 = np.array(list(range(tx, 32)))[np.newaxis]  # to fill
                    bb2 = np.array(list(range(32-tx)))[np.newaxis]  # to move
                    vv = np.matlib.repmat(32*aa, 32-tx, 1)  # 31x32
                else:
                    bb1 = np.array(list(range(32+tx)))[np.newaxis]  # to fill
                    bb2 = np.array(list(range(-tx, 32)))[np.newaxis]  # to move
                    vv = np.matlib.repmat(32*aa, 32+tx, 1)  # 31x32

                raveld_vv = np.transpose(vv.ravel('F')[np.newaxis])
                ind_fill = raveld_vv + \
                    np.matlib.repmat(np.transpose(bb1), 32,
                                     1)  # indices to fill
                ind_xx = raveld_vv + \
                    np.matlib.repmat(np.transpose(bb2), 32,
                                     1)  # indices to move

                if(ty >= 0):
                    ii = np.where(ind_fill >= ty*32)[0]
                    # indices to fill after pushing rows up
                    ind_fill = ind_fill[ii]
                    ii = np.where(ind_xx <= 1023-ty*32)[0]
                    ind_xx = ind_xx[ii]  # indices to move
                else:
                    ii = np.where(ind_fill <= 1023+ty*32)[0]
                    # indices to fill after pushing rows up
                    ind_fill = ind_fill[ii]
                    ii = np.where(ind_xx >= -ty*32)[0]
                    ind_xx = ind_xx[ii]  # indices to move

                inds_fill = np.concatenate(
                    [ind_fill, 1024+ind_fill, 2048+ind_fill]).ravel()
                inds_xx = np.concatenate(
                    [ind_xx, 1024+ind_xx, 2048+ind_xx]).ravel()

                complement = list(set(range(3072)) - set(inds_fill))
                translations.append([inds_fill, inds_xx, complement])
    return translations


def shift_images(X):
    translations = compute_translations()
    for i in range(X.shape[1]):
        translation = random.choice(translations)
        X[translation[0], i] = X[translation[1], i]
        # X[translation[2], i] = 0
    return X


def augment_data(X, Y, y):
    X_mirror = mirror_images(np.copy(X))
    X_shift = shift_images(np.copy(X))
    X_out = np.concatenate(
        (X, X_mirror, X_shift), axis=1)
    Y = np.concatenate((Y, Y, Y), axis=1)
    y = np.concatenate((y, y, y))

    return X_out, Y, y


def get_50k_training(filenames, testfile, val_size=1000):
    """Creates train and validation sets, while 5000 samples are used for validation and the rest for training.
        Furthermore, normalizes the all feature data according to mean and std from the train set.
    """
    # print('[Note] Loading batches..')
    X, Y, y = load_all_batches(filenames)
    X_test, Y_test, y_test = load_batch(testfile)
    X_training = X[:, 0:-val_size]
    Y_training = Y[:, 0:-val_size]
    y_training = y[0:-val_size]
    X_validation = X[:, -val_size:]
    Y_validation = Y[:, -val_size:]
    y_validation = y[-val_size:]

    # normalize training, validation, and test data with mean and std from train-set
    # print('[Note] Normalizing data..')
    X_norm_train, X_norm_validation, X_norm_test = preprocess_data(
        X_training, X_validation, X_test)

    return [X_norm_train, Y_training, y_training], [X_norm_validation, Y_validation, y_validation], [X_norm_test, Y_test, y_test]


def get_10k_training():
    X_training, Y_training, y_training = load_batch(FILE_NAMES[0])
    X_validation, Y_validation, y_validation = load_batch(FILE_NAMES[1])
    X_test, Y_test, y_test = load_batch(FILE_NAMES[2])
    print('[Note] Normalizing data..')
    X_norm_train, X_norm_validation, X_norm_test = preprocess_data(
        X_training, X_validation, X_test)
    return [X_norm_train, Y_training, y_training], [X_norm_validation, Y_validation, y_validation], [X_norm_test, Y_test, y_test]


def get_data_splits(all_data=True, data_augmentation=True):
    print('[Note] Loading batches..')
    if all_data == True:
        train_data, val_data, test_data = get_50k_training(
            FILE_NAMES, TEST_FILE, 5000)
    else:
        # using 10k trainig samples
        train_data, val_data, test_data = get_10k_training()

    if(data_augmentation):
        print('[Note] Augmenting Data..')
        X_train, Y_train, y_train = augment_data(
            train_data[0], train_data[1], train_data[2])
        train_data = [X_train, Y_train, y_train]

    return train_data, val_data, test_data


def report_test_performance(X_test, Y_test, y_test, weights, biases, lamda):
    # cost = compute_cost(X_test, Y_test, weights, biases, lamda)
    # loss = compute_cost(X_test, Y_test, weights, biases, 0)
    acc = compute_accuracy(X_test, y_test, weights, biases)
    # print("Test Loss:", round(loss, 2))
    # print("Test Cost:", round(cost, 2))
    print("Test Acc:", round(acc, 2))


def mini_batch_gd(training_set, validation_set,  weights, biases, hyperparams):
    # n_batch, n_epochs, eta, anneal_frequency, Lambda, isAugment, dropout = hyperparams

    # W1 = weights[0]
    # W2 = weights[1]
    # b1 = biases[0]
    # b2 = biases[1]
    X_train = training_set[0]
    Y_train = training_set[1]
    y_train = training_set[2]
    X_validation = validation_set[0]
    Y_validation = validation_set[1]
    y_validation = validation_set[2]
    n = X_train.shape[1]
    ns = hyperparams['ETA_K']*n//hyperparams['N_BATCH']  # stepsize
    eta_max = hyperparams['ETA_MAX']

    if(n % hyperparams['N_BATCH'] != 0):
        print("Data size mismatch batch_size", n, hyperparams['N_BATCH'])
        return -1
    # W1_list = []
    # W2_list = []
    # b1_list = []
    # b2_list = []
    weight_storage = []
    bias_storage = []

    train_loss_list = []
    train_cost_list = []
    train_accuracy_list = []
    val_loss_list = []
    val_cost_list = []
    val_acc_list = []
    eta_list = []
    for epoch in range(hyperparams['N_EPOCHS']):

        eta_max = anneal_eta_max(
            epoch, eta_max, hyperparams['ANNEAL_FREQUENCY'])

        X_train, Y_train, y_train = shuffle_data(X_train, Y_train, y_train)

        # Store current W and b before manipulating to allow for later visualization
        # W1_list.append(W1)
        # W2_list.append(W2)
        # b1_list.append(b1)
        # b2_list.append(b2)
        weight_storage.append(weights)
        bias_storage.append(biases)

        # Compute and print cost and accuracy for babysitting
        # Compute loss
        # print('[Note] Computing Cost..')
        cost = compute_cost(
            X_train, Y_train, weights, biases, hyperparams['LAMBDA'])
        # print('[Note] Computing Loss..')
        loss = compute_cost(
            X_train, Y_train, weights, biases, 0)
        # print('[Note] Computing Acc..')
        accuracy = compute_accuracy(X_train, y_train, weights, biases)
        train_cost_list.append(cost)
        train_loss_list.append(loss)
        train_accuracy_list.append(accuracy)
        # print('[Note] Computing Validation Performance')
        val_cost = compute_cost(
            X_validation, Y_validation, weights, biases, hyperparams['LAMBDA'])
        val_loss = compute_cost(X_validation, Y_validation, weights, biases, 0)
        val_acc = compute_accuracy(X_validation, y_validation, weights, biases)
        val_cost_list.append(val_cost)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        print('Epoch              :', epoch+1)
        print('Loss               :', round(loss, 2))
        print('Train Accuracy     :', round(accuracy, 2))
        print('Validation Accuracy:', round(val_acc, 2))
        for i in range(n//hyperparams['N_BATCH']):
            iteration = i + (epoch*n//hyperparams['N_BATCH'])
            cycle = np.floor(1+iteration/(2*ns))
            x = np.abs(iteration/ns - 2*cycle + 1)
            eta = hyperparams['ETA_MIN'] + \
                (eta_max - hyperparams['ETA_MIN'])*np.maximum(0, (1-x))
            # print(eta)
            eta_list.append(eta)

            # create current batch
            i_start = i*hyperparams['N_BATCH']
            i_end = (i+1)*hyperparams['N_BATCH']
            inds = list(range(i_start, i_end))
            Xbatch = X_train[:, inds]
            Ybatch = Y_train[:, inds]

            # Evaluate: P = softmax(WX+p)
            layers, P_batch = evaluate_classifier(
                Xbatch, weights, biases, hyperparams['DROP_RATE'])

            # Compute gradients
            weight_gradients, bias_gradients = compute_gradients(
                Xbatch, Ybatch, P_batch, layers, weights, hyperparams['LAMBDA'])

            # Update W and b
            for i in range(len(weights)):
                weights[i] -= eta*weight_gradients[i]
                biases[i] -= eta*bias_gradients[i]

            # weights[0] = weights[0]-eta*W1_grad
            # weights[1] = weights[1]-eta*W2_grad
            # biases[0] = biases[0]-eta*b1_grad
            # biases[1] = biases[1]-eta*b2_grad
    return weight_storage, bias_storage, [train_accuracy_list, train_loss_list, train_cost_list], [val_acc_list, val_loss_list, val_cost_list], eta_list


def plot_overview(accuracy_performance, validation_performance):
    """Plots accuracy and loss
    """
    figure, axis = plt.subplots(2)
    axis[0].plot(accuracy_performance[2], label="Cost-Training", color='blue')
    axis[0].plot(accuracy_performance[1], label="Loss-Training", color='green')
    axis[0].plot(validation_performance[2],
                 label="Cost-Validation", color='orange')
    axis[0].plot(validation_performance[1],
                 label="Loss-Validation", color='red')
    axis[0].legend(loc="upper left")

    axis[1].plot(accuracy_performance[0],
                 label="Accuracy-Training", color='green')
    axis[1].plot(validation_performance[0],
                 label="Accuracy-Validation", color='red')
    axis[1].legend(loc="upper left")

    plt.show()


def normal_program_execution(hyperparams, small_data=True):
    """ Gradient decent using the Constant Parameters defined at the beginning of this file and all available data.
    """
    train_data, val_data, test_data = get_data_splits(
        hyperparams['USE_ALL_DATA'], hyperparams['USE_DATA_AUGMENTATION'])

    # initialize weights and bias
    print('[Note] Initializing weights..')
    weights, biases = init_parameters(hyperparams['HIDDENLAYERS'])

    # n_batch, n_epochs, eta, anneal_frequency, Lambda, isAugment, dropout
    weight_storage, bias_storage, train_performance,  validation_performance, eta_list = mini_batch_gd(
        train_data, val_data, weights, biases, hyperparams)

    plt.plot(eta_list)

    plot_overview(train_performance,  validation_performance)

    report_test_performance(test_data[0], test_data[1], test_data[2],
                            weight_storage[-1], bias_storage[-1], hyperparams['LAMBDA'])


hps = {
    'N_BATCH': 100,
    'N_EPOCHS': 10,
    'ETA_MAX': 0.1,
    'ETA_K': 5,
    'ETA_MIN': 0.00001,
    'ANNEAL_FREQUENCY': 20,
    'LAMBDA': 0.005,
    'USE_ALL_DATA': True,
    'USE_DATA_AUGMENTATION': False,
    'DROP_RATE': 0,
    'HIDDENLAYERS': [50, 30, 20, 20, 10, 10, 10, 10]}

# check_gradients(hps)
normal_program_execution(hps)  # after 3rd gs
