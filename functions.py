import numpy as np
import pickle


# data handling
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


def load_batch(filename):
    """ Copied from the dataset website"""
    with open('Datasets/'+filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    X = np.transpose(dict[b'data'])
    y = dict[b'labels']
    Y = create_one_hot(y, 10)
    return X, Y, y


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


def create_one_hot(y, K):
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


def init_weights(mu, sigma):
    W = np.random.normal(mu, sigma, size=(K, D))
    return W


def init_bias(mu, sigma):
    b = np.random.normal(mu, sigma, size=K)[np.newaxis]
    return np.transpose(b)


def shuffle_data(features, targets_one_hot, targets):
    assert features.shape[1] == targets_one_hot.shape[1]
    p = np.random.permutation(features.shape[1])
    return features[:, p], targets_one_hot[:, p], [targets[i] for i in p]
