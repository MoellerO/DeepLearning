import numpy as np
import matplotlib.pyplot as plt
import sys


def read_data(filename):
    with open(filename, 'r') as f:
        l = f.read()
    return l


def unique_characters(string):
    return list(set(string))


def data_handling(filename='Datasets/goblet_book.txt'):
    book_data = read_data(filename)
    book_chars = unique_characters(book_data)
    ind = [i for i in range(len(book_chars))]
    char_to_int = dict(zip(book_chars, ind))
    int_to_char = dict(zip(ind, book_chars))
    # int_to_char = {key: value for (key, value) in enumerate(book_chars)}
    # char_to_int = {value: key for (key, value) in enumerate(book_chars)}
    return book_data, int_to_char, char_to_int


class RNN():
    m = None
    K = None
    eta = None
    seq_length = None
    b = None  # bias vector (mx1)
    c = None  # bias vector (Kx1)
    U = None  # weights (mxK)
    W = None  # weights (m,m)
    V = None  # weights (K,m)
    G_V = None
    G_W = None
    G_U = None
    G_b = None
    G_c = None
    h_0 = None  # initilize hidden layer
    h_list = None
    a_list = None
    p_list = None
    loss = None
    book_data = None
    int_to_char = None
    char_to_int = None

    def __init__(self, hyperparams):
        self.m = hyperparams['m']
        self.eta = hyperparams['eta']
        self.seq_length = hyperparams['seq_length']
        self.book_data, self.int_to_char, self.char_to_int = data_handling()
        self.K = len(self.int_to_char)
        self.h_0 = np.zeros(self.m)[np.newaxis].T
        self.init_bias_vectors()
        self.init_weights(0, 0.01)

    def init_bias_vectors(self):
        self.b = np.zeros(self.m)[np.newaxis].T
        self.c = np.zeros(self.K)[np.newaxis].T

    def init_weights(self, mu, sigma):
        self.U = np.random.normal(0, 1, (self.m, self.K)) * sigma
        self.W = np.random.normal(0, 1, (self.m, self.m))*sigma
        self.V = np.random.normal(0, 1, (self.K, self.m))*sigma
        # self.U = np.random.normal(mu, sigma, size=(self.m, self.K))
        # self.W = np.random.normal(mu, sigma, size=(self.m, self.m))
        # self.V = np.random.normal(mu, sigma, size=(self.K, self.m))

    def create_one_hot_from_vector(self, x_char):
        '''Create one-hot vector'''
        true_class = self.char_to_int[x_char]
        one_hot = np.zeros(self.K)
        one_hot[true_class] = 1
        return one_hot

    def create_one_hot_matrix(self, x_string):
        '''Create one-hot matrix'''
        first_column = self.create_one_hot_from_vector(x_string[0])
        matrix = first_column
        for char in x_string[1:]:
            column = self.create_one_hot_from_vector(char)
            matrix = np.c_[matrix, column]
        return matrix

    def synthesize(self, h_0, x, n):
        output = ""
        hprev = h_0
        for i in range(n):
            x_t = self.create_one_hot_from_vector(x)[np.newaxis].T
            a_t = np.matmul(self.W, hprev) + \
                np.matmul(self.U, x_t) + self.b
            h_t = np.tanh(a_t)
            hprev = h_t
            o_t = np.matmul(self.V, h_t) + self.c
            # softmax
            denominator = np.sum(np.exp(o_t), axis=0)
            p_t = np.exp(o_t)/denominator
            # randomly select a character
            cp = np.cumsum(p_t)
            a = np.random.rand()
            ii = np.where(cp-a > 0)[0][0]
            x = self.int_to_char[ii]
            output += x

        return output

    def forward(self, x_matrix, y_matrix):
        n = len(x_matrix[0])
        hidden_states = [self.h_0]
        probs = []  # TODO: maybe change to numpy array
        a_list = []
        loss = 0
        for t in range(n):
            x_t = x_matrix[:, t][np.newaxis].T
            a_t = np.matmul(self.W, hidden_states[-1]) + \
                np.matmul(self.U, x_t) + self.b
            h_t = np.tanh(a_t)
            o_t = np.matmul(self.V, h_t) + self.c
            # softmax
            denominator = np.sum(np.exp(o_t), axis=0)
            p_t = np.exp(o_t)/denominator
            # print(y_matrix[:, t][np.newaxis].shape)
            l_t = -(np.matmul(y_matrix[:, t][np.newaxis], np.log(p_t)))
            loss += l_t.item()
            hidden_states.append(h_t)
            probs.append(p_t)
            a_list.append(a_t)
        self.h_0 = hidden_states[-1]
        return loss, probs, hidden_states, a_list

    def backward(self, x_matrix, y_matrix, p_list, h_list, a_list):
        # gradient V
        n = len(y_matrix[0])
        self.G_V = np.zeros((self.K, self.m))
        self.G_W = np.zeros((self.m, self.m))
        self.G_U = np.zeros((self.m, self.K))
        self.G_b = np.zeros(self.m)[np.newaxis].T
        self.G_c = np.zeros(self.K)[np.newaxis].T
        g_a_sub = np.zeros((1, self.m))  # (g_a_t+1)
        for t in range(n-1, -1, -1):
            x_t = x_matrix[:, t][np.newaxis].T
            y_t = y_matrix[:, t][np.newaxis].T
            p_t = p_list[t]
            h_t = h_list[t+1]
            h_t_prev = h_list[t]  # (h_t_-1)

            a_t = a_list[t]
            g_o_t = (p_t-y_t).T
            G_V_t = np.matmul(g_o_t.T, h_t.T)
            self.G_V += G_V_t
            g_h_t = np.matmul(g_o_t, self.V) + np.matmul(g_a_sub, self.W)

            g_a_t = np.matmul(g_h_t,  np.diag(1-np.tanh(a_t[:, 0])**2))
            # print(g_a_t.T.shape)
            # print(h_t_prev.shape)
            g_a_sub = g_a_t
            G_W_t = np.matmul(g_a_t.T, h_t_prev.T)
            self.G_W += G_W_t

            G_U_t = np.matmul(g_a_t.T, x_t.T)
            self.G_U += G_U_t
            self.G_b += g_a_t.T
            self.G_c += g_o_t.T

    def compute_loss(self, p, y_matrix):
        # Using cross-entropy loss
        n = len(y_matrix[0])
        loss = 0
        for t in range(n):
            l_t = -(np.matmul(y_matrix[:, t][np.newaxis], np.log(p[t])))
            loss += l_t.item()
        return loss

    def compute_single_num(self, x_onehot, y_onehot, value, loss, epsilon=1e-4):
        grad = np.zeros(value.shape)
        for i in range(value.shape[0]):
            for j in range(value.shape[1]):
                value[i][j] += epsilon
                _, p, _, _ = self.forward(x_onehot, y_onehot)
                loss1 = self.compute_loss(p, y_onehot)
                grad[i][j] = (loss1 - loss) / epsilon
                value[i][j] -= epsilon
        return grad

    def compute_grad_numerical(self, x_onehot, y_onehot, ):
        """ Compute the gradients numerically """
        h0 = np.zeros((self.m, 1))
        _, p, _, _ = self.forward(x_onehot, y_onehot)
        loss = self.compute_loss(p, y_onehot)
        grad_numerical = {}

        grad_numerical['b'] = self.compute_single_num(
            x_onehot, y_onehot, self.b, loss)
        print("[Computed b Gradients]")

        grad_numerical['c'] = self.compute_single_num(
            x_onehot, y_onehot, self.c, loss)
        print("[Computed c Gradients]")

        grad_numerical['W'] = self.compute_single_num(
            x_onehot, y_onehot, self.W, loss)
        print("[Computed W Gradients]")

        grad_numerical['V'] = self.compute_single_num(
            x_onehot, y_onehot, self.V, loss)
        print("[Computed V Gradients]")

        grad_numerical['U'] = self.compute_single_num(
            x_onehot, y_onehot, self.U, loss)
        print("[Computed V Gradients]")

        return grad_numerical

    def compute_relative_error(self, name, ana_gw, num_gw):
        rel_error = np.absolute(
            ana_gw-num_gw) / np.maximum(1e-6, np.absolute(ana_gw) + np.absolute(num_gw))
        print(f'\n{name}', ':')
        print(rel_error)
        print(rel_error.max())

    def check_gradients(self, X_matrix, Y_matrix):
        grad_numerical = self.compute_grad_numerical(X_matrix, Y_matrix)
        loss, probs, hidden_states, a_list = self.forward(X_matrix, Y_matrix)
        self.backward(X_matrix, Y_matrix, probs, hidden_states, a_list)

        # relative error
        # self.compute_relative_error(
        #     'b', self.G_b, grad_numerical['b'])
        # self.compute_relative_error(
        #     'c', self.G_c, grad_numerical['c'])
        self.compute_relative_error(
            'W', self.G_W, grad_numerical['W'])
        self.compute_relative_error(
            'V', self.G_V, grad_numerical['V'])
        self.compute_relative_error(
            'U', self.G_U, grad_numerical['U'])

    def train(self, hyperparams):
        m_W = np.zeros((self.m, self.m))
        m_V = np.zeros((self.K, self.m))
        m_U = np.zeros((self.m, self.K))
        m_b = np.zeros(self.m)[np.newaxis].T
        m_c = np.zeros(self.K)[np.newaxis].T
        eps = hyperparams['eps']
        smooth_loss = 0
        epoch = 0
        e = 0
        iteration = 0
        losses = []
        while epoch < hyperparams['epochs']:
            if iteration == 0 or e >= (len(self.book_data) - self.seq_length - 1):
                self.h_0 = np.zeros(self.m)[np.newaxis].T
                e = 0
                epoch += 1

            X_chars = self.book_data[e:e+self.seq_length-1]
            Y_chars = self.book_data[e+1:e+self.seq_length]
            X_matrix = self.create_one_hot_matrix(X_chars)
            Y_matrix = self.create_one_hot_matrix(Y_chars)
            loss, probs, hidden_states, a_list = self.forward(
                X_matrix, Y_matrix)
            if epoch == 1 and iteration == 0:
                smooth_loss = loss
            else:
                smooth_loss = 0.999*smooth_loss + 0.001*loss

            if iteration % 100 == 0:
                print('Epoch: %i, iter: %i, Loss: %.2f' %
                      (epoch, iteration, np.round(smooth_loss, 2)))
                losses.append(smooth_loss)

            if iteration % 500 == 0 and epoch < 5:
                txt = self.synthesize(self.h_0, X_chars[0], 200)
                print('\nEpoch: %i, %i iters):\n %s\n' %
                      (epoch, iteration, txt))

            if iteration % 1000 == 0 and epoch >= 5:
                txt = self.synthesize(self.h_0, X_chars[0], 1000)
                print('\nEpoch: %i, %i iters):\n %s\n' %
                      (epoch, iteration, txt))

            self.backward(X_matrix, Y_matrix, probs, hidden_states, a_list)
            m_W += np.square(self.G_W)
            m_V += np.square(self.G_V)
            m_U += np.square(self.G_U)
            m_b += np.square(self.G_b)
            m_c += np.square(self.G_c)

            self.W = self.W - \
                np.multiply(np.multiply(self.eta, np.reciprocal(
                    np.sqrt(np.add(m_W, eps)))), self.G_W)
            self.V = self.V - \
                np.multiply(np.multiply(self.eta, np.reciprocal(
                    np.sqrt(np.add(m_V, eps)))), self.G_V)
            self.U = self.U - \
                np.multiply(np.multiply(self.eta, np.reciprocal(
                    np.sqrt(np.add(m_U, eps)))), self.G_U)
            self.b = self.b - \
                np.multiply(np.multiply(self.eta, np.reciprocal(
                    np.sqrt(np.add(m_b, eps)))), self.G_b)
            self.c = self.c - \
                np.multiply(np.multiply(self.eta, np.reciprocal(
                    np.sqrt(np.add(m_c, eps)))), self.G_c)

            e += self.seq_length
            iteration += 1
        self.plot_overview(losses)

    def plot_overview(self, losses):
        """Plot smooth loss
        """
        plt.plot(losses, label='Smooth loss', color='blue')
        plt.legend(loc="upper left")
        plt.show()


def main():

    sys.stdout = open('output2.txt', 'w')

    hyperparams = {
        'm': 100,  # dimensionality of hidden state
        'eta': 0.1,  # learning rate
        'seq_length': 25,  # input sequence used for training
        'epochs': 10,
        'eps': 1e-8,
    }
    rnn = RNN(hyperparams)
    # X_chars = rnn.book_data[0:hyperparams['seq_length']]
    # Y_chars = rnn.book_data[1:hyperparams['seq_length']+1]
    # # rnn.synthesize(rnn.h0, ".", 1)
    # X_matrix = rnn.create_one_hot_matrix(X_chars)
    # Y_matrix = rnn.create_one_hot_matrix(Y_chars)
    # loss, probs, hidden_states, a_list = rnn.forward(X_matrix, Y_matrix)
    # rnn.backward(X_matrix, Y_matrix, probs, hidden_states, a_list)

    # rnn.check_gradients(X_matrix, Y_matrix)
    rnn.train(hyperparams)

    sys.stdout.close()


main()
