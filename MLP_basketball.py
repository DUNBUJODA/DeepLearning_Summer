import numpy as np
import activators
from MLP_utils_from_theano import *
import configparser
from keras.utils import to_categorical
import time
from progressbar import *


def load_mnist_data(dataset='mnist.pkl.gz'):
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)

    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = np.asarray(data_x, dtype=np.float)
        shared_y = np.asarray(data_y, dtype=np.int)
        return shared_x, shared_y

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    train_set_y = to_categorical(train_set_y)
    valid_set_y = to_categorical(valid_set_y)
    test_set_y = to_categorical(test_set_y)

    return train_set_x.T, train_set_y.T, valid_set_x.T, valid_set_y.T, test_set_x.T, test_set_y.T


def load_basketball_data(dir='v3_data_cur/'):
    with open(dir + 'X_train.pkl', 'rb') as f:
        train_set_x = pickle.load(f)
        train_set_x = train_set_x/255
    with open(dir + 'Y_train.pkl', 'rb') as f:
        train_set_y = pickle.load(f)
        # train_set_y = train_set_y.reshape(-1, train_set_y.shape[0])
    with open(dir + 'X_test.pkl', 'rb') as f:
        test_set_x = pickle.load(f)
        test_set_x = test_set_x/255
    with open(dir + 'Y_test.pkl', 'rb') as f:
        test_set_y = pickle.load(f)
        # test_set_y = test_set_y.reshape(-1, test_set_y.shape[0])
    with open(dir + 'X_val.pkl', 'rb') as f:
        valid_set_x = pickle.load(f)
        valid_set_x = valid_set_x/255
    with open(dir + 'Y_val.pkl', 'rb') as f:
        valid_set_y = pickle.load(f)
        # valid_set_y = valid_set_y.reshape(-1, valid_set_y.shape[0])

    train_set_y = to_categorical(train_set_y)
    valid_set_y = to_categorical(valid_set_y)
    test_set_y = to_categorical(test_set_y)

    return train_set_x.T, train_set_y.T, \
           valid_set_x.T, valid_set_y.T, \
           test_set_x.T, test_set_y.T


def load_basketball_2_data(dir='v3_data_cz/'):
    with open(dir + 'X_train.pkl', 'rb') as f:
        train_set_x = pickle.load(f)
    with open(dir + 'Y_train.pkl', 'rb') as f:
        train_set_y = pickle.load(f)
        # train_set_y = train_set_y.reshape(-1, train_set_y.shape[0])
    with open(dir + 'X_test.pkl', 'rb') as f:
        test_set_x = pickle.load(f)
    with open(dir + 'Y_test.pkl', 'rb') as f:
        test_set_y = pickle.load(f)
        # test_set_y = test_set_y.reshape(-1, test_set_y.shape[0])
    with open(dir + 'X_valid.pkl', 'rb') as f:
        valid_set_x = pickle.load(f)
    with open(dir + 'Y_valid.pkl', 'rb') as f:
        valid_set_y = pickle.load(f)
        # valid_set_y = valid_set_y.reshape(-1, valid_set_y.shape[0])

    train_set_y = to_categorical(train_set_y)
    valid_set_y = to_categorical(valid_set_y)
    test_set_y = to_categorical(test_set_y)

    return train_set_x.T, train_set_y.T, \
           valid_set_x.T, valid_set_y.T, \
           test_set_x.T, test_set_y.T


class HiddenLayer(object): # hidden layers

    def __init__(self, input_size, output_size, learning_rate, activator, L2_reg, rng: np.random.RandomState = None):
        self.input_size = input_size
        self.output_size = output_size

        if rng is None:
            rng = np.random.RandomState(int(time.time()))

        if activator == 'tanh':
            self.activator = activators.TanhActivator()
            self.W = np.asarray(rng.uniform(
                low=-np.sqrt(6. / (input_size + output_size)),
                high=np.sqrt(6. / (input_size + output_size)),
                size=(input_size, output_size)
            ), dtype=np.float)
            self.W = self.W.T
        elif activator == 'sigmoid':
            self.activator = activators.SigmoidActivator()
            self.W = np.asarray(rng.uniform(
                low=-np.sqrt(6. / (input_size + output_size)) * 4,
                high=np.sqrt(6. / (input_size + output_size)) * 4,
                size=(output_size, input_size)
            ), dtype=np.float)

        self.b = np.zeros((output_size, 1))
        self.A = np.zeros((output_size, 1))
        self.X = 0
        self.Z = 0
        self.dW = 0
        self.db = 0
        self.learning_rate = learning_rate
        self.L2_reg = L2_reg

    def forward(self, X):
        self.X = X
        self.Z = np.dot(self.W, X) + self.b
        self.A = self.activator.forward(self.Z)
        return self.A

    def backward(self, W_lplus1, delta_lplus1):
        # if self.isOL:
        #     self.delta = Y - self.A
        # else:
        #     self.delta = np.dot(self.W.T, delta_lplus1) * self.activator.backward(self.A)
        delta = np.dot(W_lplus1.T, delta_lplus1) * self.activator.backward(self.A)
        self.dW = np.dot(delta, self.X.T) / self.X.shape[1]
        self.db = np.mean(delta, axis=1)[:, np.newaxis]
        return self.W, delta

    def upgrade(self):
        self.W = self.W - self.learning_rate * (self.dW + self.L2_reg * self.W)
        self.b = self.b - self.learning_rate * self.db


class OutputLayer():

    def __init__(self, input_size, output_size, learning_rate, L2_reg, rng: np.random.RandomState = None):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        # self.W = np.asarray(rng.uniform(
        #     low=-np.sqrt(6. / (input_size + output_size)),
        #     high=np.sqrt(6. / (input_size + output_size)),
        #     size=(output_size, input_size)
        # ), dtype=np.float)
        self.W = np.zeros((output_size, input_size))
        self.b = np.zeros((output_size, 1))
        self.A = np.zeros((output_size, 1))
        self.X = 0
        self.Z = 0
        self.L2_reg = L2_reg

    def softmax(self, pred):  # normalization for prediction
        mean = np.sum(np.exp(pred), axis=0)
        mean = mean[np.newaxis, :]
        A = np.exp(pred) / mean
        return A

    def forward(self, X):
        self.X = X
        self.Z = np.dot(self.W, X) + self.b
        self.A = self.softmax(self.Z)
        return self.A

    def backward(self, Y):
        delta = self.A - Y
        # delta = np.dot(self.W.T, delta)
        self.dW = np.dot(delta, self.X.T) / self.X.shape[1]
        self.db = np.mean(delta, axis=1)[:, np.newaxis]
        self.upgrade()
        return self.W, delta

    def upgrade(self):
        self.W = self.W - self.learning_rate * (self.dW + self.L2_reg * self.W)
        self.b = self.b - self.learning_rate * self.db


class MLP(object):

    def __init__(self, input_size: int, hidden_layers: list, output_size: int):
        config_path = 'config/'
        cf = configparser.ConfigParser()
        cf.read(config_path + "config_basketball.ini")
        self.learning_rate = cf.getfloat("parameters", "learning_rate")
        self.batch_size = cf.getint("parameters", "batch_size")
        self.line_search = cf.getboolean("parameters", "line_search")
        self.weight_decay = cf.getboolean("parameters", "weight_decay")
        self.early_stop = cf.getboolean("parameters", "early_stop")
        self.epochs = cf.getint("parameters", "epochs")
        self.L2_reg = cf.getfloat("parameters", "l2_reg")
        self.hiddens = hidden_layers
        self.count_cost = False
        ######## validation set
        self.patience = 100000  # look as this many examples regardless
        self.patience_increase = 2  # wait this much longer when a new best is
        # found
        self.improvement_threshold = 0.995  # a relative improvement of this much is
        # considered significant
        self.validation_frequency = 0
        self.best_validation_loss = np.inf  # 无穷大
        ##############################################
        self.hiddenLayers = []
        l = len(hidden_layers)
        for i in range(l):
            if i == 0:
                self.hiddenLayers.append(HiddenLayer(
                    input_size=input_size,
                    output_size=hidden_layers[i],
                    learning_rate=self.learning_rate,
                    activator="tanh",
                    L2_reg=self.L2_reg,
                    rng=np.random.RandomState(1234))
                )
            else:
                self.hiddenLayers.append(HiddenLayer(
                    input_size=hidden_layers[i - 1],
                    output_size=hidden_layers[i],
                    learning_rate=self.learning_rate,
                    activator="tanh",
                    L2_reg=self.L2_reg,
                    rng=np.random.RandomState(1234))
                )
        self.outputLayer = OutputLayer(
            input_size=hidden_layers[l - 1],
            output_size=output_size,
            learning_rate=self.learning_rate,
            L2_reg=self.L2_reg,
        )

    def propagate(self, X, Y, flag):
        if flag == 1:  # forward propagate
            input = X
            reg = 0.0
            for layer in self.hiddenLayers:
                reg = reg + (layer.W ** 2).sum()
                output = layer.forward(input)
                input = output
            A = self.outputLayer.forward(input)
            if self.count_cost:
                if self.weight_decay:
                    cost = - np.mean(np.min(Y * np.log(A), axis=0)) + self.L2_reg / 2 * reg
                else:
                    cost = - np.mean(np.min(Y * np.log(A), axis=0))
                return cost
        elif flag == 3:
            input = X
            output = 0
            for layer in self.hiddenLayers:
                output = layer.forward(input)
                input = output
            A = self.outputLayer.forward(output)
            Y_predict = np.argmax(A, axis=0)
            return Y_predict
        elif flag == 4:
            input = X
            output = 0
            for layer in self.hiddenLayers:
                output = layer.forward(input)
                input = output
            A = self.outputLayer.forward(output)
            probability = A[1, :]
            return probability

    def backpropagate(self, X, Y):
        layer = self.outputLayer
        W, delta = layer.backward(Y)
        for layer in self.hiddenLayers[::-1]:
            W, delta = layer.backward(W, delta)
            layer.upgrade()

    def train(self, X_train, Y_train, X_val, Y_val, X_test, Y_test):
        batch_num = X_train.shape[1] // self.batch_size

        for l in range(self.epochs):
            for i in range(batch_num):
                minibatch_X = X_train[:, i * self.batch_size:(i + 1) * self.batch_size]
                minibatch_Y = Y_train[:, i * self.batch_size:(i + 1) * self.batch_size]
                cost = 0.0
                if self.count_cost:
                    cost = self.propagate(minibatch_X, minibatch_Y, 1)
                else:
                    self.propagate(minibatch_X, minibatch_Y, 1)
                self.backpropagate(minibatch_X, minibatch_Y)

                iter = l * batch_num + i
                if self.early_stop:
                    self.validation_frequency = min(batch_num, self.patience // 2)
                    if (iter + 1) % self.validation_frequency == 0:
                        predict = self.propagate(X_val, Y_val, 3)
                        Y_predict = np.argmax(Y_val, axis=0)
                        this_validation_loss = np.nonzero(predict - Y_predict)[0].shape[0] / X_val.shape[1]
                        if self.count_cost:
                            print(
                                'epoch %i, minibatch %i/%i, patience %i, train loss %f, validation error %f %%' %
                                (
                                    l,
                                    i,
                                    batch_num,
                                    self.patience,
                                    cost,
                                    this_validation_loss * 100
                                )
                            )
                        else:
                            print(
                                'epoch %i, minibatch %i/%i, patience %i, validation error %f %%' %
                                (
                                    l,
                                    i,
                                    batch_num,
                                    self.patience,
                                    this_validation_loss * 100
                                )
                            )

                        # if we got the best validation score until now
                        if this_validation_loss < self.best_validation_loss:
                            # improve patience if loss improvement is good enough
                            if this_validation_loss < self.best_validation_loss * self.improvement_threshold:
                                self.patience = max(self.patience, iter * self.patience_increase)

                                predict = self.propagate(X_test, Y_test, 3)
                                Y_predict = np.argmax(Y_test, axis=0)
                                test_loss = np.nonzero(predict - Y_predict)[0].shape[0] / X_test.shape[1]
                                print(
                                    '     epoch %i, minibatch %i/%i, test error of best model %f %%' %
                                    (
                                        l,
                                        i,
                                        batch_num,
                                        test_loss * 100.
                                    )
                                )
                            self.best_validation_loss = this_validation_loss
                            self.save_weights()
                else:
                    print("epoch %i, minibatch %i, train_loss %f" % (l, i, cost))
                if self.patience <= iter:
                    break
                if self.best_validation_loss < 1e-5:
                    break

            lastbatch = X_train.shape[1] % self.batch_size
            if lastbatch != 0:
                lastbatch_X = X_train[:, batch_num * self.batch_size:batch_num * self.batch_size + lastbatch]
                lastbatch_Y = Y_train[:, batch_num * self.batch_size:batch_num * self.batch_size + lastbatch]
                cost = self.propagate(lastbatch_X, lastbatch_Y, 1)
                self.backpropagate(lastbatch_X, lastbatch_Y)

                if self.early_stop:
                    iter = l * batch_num + batch_num * self.batch_size
                    self.validation_frequency = min(batch_num, self.patience // 2)
                    if iter % self.validation_frequency == 0:
                        predict = self.propagate(X_val, Y_val, 3)
                        Y_predict = np.argmax(Y_val, axis=0)
                        this_validation_loss = np.nonzero(predict - Y_predict)[0].shape[0] / X_val.shape[1]
                        if self.count_cost:
                            print(
                                'epoch %i, minibatch %i/%i, patienct %i, train loss %f, validation error %f %%' %
                                (
                                    l,
                                    batch_num,
                                    batch_num,
                                    self.patience,
                                    cost,
                                    this_validation_loss * 100.
                                )
                            )
                        else:
                            print(
                                'epoch %i, minibatch %i/%i, patience %i, validation error %f %%' %
                                (
                                    l,
                                    batch_num,
                                    batch_num,
                                    self.patience,
                                    this_validation_loss * 100.
                                )
                            )

                        # if we got the best validation score until now
                        if this_validation_loss < self.best_validation_loss:
                            # improve patience if loss improvement is good enough
                            if this_validation_loss < self.best_validation_loss * self.improvement_threshold:
                                self.patience = max(self.patience, iter * self.patience_increase)
                                predict = self.propagate(X_test, Y_test, 3)
                                Y_predict = np.argmax(Y_test, axis=0)
                                test_loss = np.nonzero(predict - Y_predict)[0].shape[0] / X_test.shape[1]
                                print(
                                    '     epoch %i, minibatch %i/%i, test error of best model %f %%' %
                                    (
                                        l,
                                        batch_num,
                                        batch_num,
                                        test_loss * 100.
                                    )
                                )
                            self.best_validation_loss = this_validation_loss

                        if self.patience <= iter:
                            break
                else:
                    print("epoch %i, minibatch %i, train_loss %f" % (l, batch_num, cost))
                if self.best_validation_loss < 1e-5:
                    break

    def save_weights(self):
        dir = 'model/'
        parameters = self.learning_rate, self.batch_size, self.line_search, \
                     self.weight_decay, self.early_stop, self.early_stop, \
                     self.epochs, self.L2_reg
        with open(dir + '_pm_' + 'model_hls' + "+".join(list(map(str, self.hiddens))), 'wb') as f:
            pickle.dump(parameters, f)
        weights = self.hiddenLayers, self.outputLayer
        with open(dir + '_wt_' + 'model_hls' + "+".join(list(map(str, self.hiddens))), 'wb') as f:
            pickle.dump(weights, f)

    def load_weights(self):
        dir = 'model/'
        with open(dir + '_wt_' + 'model_hls' + "+".join(list(map(str, self.hiddens))), 'rb') as f:
            weights = pickle.load(f)
        self.hiddenLayers, self.outputLayer = weights

    def propagate_ROC(self, thr, threshold, k):
        Y_predict = np.zeros(thr.shape)
        for i in range(len(thr)):
            if thr[i] >= threshold[k]:
                Y_predict[i] = 1
            else:
                Y_predict[i] = 0
        return Y_predict.reshape(-1, Y_predict.shape[0])

    def draw_and_save_ROC(self, X, Y):
        self.load_weights()
        thr = self.propagate(X, Y, 4)
        threshold = np.sort(thr)
        Y = np.argmax(Y, axis=0)
        Y = Y.reshape(-1, Y.shape[0])
        recall = np.zeros(Y.shape)
        FAR = np.zeros(Y.shape)

        progress = ProgressBar(Y.shape[1])
        progress.start()
        for k in range(threshold.shape[0]):
            Y_predict = self.propagate_ROC(thr, threshold, k)  # (1,m_test)
            TP = 0
            TN = 0
            FP = 0
            FN = 0
            for i in range(Y.shape[1]):
                if Y_predict[0, i] - Y[0, i] == 0 and Y[0, i] == 1:
                    TP = TP + 1
                elif Y_predict[0, i] - Y[0, i] == 0 and Y[0, i] == 0:
                    TN = TN + 1
                elif Y_predict[0, i] - Y[0, i] == 1:
                    FP = FP + 1
                elif Y_predict[0, i] - Y[0, i] == -1:
                    FN = FN + 1
                else:
                    pass
            recall[0, k] = TP / (TP + FN)
            FAR[0, k] = FP / (FP + TN)
            progress.show_progress(k)
        progress.end()
        ######################save ROC
        dir = 'ROCcurve/'
        cuv = FAR, recall
        with open(dir + '_roc_' + 'model_hls' + "+".join(list(map(str, self.hiddens))) + '.pkl', 'wb') as f:
            pickle.dump(cuv, f)


train_set_x, train_set_y, valid_set_x, valid_set_y, test_set_x, test_set_y = load_basketball_2_data()
model = MLP(input_size=train_set_x.shape[0], hidden_layers=[20, 8], output_size=2)
# model.train(train_set_x, train_set_y, valid_set_x, valid_set_y, test_set_x, test_set_y)
model.draw_and_save_ROC(test_set_x, test_set_y)


