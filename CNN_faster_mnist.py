import numpy as np
import pickle
import gzip
from activators import *
import configparser
from MLP import *
from progressbar import *
from scipy.signal import correlate2d, convolve2d
from scipy import fftpack


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

    test_set_x = test_set_x.reshape((test_set_x.shape[0], 1, 28, 28))
    valid_set_x = valid_set_x.reshape((valid_set_x.shape[0], 1, 28, 28))
    train_set_x = train_set_x.reshape((train_set_x.shape[0], 1, 28, 28))

    train_y_cato = np.zeros((10, train_set_y.shape[0]))
    train_y_cato[train_set_y, np.arange(train_set_y.shape[0])] = 1.0
    test_y_cato = np.zeros((10, test_set_y.shape[0]))
    test_y_cato[test_set_y, np.arange(test_set_y.shape[0])] = 1.0
    valid_y_cato = np.zeros((10, valid_set_y.shape[0]))
    valid_y_cato[valid_set_y, np.arange(valid_set_y.shape[0])] = 1.0

    return train_set_x, train_y_cato, valid_set_x, valid_y_cato, test_set_x, test_y_cato


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

    # train_set_y = to_categorical(train_set_y)
    # valid_set_y = to_categorical(valid_set_y)
    # test_set_y = to_categorical(test_set_y)

    return train_set_x.T, train_set_y.T, \
           valid_set_x.T, valid_set_y.T, \
           test_set_x.T, test_set_y.T


def build_model():
        config_path = 'config/'
        cf = configparser.ConfigParser()
        cf.read(config_path + "config_mnist.ini")
        learning_rate = cf.getfloat("parameters", "learning_rate")
        batch_size = cf.getint("parameters", "batch_size")
        line_search = cf.getboolean("parameters", "line_search")
        weight_decay = cf.getboolean("parameters", "weight_decay")
        early_stop = cf.getboolean("parameters", "early_stop")
        epochs = cf.getint("parameters", "epochs")
        L2_reg = cf.getfloat("parameters", "l2_reg")

        rng = np.random.RandomState(1234)
        convs = []
        hiddens = []
        conv1 = LeNetConvPoolLayer(
            rng=rng,
            input_shape=(batch_size, 1, 28, 28),
            filter_shape=(20, 1, 5, 5),
            f_strides=(1, 1),
            pooling='max',
            pooling_shape=(2, 2),
            p_strides=(2, 2),
            activator='tanh',
            padding='valid',
            learning_rate=learning_rate,
            ignore_border=True,
            name='conv1',
            L2_reg=L2_reg
        )
        convs.append(conv1)
        conv2 = LeNetConvPoolLayer(
            rng=rng,
            input_shape=conv1.pool_output_shape,
            filter_shape=(50, 20, 5, 5),
            f_strides=(1, 1),
            pooling='max',
            pooling_shape=(2, 2),
            p_strides=(2, 2),
            activator='tanh',
            padding='valid',
            learning_rate=learning_rate,
            ignore_border=True,
            name='conv2',
            L2_reg=L2_reg
        )
        convs.append(conv2)
        hid1 = HiddenLayer(
            input_size=np.prod(conv2.pool_output_shape[1:]),
            output_size=500,
            activator='tanh',
            rng=rng,
            learning_rate=learning_rate,
            L2_reg=L2_reg,
            name='hidden'
        )
        hiddens.append(hid1)
        op = OutputLayer(
            input_size=500,
            output_size=10,
            learning_rate=learning_rate,
            L2_reg=L2_reg,
            name='output'
        )
        return convs, hiddens, op, batch_size, epochs


class LeNetConvPoolLayer(object):

    def __init__(self, rng, input_shape, filter_shape, f_strides, padding,
                 pooling, pooling_shape, p_strides, ignore_border,
                 activator, learning_rate, name: str, L2_reg: float):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        (???????)
        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps->input图像的通道数,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)

        :type padding: string='valid' or 'same'

        :type pooling: string='max' or 'mean'
        """

        self.input_shape = input_shape
        self.filter_shape = filter_shape
        self.pooling_shape = pooling_shape
        self.f_strides = f_strides
        self.pooling = pooling
        self.p_strides = p_strides
        self.learning_rate = learning_rate
        self.ignore_border = ignore_border
        self.name = name
        self.L2_reg = L2_reg

        fan_in = np.prod(filter_shape[1:])  # num_of_channels*filter_height*filter_width
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) //  # num_of_filters*filter_height*filter_width//(pooling_height*pooling_width)
                   np.prod(pooling_shape))
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        if activator == 'tanh':
            self.activator = TanhActivator()
            self.W = np.asarray(
                    rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                    dtype=np.float
                )
            self.dW = np.zeros(self.W.shape)

        self.b = np.zeros(filter_shape[0], dtype=np.float)
        self.db = np.zeros(self.b.shape)
        self.padding = padding

        if padding == 'valid':
            h = int((input_shape[2] - filter_shape[2]) / f_strides[0]) + 1
            w = int((input_shape[3] - filter_shape[3]) / f_strides[1]) + 1
            self.corr_output_shape = input_shape[0], filter_shape[0], h, w
        elif padding == 'same':
            ## pass
            h = int((input_shape[2] + filter_shape[2] - 1) / p_strides[0])
            w = int((input_shape[3] + filter_shape[3] - 1) / p_strides[1])
            self.corr_output_shape = input_shape[0], filter_shape[0], h, w
        else:
            ## pass
            h = 0
            w = 0
            self.corr_output_shape = 0, h, w

        if ignore_border:
            h = int((h - pooling_shape[0]) / p_strides[0]) + 1
            w = int((w - pooling_shape[1]) / p_strides[1]) + 1
        else:
            ## pass
            h = int((h - pooling_shape[0]) / p_strides[0]) + 2
            w = int((w - pooling_shape[1]) / p_strides[1]) + 2
        self.pool_output_shape = input_shape[0], filter_shape[0], h, w

        self.A_corr = np.zeros(self.corr_output_shape)
        # self.dZ = np.zeros(self.corr_output_shape)
        self.A_pool = np.zeros(self.pool_output_shape)
        self.X = None
        self.id_max = None

    # def corr2d(self, X, k):  # cross correlation
    #     W = self.W[k]
    #     b = self.b[k]
    #     z0 = np.sum(X*W) + b
    #     return z0
    def corr2d(self, X, W):  # cross correlation
        z0 = np.sum(X*W)
        return z0

    def split_max_pooling(self):
        rows = int(self.corr_output_shape[2] / self.pooling_shape[0])
        cols = int(self.corr_output_shape[3] / self.pooling_shape[1])
        # instructions: https://blog.csdn.net/summer2day/article/details/79934612
        tmp = np.asarray(np.split(self.A_corr, cols, axis=3))
        tmp = np.asarray(np.split(tmp, rows, axis=3))
        # tmp = tmp.reshape((tmp.shape[2], tmp.shape[3], tmp.shape[1], tmp.shape[0], self.pooling_shape[0], self.pooling_shape[1]))
        return tmp

    def concatenate_max_pooling(self, x):
        # instructions: https://blog.csdn.net/brucewong0516/article/details/79158758
        # axis=i: concatenate the ith dimension of data
        return np.concatenate(np.concatenate(x, axis=3), axis=3)

    def forward_fft(self, X):
        '''
        :param X: shape=(batch_size, num_of_channels, img_height, img_width)
        :return:
        '''
        assert X.shape[1] == self.W.shape[1]
        self.X = X
        W_rotate = self.W[:, :, ::-1][:, :, :, ::-1]

        ############cross corr
        Z = np.zeros(self.corr_output_shape, dtype=np.float)
        partial_filter_shape = X.shape[2:]
        fft_x = np.asarray(fftpack.fft2(X, partial_filter_shape), np.complex64)
        fft_W = np.asarray(fftpack.fft2(W_rotate, partial_filter_shape), np.complex64)
        for i in range(X.shape[0]):
            t = fft_x[i] * fft_W
            tt = np.sum(t, axis=1)
            ttt = np.real(fftpack.ifft2(tt))
            Z[i] = ttt[:, (self.filter_shape[2]-1): X.shape[2], (self.filter_shape[3]-1): X.shape[3]]
        self.A_corr = Z

        ###########pooling layer
        tmp = self.split_max_pooling()
        tmp1 = tmp.reshape(
            (tmp.shape[0], tmp.shape[1], tmp.shape[2], tmp.shape[3], self.pooling_shape[0] * self.pooling_shape[1]))
        tmp2 = tmp1.max(axis=4)
        self.id_max = tmp1.argmax(axis=4)
        tmp3 = tmp2.reshape((tmp2.shape[0], tmp2.shape[1], tmp2.shape[2], tmp2.shape[3], 1, 1))
        A = self.concatenate_max_pooling(tmp3)
        self.A_pool = self.activator.forward(A)
        return self.A_pool

    def forward(self, X):
        '''
        :param X: shape=(batch_size, num_of_channels, img_height, img_width)
        :return:
        '''
        assert X.shape[1] == self.W.shape[1]
        self.X = X

        ############cross corr
        Z = np.zeros(self.corr_output_shape)
        for i in range(X.shape[0]):  # batch size
            for j in range(self.filter_shape[0]):  # num of kernel
                tmp = np.zeros(self.corr_output_shape[2:])
                for c in range(self.filter_shape[1]):  # channel of kernel
                    tmp += correlate2d(X[i, c], self.W[j, c], 'valid')
                Z[i, j] += tmp + self.b[j]
        self.A_corr = Z

        ###########pooling layer
        tmp = self.split_max_pooling()
        tmp1 = tmp.reshape((tmp.shape[0], tmp.shape[1], tmp.shape[2], tmp.shape[3], self.pooling_shape[0] * self.pooling_shape[1]))
        tmp2 = tmp1.max(axis=4)
        self.id_max = tmp1.argmax(axis=4)
        tmp3 = tmp2.reshape((tmp2.shape[0], tmp2.shape[1], tmp2.shape[2], tmp2.shape[3], 1, 1))
        A = self.concatenate_max_pooling(tmp3)
        self.A_pool = self.activator.forward(A)
        return self.A_pool

    def create_mask(self, x):
        mask = (x == np.max(x))
        return mask

    def distribute(self, x, delta):
        ones = np.ones(x.shape)
        mean = np.mean(delta)
        return ones * mean

    def backward_fft(self, para, delta_prev):
        ##########previous layer->pooling layer
        if type(para) == np.ndarray:
            W_prev = para
            delta = np.zeros(self.pool_output_shape, dtype=np.float)
            delta = np.dot(W_prev.T, delta_prev)
            delta = delta.T
            delta_shape = delta.shape[0], self.pool_output_shape[1], self.pool_output_shape[2], self.pool_output_shape[3]
            delta = delta.reshape(delta_shape)
            delta = delta * self.activator.backward(self.A_pool)
        else:
            W_prev, strides_prev = para
            delta = np.zeros(self.pool_output_shape)
            full_filter_shape = self.pool_output_shape[2:]
            fft_W_prev = np.asarray(fftpack.fft2(W_prev, full_filter_shape), dtype=np.complex64)
            fft_delta_prev = np.asarray(fftpack.fft2(delta_prev, full_filter_shape), dtype=np.complex64)
            for i in range(delta_prev.shape[0]):  # ith img
                t = fft_delta_prev[i].reshape(delta_prev.shape[1], 1, full_filter_shape[0], full_filter_shape[1])
                tt = t * fft_W_prev
                ttt = np.sum(tt, axis=0)
                delta[i] = np.asarray(np.real(fftpack.ifft2(ttt)), dtype=float)
            delta = delta * self.activator.backward(self.A_pool)
        ############pooling layer->corr layer
        tt_shape = (int(self.corr_output_shape[2] / self.pooling_shape[0]),
                    int(self.corr_output_shape[3] / self.pooling_shape[1]),
                    self.corr_output_shape[0],
                    self.corr_output_shape[1],
                    self.pooling_shape[0] * self.pooling_shape[1])
        tmp = np.zeros((np.prod(tt_shape[:4]), tt_shape[4]), dtype=np.float)
        t = np.asarray(np.split(delta, delta.shape[3], axis=3))
        tt = np.asarray(np.split(t, delta.shape[2], axis=3))
        ttt = tt.reshape(np.prod(delta.shape), )
        tmp[np.arange(np.prod(tt_shape[:4])), self.id_max.reshape(np.prod(self.id_max.shape))] = ttt
        tmp2 = tmp.reshape((tt_shape[0], tt_shape[1], tt_shape[2], tt_shape[3], self.pooling_shape[0], self.pooling_shape[1]))
        delta_corr = np.zeros(self.corr_output_shape, dtype=np.float)
        delta_corr = self.concatenate_max_pooling(tmp2)
        assert delta_corr.shape == self.corr_output_shape  # delta_corr = self.dZ
        #############corr layer-> dW db
        delta_corr_rotate = delta_corr[:, :, ::-1][:, :, :, ::-1]
        partial_kernel_shape = self.X.shape[2:]
        fft_x = np.asarray(fftpack.fft2(self.X, partial_kernel_shape), dtype=np.complex64)
        fft_delta_corr = np.asarray(fftpack.fft2(delta_corr_rotate, partial_kernel_shape), dtype=np.complex64)
        fft_x_reshape = fft_x.reshape(self.X.shape[0], 1, self.X.shape[1],
                                      partial_kernel_shape[0],
                                      partial_kernel_shape[1])
        fft_delta_conv_reshape = fft_delta_corr.reshape(self.X.shape[0], delta_corr_rotate.shape[1], 1,
                                                        partial_kernel_shape[0],
                                                        partial_kernel_shape[1])
        t = fft_x_reshape * fft_delta_conv_reshape
        tt = np.sum(t, axis=0)
        ttt = np.asarray(np.real(fftpack.ifft2(tt)), dtype=np.float64)
        delta_W = ttt[:, :, (delta_corr_rotate.shape[2] - 1): self.X.shape[2],
                  (delta_corr_rotate.shape[3] - 1): self.X.shape[3]]
        delta_b = np.zeros(self.b.shape, dtype=np.float64)
        t = np.sum(delta_corr_rotate, axis=0)
        t = np.sum(t, axis=1)
        delta_b = np.sum(t, axis=1)
        self.dW = delta_W / delta_corr.shape[0]
        self.db = delta_b / delta_corr.shape[0]
        para = self.W, self.f_strides
        return para, delta_corr

    def backward(self, para, delta_prev):
        ##########previous layer->pooling layer
        if type(para) == np.ndarray:
            W_prev = para
            delta = np.zeros(self.pool_output_shape, dtype=np.float)
            delta = np.dot(W_prev.T, delta_prev)
            delta = delta.T
            delta_shape = delta.shape[0], self.pool_output_shape[1], self.pool_output_shape[2], self.pool_output_shape[3]
            delta = delta.reshape(delta_shape)
            delta = delta * self.activator.backward(self.A_pool)
        else:
            W_prev, strides_prev = para
            delta = np.zeros(self.pool_output_shape)
            for i in range(delta_prev.shape[0]):  # ith img
                for j in range(W_prev.shape[1]):  # jth channel in W_prev = jth channel in self.pooling
                    tmp = np.zeros(self.pool_output_shape[2:])
                    for k in range(delta_prev.shape[1]):  # kth channel in delta_prev = kth kernel
                        tmp += convolve2d(delta_prev[i, k], W_prev[k, j], 'full')
                    delta[i, j] = tmp
            delta = delta * self.activator.backward(self.A_pool)
        ############pooling layer->corr layer
        tt_shape = (int(self.corr_output_shape[2] / self.pooling_shape[0]),
                    int(self.corr_output_shape[3] / self.pooling_shape[1]),
                    self.corr_output_shape[0],
                    self.corr_output_shape[1],
                    self.pooling_shape[0] * self.pooling_shape[1])
        tmp = np.zeros((np.prod(tt_shape[:4]), tt_shape[4]), dtype=np.float)
        t = np.asarray(np.split(delta, delta.shape[3], axis=3))
        tt = np.asarray(np.split(t, delta.shape[2], axis=3))
        ttt = tt.reshape(np.prod(delta.shape), )
        tmp[np.arange(np.prod(tt_shape[:4])), self.id_max.reshape(np.prod(self.id_max.shape))] = ttt
        tmp2 = tmp.reshape((tt_shape[0], tt_shape[1], tt_shape[2], tt_shape[3], self.pooling_shape[0], self.pooling_shape[1]))
        delta_corr = np.zeros(self.corr_output_shape, dtype=np.float)
        delta_corr = self.concatenate_max_pooling(tmp2)
        assert delta_corr.shape == self.corr_output_shape  # delta_corr = self.dZ
        #############corr layer-> dW db
        for i in range(self.X.shape[0]):  # batch size
            for j in range(self.corr_output_shape[1]):  # the jth "kernel"
                for c in range(self.X.shape[1]):  # channel
                    self.dW[j, c] += correlate2d(self.X[i, c], delta_corr[i, j], 'valid')
                self.db[j] += np.sum(delta_corr[i, j])
        self.dW /= delta_corr.shape[0]
        self.db /= delta_corr.shape[0]
        para = self.W, self.f_strides
        return para, delta_corr

    def upgrade(self):
        self.W -= self.learning_rate * (self.L2_reg * self.W + self.dW)
        self.b -= self.learning_rate * self.db


class CNN(object):

    def __init__(self, convs:list, hiddens:list, output:OutputLayer, batch_size, epochs):
        self.learning_rate = 0
        self.batch_size = batch_size
        self.line_search = 0
        self.weight_decay = 0
        self.early_stop = True
        self.epochs = epochs
        self.L2_reg = 0
        ######## validation set
        self.patience = 100000  # look as this many examples regardless
        self.patience_increase = 2  # wait this much longer when a new best is
        # found
        self.improvement_threshold = 0.995  # a relative improvement of this much is
        # considered significant
        self.validation_frequency = 0
        self.best_validation_loss = np.inf
        ##############################################
        self.convLayers = convs
        self.hiddenLayers = hiddens
        self.outputLayer = output

    def propagate(self, X, Y, flag):
        input = X
        for conv in self.convLayers:
            output = conv.forward_fft(input)
            input = output
        # input = input.reshape((-1, input.shape[0]))
        input = input.reshape((input.shape[0], -1)).T
        for hidden in self.hiddenLayers:
            output = hidden.forward(input)
            input = output
        A = self.outputLayer.forward(input)
        if flag == 1:
            pass
        elif flag == 2:
            return - np.mean(np.min(Y * np.log(A), axis=0))
        elif flag == 3:
            return A

    def backpropagate(self, X, Y):
        W, delta = self.outputLayer.backward(Y)
        for hidden in self.hiddenLayers[::-1]:
            W, delta = hidden.backward(W, delta)
        para = W
        for conv in self.convLayers[::-1]:
            para, delta = conv.backward_fft(para, delta)

        for conv in self.convLayers:
            conv.upgrade()
        for hidden in self.hiddenLayers[::-1]:
            hidden.upgrade()
        self.outputLayer.upgrade()

    def train(self, X_train, Y_train, X_val, Y_val):
        batch_num = X_train.shape[0] // self.batch_size

        for l in range(self.epochs):
            for i in range(batch_num):
                minibatch_X = X_train[i * self.batch_size:(i + 1) * self.batch_size]
                minibatch_Y = Y_train[:, i * self.batch_size:(i + 1) * self.batch_size]
                self.propagate(minibatch_X, minibatch_Y, 1)
                self.backpropagate(minibatch_X, minibatch_Y)
                self.save_weights()

                iter = l * batch_num + i
                print('training @ iter = ', iter)
                minibatch_avg_cost = self.propagate(minibatch_X, minibatch_Y, 2)
                print("minibatch_avg_cost" + str(minibatch_avg_cost))
                if self.early_stop:
                    self.validation_frequency = min(batch_num, self.patience // 2)
                    if (iter + 1) % self.validation_frequency == 0:
                        batch_val_num = X_val.shape[0] // self.batch_size
                        predict = np.zeros(Y_val.shape[1])
                        for k in range(batch_val_num):
                            miniX = X_val[k * self.batch_size:(k + 1) * self.batch_size]
                            A = self.propagate(miniX, Y_val, 3)
                            predict[k * self.batch_size:(k + 1) * self.batch_size] = np.argmax(A, axis=0)
                        Y_predict = np.argmax(Y_val, axis=0)
                        this_validation_loss = np.nonzero(predict - Y_predict)[0].shape[0] / X_val.shape[0]
                        print(
                            '       epoch %i, minibatch %i/%i, patience %i, validation error %f %%' %
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
                            self.best_validation_loss = this_validation_loss
                else:
                    pass
                if self.patience <= iter:
                    break
                if self.best_validation_loss < 1e-5:
                    break

    def save_weights(self):
        dir = 'model/'
        weights = self.convLayers, self.hiddenLayers, self.outputLayer
        names = []
        layers = {'convs': self.convLayers, 'hiddens': self.hiddenLayers}
        for k, v in layers.items():
            for l in v:
                names.append(l.name)
        names.append(self.outputLayer.name)
        with open(dir + '_wt_' + 'CNNmodel_' + "+".join(names) + '.pkl', 'wb') as f:
            pickle.dump(weights, f)

    def load_weights(self):
        dir = 'model/'
        names = []
        layers = {'convs': self.convLayers, 'hiddens': self.hiddenLayers}
        for k, v in layers.items():
            for l in v:
                names.append(l.name)
        names.append(self.outputLayer.name)
        with open(dir + '_wt_' + 'CNNmodel_' + "+".join(names) + '.pkl', 'rb') as f:
            weights = pickle.load(f)
        self.convLayers, self.hiddenLayers, self.outputLayer = weights

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
        batch_test_num = X.shape[0] // self.batch_size
        thr = np.zeros(Y.shape[1])

        progress2 = ProgressBar(batch_test_num)
        progress2.start()
        for k in range(batch_test_num):
            miniX = X[k * self.batch_size:(k + 1) * self.batch_size]
            A = self.propagate(miniX, Y, 3)
            thr[k * self.batch_size:(k + 1) * self.batch_size] = np.max(A, axis=0)
            progress2.show_progress(k)
        progress2.end()
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
        names = []
        layers = {'convs': self.convLayers, 'hiddens': self.hiddenLayers}
        for k, v in layers.items():
            for l in v:
                names.append(l.name)
        names.append(self.outputLayer.name)
        with open(dir + '_roc_' + 'model_hls' + "+".join(names) + '.pkl', 'wb') as f:
            pickle.dump(cuv, f)


train_set_x, train_set_y, valid_set_x, valid_set_y, test_set_x, test_set_y = load_mnist_data()
convs, hiddens, op, batch_size, epochs = build_model()
model = CNN(convs=convs, hiddens=hiddens, output=op, batch_size=batch_size, epochs=epochs)
model.train(train_set_x, train_set_y, valid_set_x, valid_set_y)
# model.draw_and_save_ROC(test_set_x, test_set_y)








