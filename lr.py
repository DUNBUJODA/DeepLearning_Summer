from shuffle import *
import pickle
import configparser
from progressbar import *

def lodd_data():
    config_path = 'config/'
    cf = configparser.ConfigParser()
    cf.read(config_path + "lr_cfg.ini")
    conti_frame = cf.getint("parameters", "conti_frame")
    if conti_frame == 1:
        X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, X_val_orig, Y_val_orig = load_data('1frame/')
    elif conti_frame == 2:
        X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, X_val_orig, Y_val_orig = load_data('2frames/')
        # train add
    Xtrain_add = np.zeros((X_train_orig.shape[0], X_train_orig.shape[1] + 1))
    Xtrain_add[:, X_train_orig.shape[1]] = 1
    Xtrain_add[:, 0:X_train_orig.shape[1]] = X_train_orig
    # test add
    Xtest_add = np.zeros((X_test_orig.shape[0], X_test_orig.shape[1] + 1))
    Xtest_add[:, X_train_orig.shape[1]] = 1
    Xtest_add[:, 0:X_train_orig.shape[1]] = X_test_orig
    # val add
    Xval_add = np.zeros((X_val_orig.shape[0], X_val_orig.shape[1] + 1))
    Xval_add[:, X_val_orig.shape[1]] = 1
    Xval_add[:, 0:X_val_orig.shape[1]] = X_val_orig
    ##########
    X_train = Xtrain_add.T
    Y_train = Y_train_orig.T[np.newaxis, :]
    X_test = Xtest_add.T
    Y_test = Y_test_orig.T[np.newaxis, :]
    X_val = Xval_add.T
    Y_val = Y_val_orig.T[np.newaxis, :]
    return X_train, Y_train, X_test, Y_test, X_val, Y_val


class Model:
    def __init__(self, dim):  # binary classifier
        config_path = 'config/'
        cf = configparser.ConfigParser()
        cf.read(config_path + "lr_cfg.ini")
        self.learning_rate = cf.getfloat("parameters", "learning_rate")
        self.batch_size = cf.getint("parameters", "batch_size")
        self.train_prop = cf.getfloat("parameters", "train_prop")
        self.conti_frame = cf.getint("parameters", "conti_frame")
        self.line_search = cf.getboolean("parameters", "line_search")
        self.weight_decay = cf.getboolean("parameters", "weight_decay")
        self.early_stop = cf.getboolean("parameters", "early_stop")
        self.epochs = cf.getint("parameters", "epochs")
        self.lambdaa = cf.getfloat("parameters", "lambdaa")
        ######## validation set
        self.patience = 0  # look as this many examples regardless
        self.patience_increase = 2  # wait this much longer when a new best is
        # found
        self.improvement_threshold = 0.995  # a relative improvement of this much is
        # considered significant
        self.validation_frequency = 0
        self.best_validation_loss = np.inf  # 无穷大
        ##############################################
        self.theta = np.zeros((dim, 1))
        self.ls_theta = self.theta
        # self.threshold = np.linspace(0, 1, 100)
        self.threshold = 0
        self.recall = []
        self.FAR = []
    def sigmoid(self, z):
        s = 1.0 / (1 + np.exp(-z))
        return s
    def propagate(self, X, Y, flag):
        m = X.shape[1]
        A = np.zeros(Y.shape)
        cost = 0
        if flag == 1:
            A = self.sigmoid(np.dot(self.theta.T, X))
            cost = -(1 / m) * np.sum(np.log(A) * Y + np.log(1 - A) * (1 - Y))
            return A, cost
        elif flag == 2:  # validation cost
            A = self.sigmoid(np.dot(self.theta.T, X))
            cost = -(1 / m) * np.sum(np.log(A) * Y + np.log(1 - A) * (1 - Y))
            return cost
        elif flag == 3:
            A = self.sigmoid(np.dot(self.theta.T, X))
            Y_predict = np.zeros(A.shape)
            for i in range(A.shape[1]):
                if A[0, i] >= 0.5:
                    Y_predict[0, i] = 1
                else:
                    Y_predict[0, i] = 0
            return Y_predict
        elif flag == 4:
            A = self.sigmoid(np.dot(self.theta.T, X))
            return A
    def propagate_ROC(self, X, Y, k):
        A = np.zeros(Y.shape)
        A = self.sigmoid(np.dot(self.theta.T, X))
        Y_predict = np.zeros(A.shape)
        for i in range(A.shape[1]):
            if A[0, i] >= self.threshold[0, k]:
                Y_predict[0, i] = 1
            else:
                Y_predict[0, i] = 0
        return Y_predict
    def gradient_descent(self, A, X, Y):
        m = X.shape[1]
        if self.weight_decay:
            dtheta = 1 / m * np.dot((A - Y), X.T) + self.lambdaa * self.theta.T
        else:
            dtheta = 1 / m * np.dot((A - Y), X.T)
        return dtheta.T
    def optimize(self, dtheta, X, Y, cost):
        # if self.line_search == True:
        #     # step = 1
        #     # c = 0.5
        #     # beta = 0.8
        #     # self.ls_theta = self.theta - step * dtheta
        #     # cur_cost = self.propagate(X, Y, 4)
        #     # while cur_cost > cost - c * step * np.sum(dtheta ** 2):
        #     #     self.ls_theta = self.theta - step * dtheta
        #     #     step = step * beta
        #     # learning_rate = step
        #     pass
        # else:
        #     learning_rate = self.learning_rate
        learning_rate = self.learning_rate
        self.theta = self.theta - learning_rate * dtheta
    def split_trainingdata(self, X, Y):
        train_num = int(X.shape[1] * self.train_prop)
        index = [i for i in range(X.shape[1])]
        np.random.shuffle(index)
        index = index[0:train_num]
        X_train = X[:, index]
        Y_train = Y[0, index]
        Y_train = Y_train[np.newaxis, :]
        assert X_train.shape == (X.shape[0], train_num)
        assert Y_train.shape == (Y.shape[0], train_num)
        return X_train, Y_train
    def save_model(self, dir):
        parameters = self.learning_rate, self.batch_size, self.train_prop, \
                     self.conti_frame, self.line_search, self.weight_decay, \
                     self.early_stop, self.epochs, self.lambdaa
        with open(dir + 'parameters.pkl', 'wb') as f:
            pickle.dump(parameters, f)
        ROC = self.recall, self.FAR
        with open(dir + 'ROC.pkl', 'wb') as f:
            pickle.dump(ROC, f)
        weights = self.theta
        with open(dir + 'weights.pkl', 'wb') as f:
            pickle.dump(weights, f)
    def calculate_ROC(self, X, Y):
        threshold = self.propagate(X, Y, 4)
        self.threshold = np.sort(threshold, axis=1)

        progress = ProgressBar(Y.shape[1])
        progress.start()
        for k in range(self.threshold.shape[1]):
            Y_predict = self.propagate_ROC(X, Y, k)  # (1,m_test)
            TP = 0
            TN = 0
            FP = 0
            FN = 0
            for i in range(Y.shape[1]):
                if Y_predict[0, i]-Y[0, i] == 0 and Y[0, i] == 1:
                    TP = TP + 1
                elif Y_predict[0, i]-Y[0, i] == 0 and Y[0, i] == 0:
                    TN = TN + 1
                elif Y_predict[0, i]-Y[0, i] == 1:
                    FP = FP + 1
                elif Y_predict[0, i] - Y[0, i] == -1:
                    FN = FN + 1
                else:
                    pass
            self.recall.append(TP/(TP+FN))
            self.FAR.append(FP/(FP+TN))
            progress.show_progress(k)
        progress.end()
    def load_model_cal_ROC(self, dir, X, Y):
        with open(dir + 'weights.pkl', 'rb') as f:
            self.theta = pickle.load(f)
        self.calculate_ROC(X, Y)
        with open(dir + 'parameters.pkl', 'rb') as f:
            parameters = pickle.load(f)
            self.learning_rate, self.batch_size, self.train_prop, \
            self.conti_frame, self.line_search, self.weight_decay, \
            self.early_stop, self.epochs, self.lambdaa = parameters
        self.save_model(dir)
    def train(self, X_train, Y_train, X_val, Y_val, X_test, Y_test, dir):
        batch_num = int(X_train.shape[1] / self.batch_size)
        self.patience = batch_num * 1000
        for l in range(self.epochs):
            for i in range(batch_num):
                minibatch_X = X_train[:, i * self.batch_size:(i + 1) * self.batch_size]
                minibatch_Y = Y_train[0, i * self.batch_size:(i + 1) * self.batch_size]
                minibatch_Y = minibatch_Y[np.newaxis, :]
                A, cost = self.propagate(minibatch_X, minibatch_Y, 1)
                dtheta = self.gradient_descent(A, minibatch_X, minibatch_Y)
                self.optimize(dtheta, minibatch_X, minibatch_Y, cost)

                iter = l * batch_num + i
                if self.early_stop:
                    # 一个batch消耗一个patience
                    self.validation_frequency = min(batch_num, self.patience // 2)
                    if (iter + 1) % self.validation_frequency == 0:
                        Y_predict = self.propagate(X_val, Y_val, 3)
                        this_validation_loss = np.sum(np.fabs(Y_predict - Y_val)) / X_val.shape[1]
                        # print("epoch %i, minibatch %i, train_loss %f, val_loss %f" % (l, i, cost, val_cost))
                        print(
                            'epoch %i, minibatch %i/%i, patienct %f, train loss %f, validation error %f %%' %
                            (
                                l,
                                i,
                                batch_num,
                                self.patience,
                                cost,
                                this_validation_loss * 100.
                            )
                        )

                        # if we got the best validation score until now
                        if this_validation_loss < self.best_validation_loss:
                            # improve patience if loss improvement is good enough
                            if this_validation_loss < self.best_validation_loss * self.improvement_threshold:
                                self.patience = max(self.patience, iter * self.patience_increase)
                            self.best_validation_loss = this_validation_loss
                else:
                    Y_predict = self.propagate(X_val, Y_val, 3)
                    this_validation_loss = np.sum(np.fabs(Y_predict - Y_val)) / X_val.shape[1]
                    if i % 100 == 0:
                        print(
                            'epoch %i, minibatch %i/%i, patienct %f, train loss %f, validation error %f %%' %
                            (
                                l,
                                i,
                                batch_num,
                                self.patience,
                                cost,
                                this_validation_loss * 100.
                            )
                        )
                if self.patience <= iter:
                    break
                if self.best_validation_loss < 1e-5:
                    break

            lastbatch = X_train.shape[1] % self.batch_size
            if lastbatch != 0:
                lastbatch_X = X_train[:, batch_num * self.batch_size:batch_num * self.batch_size + lastbatch]
                lastbatch_Y = Y_train[0, batch_num * self.batch_size:batch_num * self.batch_size + lastbatch]
                lastbatch_Y = lastbatch_Y[np.newaxis, :]
                A, cost = self.propagate(lastbatch_X, lastbatch_Y, 1)
                dtheta = self.gradient_descent(A, lastbatch_X, lastbatch_Y)
                self.optimize(dtheta, lastbatch_X, lastbatch_Y, cost)

                if self.early_stop:
                    iter = l * batch_num + batch_num * self.batch_size
                    # 保证每个epoch会有一个validationloss的输出。保证在结束之前有一次输出。
                    self.validation_frequency = min(batch_num, self.patience // 2)
                    if (iter + 1) % self.validation_frequency == 0:
                        Y_predict = self.propagate(X_val, Y_val, 3)
                        this_validation_loss = np.sum(np.fabs(Y_predict - Y_val)) / X_val.shape[1]
                        print(
                            'epoch %i, minibatch %i/%i, patienct %f, train loss %f, validation error %f %%' %
                            (
                                l,
                                batch_num,
                                batch_num,
                                self.patience,
                                cost,
                                this_validation_loss * 100.
                            )
                        )

                        # if we got the best validation score until now
                        if this_validation_loss < self.best_validation_loss:
                            # improve patience if loss improvement is good enough
                            if this_validation_loss < self.best_validation_loss * self.improvement_threshold:
                                self.patience = max(self.patience, iter * self.patience_increase)
                            self.best_validation_loss = this_validation_loss

                        if self.patience <= iter:
                            break
                else:
                    Y_predict = self.propagate(X_val, Y_val, 3)
                    this_validation_loss = np.sum(np.fabs(Y_predict - Y_val)) / X_val.shape[1]
                    print(
                        'epoch %i, minibatch %i/%i, patienct %f, train loss %f, validation error %f %%' %
                        (
                            l,
                            batch_num,
                            batch_num,
                            self.patience,
                            cost,
                            this_validation_loss * 100.
                        )
                    )
                if self.best_validation_loss < 1e-5:
                    break
        self.calculate_ROC(X_test, Y_test)
        self.save_model(dir)

X_train, Y_train, X_test, Y_test, X_val, Y_val = lodd_data()
dim = X_train.shape[0]
model = Model(dim)
X_train, Y_train= model.split_trainingdata(X_train, Y_train)
dir = 'cv_trainingset/2'
# model.train(X_train, Y_train, X_val, Y_val, X_test, Y_test, dir)
model.load_model_cal_ROC(dir, X_test, Y_test)


















