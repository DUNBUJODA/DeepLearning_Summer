DeepLearning_Summar Code Documantation
===========================
该文件为本工程的说明文档

****
    
|Project|Deep Learning Frameworks|
|---|---
|Author|DUNBUJODA


****
## 目录
* [lr.py](#lr.py)
* [MLP_MNIST.py](#MLP_MNIST.py)
* [MLP_basketball.py](#MLP_basketball.py)
* [CNN_faster_mnist.py](#CNN_faster_mnist.py)
* [CNN_faster_basketball.py](#CNN_faster_basketball.py)

lr.py
-----
```
class Model:
    def __init__(self, dim):  # parameters
    def sigmoid(self, z)  # activator
    def propagate(self, X, Y, flag)  # forward propagation
    def propagate_ROC(self, X, Y, k)  # forward propagation for calculating predictions
    def gradient_descent(self, A, X, Y)  # back propagation
    def optimize(self, dtheta, X, Y, cost)  # update parameters
    def split_trainingdata(self, X, Y)  # split training set
    def save_model(self, dir)  # save parameters of the model
    def calculate_ROC(self, X, Y)  # calculate recall & FAR
    def load_model_cal_ROC(self, dir, X, Y)  # load parameters of the model and calculate ROC
    def train(self, X_train, Y_train, X_val, Y_val, X_test, Y_test, dir)  # train the model
```
train the model
```
model.train(X_train, Y_train, X_val, Y_val, X_test, Y_test, dir)
```

MLP_MNIST.py
-----
```
class HiddenLayer(object):  # hidden layers
    def __init__(self, input_size, output_size, learning_rate, activator, L2_reg, rng: np.random.RandomState = None)
    def forward(self, X)
    def backward(self, W_lplus1, delta_lplus1)
    def upgrade(self)
```
```
class OutputLayer():
    def __init__(self, input_size, output_size, learning_rate, L2_reg, rng: np.random.RandomState = None)
    def softmax(self, pred)
    def forward(self, X)
    def backward(self, Y)
    def upgrade(self)
```
```
class MLP(object):
    def __init__(self, input_size: int, hidden_layers: list, output_size: int)
    def propagate(self, X, Y, flag)
    def backpropagate(self, X, Y)
    def train(self, X_train, Y_train, X_val, Y_val, X_test, Y_test)
```

train the model
```
model = MLP(input_size=train_set_x.shape[0], hidden_layers=[500], output_size=10)
model.train(train_set_x, train_set_y, valid_set_x, valid_set_y, test_set_x, test_set_y)
```


MLP_basketball.py
-----


CNN_faster_mnist.py
-----


CNN_faster_basketball.py
-----
