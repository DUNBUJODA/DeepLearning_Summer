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

MLP_MNIST.py
-----


MLP_basketball.py
-----


CNN_faster_mnist.py
-----


CNN_faster_basketball.py
-----
