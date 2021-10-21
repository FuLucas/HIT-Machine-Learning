import numpy as np
import pandas as pd
from operation import *
from newton import *
from gradient_descent import *
import prettytable as pt

def GetData():
    data_set = pd.read_csv("./data/sobar-72.csv")
    x = data_set.drop('ca_cervix', axis=1)
    y = data_set['ca_cervix']
    return x, y

if __name__ == '__main__':
    x, y = GetData()
    xPlus = x2xPlus(x)
    Train_x, Train_y, Test_x, Test_y = SplitData(xPlus, y)
    beta_0 = np.zeros(xPlus.shape[1])

    hyper = np.exp(-6)
    # 无惩罚项（正则项）的牛顿法
    newton = Newton(Train_x, Train_y, beta_0)
    k_newton, beta_newton = newton.fit()
    accuracy_newton = accuracy(Test_x, Test_y, beta_newton)
    # 带惩罚项（正则项）的牛顿法
    newton_penalty = Newton(Train_x, Train_y, beta_0, hyper=hyper)
    k_newton_penalty, beta_newton_penalty = newton_penalty.fit()
    accuracy_newton_penalty = accuracy(Test_x, Test_y, beta_newton_penalty)

    # 无惩罚项（正则项）的梯度下降法
    gradient_descent = GradientDescent(Train_x, Train_y, beta_0)
    k_gradient, beta_gradient, losses_gradient = gradient_descent.fit()
    accuracy_gradient = accuracy(Test_x, Test_y, beta_gradient)
    # 带惩罚项（正则项）的梯度下降法
    gradient_descent_penalty = GradientDescent(Train_x, Train_y, beta_0, hyper=hyper)
    k_gradient_penalty, beta_gradient_penalty, losses_penalty = gradient_descent_penalty.fit()
    accuracy_gradient_penalty = accuracy(Test_x, Test_y, beta_gradient_penalty)

    tb = pt.PrettyTable()
    tb.field_names = ["Methods", "Number of iterations", "beta(b, w0, w1)", "Accuracy"]
    tb.add_row(["Newton Method without Penalty", k_newton, beta_newton, accuracy_newton])
    tb.add_row(["Newton Method with Penalty", k_newton_penalty, beta_newton_penalty, accuracy_newton_penalty])
    tb.add_row(["Gradient Descent without Penalty", k_gradient, beta_gradient, accuracy_gradient])
    tb.add_row(["Gradient Descent with Penalty", k_gradient_penalty, beta_gradient_penalty, accuracy_gradient_penalty])
    print(tb)
