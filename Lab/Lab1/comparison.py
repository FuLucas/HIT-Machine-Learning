import numpy as np
import matplotlib.pyplot as plt
from analytical_solution import *
from conjugate_gradient import *
from gradient_descent import *
from generate_data import *


if __name__ == '__main__':
    degree = 9
    number_train = 20 # 训练样本的数量
    number_test = 100  # 测试样本的数量
    # 训练样本数据生成
    xn_train, T_train = geneData(number_train, 0.0, 0.2)
    # 测试样本数据生成
    xn_test = np.linspace(0, 1, number_test)
    T_test = np.sin(2 * np.pi * xn_test)
    # 生成训练、测试样本相关X
    X_train = generateX(xn_train, degree)
    X_test = generateX(xn_test, degree)
    # 训练数据点图
    plt.scatter(xn_train, T_train, marker="+", color="b", label="train data")
    # 测试数据图
    plt.plot(xn_test, T_test, color="k", label="$\sin(2\pi x)$")
    # 无惩罚项（正则项）的解析解
    fit_with_regular_term = AnalyticalSolution(X_train, T_train, np.exp(-8))
    w_AnalyticalSolution = fit_with_regular_term.fit()
    conjugate_gradient = ConjugateGradient(X_train, T_train, np.zeros(degree + 1))
    k, w_ConjugateGradient, losses = conjugate_gradient.fit()
    gradient_descent = GradientDescent(X_train, T_train, np.zeros(degree + 1))
    k_1, w_GradientDescent, losses_1 = gradient_descent.fit()
    # 拟合结果图
    plt.plot(xn_test, np.dot(X_test, w_AnalyticalSolution), color="r", label="analytical solution")
    plt.plot(xn_test, np.dot(X_test, w_ConjugateGradient), color="g", label="conjugate gradient")
    plt.plot(xn_test, np.dot(X_test, w_GradientDescent), color="b", label="gradient descent")
    plt.legend(loc='best')
    plt.title("degree = " + str(degree) + ", train number = "+
                str(number_train)+ ", test number = "+ str(number_test), fontsize= "medium")
    plt.show()

    plt.plot(losses, color="k", label="Conjugate Gradient Loss")
    plt.plot(losses_1, color="r", label="Gradient Descent Loss")
    plt.xlabel("Number of iterations")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.show()
