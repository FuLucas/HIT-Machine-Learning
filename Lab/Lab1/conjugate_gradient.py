"""
    优化方法求解最优解（共轭梯度法）
"""
import numpy as np
import matplotlib.pyplot as plt
from generate_data import *

class ConjugateGradient(object):
    def __init__(self, X, T, w_0, lamda=np.exp(-8), delta=1e-6):
        """ 共轭梯度初始化
        Args:
            X: len(row) * (degree + 1) 的矩阵，每一行是每个元素的0到degree次幂
            T: 目标值的列向量
            w_0: 一般假设初始化为0
            lamda: 正则项系数部分，之前求得最合适的范围在1e-7或者1e-8之间，初始化为e^(-8)
            delta: 精度要求，初始为1e-6，当小于这个值时认为趋于0
        """
        self.X = X
        self.T = T
        self.w_0 = w_0
        self.lamda = lamda
        self.delta = delta
        # A = X'X + lamda I
        self.A = X.T @ X + np.identity(len(X.T)) * lamda
        # b = X'X
        self.b = X.T @ T

    def loss(self, w):
        temp = self.X @ w - self.T
        return 0.5 * np.mean(temp.T @ temp + self.lamda * w.T @ w)

    def fit(self):
        w = self.w_0
        r_0 = self.b - self.A @ self.w_0
        p = r_0
        k = 0
        losses = []
        losses.append(self.loss(w))
        while True:
            alpha = (r_0.T @ r_0) / (p.T @ self.A @ p)
            w = w + alpha * p
            r = r_0 - alpha * self.A @ p
            if r_0.T @ r_0 < self.delta:
                break
            beta = (r.T @ r) / (r_0.T @ r_0)
            p = r + beta * p
            r_0 = r
            k += 1
            losses.append(self.loss(w))
        return k, w, losses


if __name__ == "__main__":

    number_train = 20  # 训练样本的数量
    number_test = 100  # 测试样本的数量

    # 训练样本数据生成
    xn_train, T_train = geneData(number_train, 0.0, 0.4)
    # 测试样本数据生成
    xn_test = np.linspace(0, 1, number_test)
    T_test = np.sin(2 * np.pi * xn_test)

    degree = 9
    # 生成训练、测试样本相关X
    X_train = generateX(xn_train, degree)
    X_test = generateX(xn_test, degree)
    conjugate_gradient = ConjugateGradient(X_train, T_train, np.zeros(degree + 1))
    k, w, losses = conjugate_gradient.fit()
    print(k,w)

    # 训练数据点图
    plt.scatter(xn_train, T_train, marker="+", color="b", label="train data")
    # 测试数据图
    plt.plot(xn_test, T_test, color="k", label="$\sin(2\pi x)$")
    # 拟合结果图
    plt.plot(xn_test, np.dot(X_test, w), color="r", label="conjugate gradient")
    plt.legend(loc='best')
    plt.title("degree = " + str(degree) + ", train number = 10, test number = 100", fontsize= "medium")
    plt.show()

    plt.plot(losses, color="k", label="Conjugate Gradient Loss")
    plt.xlabel("Number of iterations")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.show()

    # num = 0
    # for index in range(1, 4):
    #     number_train = 10 * index  # 训练样本的数量
    #     number_test = 100  # 测试样本的数量

    #     # 训练样本数据生成
    #     xn_train, T_train = geneData(number_train, 0.0, 0.4)
    #     # 测试样本数据生成
    #     xn_test = np.linspace(0, 1, number_test)
    #     T_test = np.sin(2 * np.pi * xn_test)

    #     degrees = [3, 6, 9]
    #     for degree in degrees:
    #         num += 1
    #         # 生成训练、测试样本相关X
    #         X_train = generateX(xn_train, degree)
    #         X_test = generateX(xn_test, degree)
    #         conjugate_gradient = ConjugateGradient(X_train, T_train, np.zeros(degree + 1))
    #         k, w, losses = conjugate_gradient.fit()
    #         print("training number: "+ str(number_train) + " degree: " +str(degree) + 
    #                 " 迭代次数："+ str(k) + "\n w: " + str(w))

    #         plt.subplot(3, 3, num)
    #         # 训练数据点图
    #         plt.scatter(xn_train, T_train, marker="+", color="b", label="train data")
    #         # 测试数据图
    #         plt.plot(xn_test, T_test, color="k", label="$\sin(2\pi x)$")
    #         # 拟合结果图
    #         plt.plot(xn_test, np.dot(X_test, w), color="r", label="conjugate gradient")
    #         plt.legend(loc='best')
    #         plt.title("degree = " + str(degree) + ", train number = " + str(number_train) + 
    #                     ", test number = 100", fontsize= "medium")
    # plt.show()