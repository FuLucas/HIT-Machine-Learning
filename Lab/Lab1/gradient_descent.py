"""
    优化方法求解最优解（梯度下降法）
"""
import numpy as np
import matplotlib.pyplot as plt
from generate_data import *

class GradientDescent(object):
    def __init__(self, X, T, w_0, lamda=np.exp(-8), rate=0.01, delta=1e-6):
        """ 用于多项式函数使用梯度下降拟合初始化
        Args:
            X: len(row) * (degree + 1) 的矩阵，每一行是每个元素的0到degree次幂
            T: 目标值的列向量
            w_0: 初始解，通常以全零向量
            lamda: 正则项系数部分，之前求得最合适的范围在1e-7或者1e-8之间，默认为1e-8
            rate: 学习率，初始为0.5
            delta: 精度要求，初始为1e-6，当小于这个值时认为趋于0
        """
        self.X = X
        self.T = T
        self.w_0 = w_0
        self.lamda = lamda
        self.rate = rate
        self.delta = delta

    def loss(self, w):
        temp = self.X @ w - self.T
        return 0.5 * np.mean(temp.T @ temp + self.lamda * w.T @ w)

    def __derivative(self, w):
        """ 一阶函数导数 
        """
        return self.X.T @ self.X @ w + self.lamda * w - self.X.T @ self.T

    def fit(self):
        """ 多项式函数使用梯度下降拟合
        Returns:
            w: 梯度下降优化得到的最优解
        """
        losses = []
        loss_0 = self.loss(self.w_0)
        losses.append(loss_0)
        k = 0
        w = self.w_0
        der_0 = self.__derivative(w)
        while True:
            # der_0 = der
            der = self.__derivative(w)
            wk = w - self.rate * der
            loss = self.loss(wk)
            losses.append(loss)
            # 增加条件，当一阶导数收敛（在这里取序列收敛性）时，才认为函数收敛
            max, min = (der - der_0).max(), (der - der_0).min()
            if np.abs(loss - loss_0) < self.delta and np.abs(max) < self.delta and np.abs(min) < self.delta:
            # if np.abs(loss - loss_0) < self.delta and (der - der_0) < self.delta:
                # print(der, der_0)
                break
            else:
                k = k + 1
                loss_0 = loss
                w = wk
                der_0 = der
        return k, w, losses


if __name__ == "__main__":

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
    #         gradient_descent = GradientDescent(X_train, T_train, np.zeros(degree + 1))
    #         k, w, losses = gradient_descent.fit()
    #         print("training number: "+ str(number_train) + " degree: " +str(degree) + 
    #                 " 迭代次数："+ str(k) + "\n w: " + str(w))

    #         plt.subplot(3, 3, num)
    #         # 训练数据点图
    #         plt.scatter(xn_train, T_train, marker="+", color="b", label="train data")
    #         # 测试数据图
    #         plt.plot(xn_test, T_test, color="k", label="$\sin(2\pi x)$")
    #         # 拟合结果图
    #         plt.plot(xn_test, np.dot(X_test, w), color="r", label="gradient descent")
    #         plt.legend(loc='best')
    #         plt.title("degree = " + str(degree) + ", train number = " + str(number_train) + 
    #                     " test number = 100", fontsize= "medium")
    # plt.show()


    """
        对比不同w_0给梯度下降结果带来的差异
    """
    number_train = 20  # 训练样本的数量
    number_test = 100  # 测试样本的数量

    # 训练样本数据生成
    xn_train, T_train = geneData(number_train, 0.0, 0.4)
    # 测试样本数据生成
    xn_test = np.linspace(0, 1, number_test)
    T_test = np.sin(2 * np.pi * xn_test)

    degree = 6
    # 生成训练、测试样本相关X
    X_train = generateX(xn_train, degree)
    X_test = generateX(xn_test, degree)
    conjugate_gradient_0 = GradientDescent(X_train, T_train, np.zeros(degree + 1))
    # conjugate_gradient_1 = GradientDescent(X_train, T_train, np.ones(degree + 1))
    k_0, w_0, losses_0 = conjugate_gradient_0.fit()
    # k_1, w_1, losses_1 = conjugate_gradient_1.fit()
    print(k_0, w_0)

    # 训练数据点图
    plt.scatter(xn_train, T_train, marker="+", color="b", label="train data")
    # 测试数据图
    plt.plot(xn_test, T_test, color="k", label="$\sin(2\pi x)$")
    # 拟合结果图
    plt.plot(xn_test, np.dot(X_test, w_0), color="r", label="$w_0 = 0$")
    # plt.plot(xn_test, np.dot(X_test, w_1), color="g", label="$w_0 = 1$")
    plt.legend(loc='best')
    plt.title("degree = " + str(degree) + ", train number = 10, test number = 100", fontsize= "medium")
    plt.show()

    plt.plot(losses_0, color="k", label="Gradient Descent Loss")
    plt.xlabel("Number of iterations")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.show()