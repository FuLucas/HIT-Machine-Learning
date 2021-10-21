import numpy as np
from numpy.matrixlib.defmatrix import matrix
from operation import *
import matplotlib.pyplot as plt
import prettytable as pt

# 牛顿法
class Newton(object):
    def __init__(self, x, y, beta_0, hyper=0, tolerance=1e-6, max_iter = 50):
        """牛顿法初始化变量
        Args:
            x (array): 在原始特征array前加了一列的数组
            y (array): 训练样本的标签
            beta_0 (array): 初始化beta，一般是0
            hyper (int, optional): 惩罚项的系数，即lambda. Defaults to 0.
            tolerance (float, optional): 容忍度，即当一阶导数均小于这个值时认为收敛. Defaults to 1e-6.
            max_iter (int, optional): 最多迭代次数，超过这个值认为不收敛. Defaults to 50.
        """
        self.x = x
        self.y = y
        self.beta_0 = beta_0
        self.hyper = hyper
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.__row = len(x)
        self.__col = len(x.T)

    def __derivative(self, beta):
        """求一阶导数
        Args:
            beta (array): 特征值的系数
        Returns:
            array: 根据beta求出的导数值
        """
        ans = np.zeros(self.__col)
        for i in range(self.__row):
            ans += (self.x[i] * (self.y[i] - sigmoid( - beta @ self.x[i].T)))
        return - ans + self.hyper * beta

    def __hessian(self, beta):
        """求二阶导数，即海森矩阵
        Args:
            beta (array): 特征值系数
        Returns:
            array: 根据beta得到的二阶导数
        """
        ans = np.eye(self.__col) * self.hyper
        for i in range(self.__row):
            temp = sigmoid(beta @ self.x[i].T)
            m = np.mat(self.x[i]).T
            ans += np.array(m * m.T) * temp * (1 - temp)
        return ans

    def fit(self):
        k = 0
        beta = self.beta_0
        while k <= self.max_iter:
            gradient = self.__derivative(beta)
            if np.linalg.norm(gradient) < self.tolerance:
                break
            hess = self.__hessian(beta)
            beta_t = beta - np.linalg.inv(hess) @ gradient
            beta = beta_t
            k += 1
        return k, beta

if __name__ == '__main__':
    # x, y = Data(naive=False)
    x, y = Data()
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

    # 训练样本
    type1_x = Train_x[Train_y==1][:,1]
    type1_y = Train_x[Train_y==1][:,2]
    type0_x = Train_x[Train_y==0][:,1]
    type0_y = Train_x[Train_y==0][:,2]
    plt.scatter(type1_x, type1_y, marker="x", c="b", label="Positive")
    plt.scatter(type0_x, type0_y, marker="x", c="r", label="Negative")

    # 无惩罚项的结果图
    x_results = np.linspace(-3, 3)
    y_results = - (beta_newton[0] +beta_newton[1] * x_results) / beta_newton[2]
    plt.plot(x_results, y_results, color="k", label='Newton without Penalty')
    # 带惩罚项的结果图
    y_results_penalty = - (beta_newton_penalty[0] +beta_newton_penalty[1] * x_results) / beta_newton_penalty[2]
    plt.plot(x_results, y_results_penalty, color="r", label='Newton with Penalty')

    plt.xlabel("$X_0$")
    plt.ylabel("$X_1$")
    plt.legend(loc='best')
    plt.show()

    tb = pt.PrettyTable()
    tb.field_names = ["Methods", "Number of iterations", "beta(b, w0, w1)", "Accuracy"]
    tb.add_row(["Newton Method without Penalty", k_newton, beta_newton, accuracy_newton])
    tb.add_row(["Newton Method with Penalty", k_newton_penalty, beta_newton_penalty, accuracy_newton_penalty])
    print(tb)
