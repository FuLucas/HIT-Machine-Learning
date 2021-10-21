import numpy as np
from operation import *
import matplotlib.pyplot as plt
import prettytable as pt

# 梯度下降
class GradientDescent(object):
    def __init__(self, x, y, beta_0, hyper=0, rate=0.1, tolerance=1e-6):
        """梯度下降法初始化数据

        Args:
            x (array): 在原始特征array前加了一列的数组
            y (array): 训练样本的标签
            beta_0 (array): 初始化beta，一般是0
            hyper (int, optional): 惩罚项的系数，即lambda. Defaults to 0.
            rate (float, optional): 学习率，即梯度下降步长. Defaults to 0.1.
            tolerance (float, optional): 容忍度，即当一阶导数均小于这个值时认为收敛. Defaults to 1e-6.
        """
        self.x = x
        self.y = y
        self.beta_0 = beta_0
        self.hyper = hyper
        self.rate = rate
        self.tolerance = tolerance
        self.__row = len(x)
        self.__col = len(x.T)

    def __loss(self, beta):
        ans = 0.5 * self.hyper * beta @ beta.T
        for i in range(self.__row):
            ans -= self.y[i] * beta @ self.x[i].T
            ans += np.log(1 + np.exp(beta @ self.x[i].T))
        return ans / self.__row

    def __derivative(self, beta):
        ans = np.zeros(self.__col)
        for i in range(self.__row):
            ans += self.x[i] * (self.y[i] - (1.0 - sigmoid(beta @ self.x[i].T)))
        return (-1 * ans + self.hyper * beta) / self.__row

    def fit(self):
        losses = []
        loss_0 = self.__loss(self.beta_0)
        losses.append(loss_0)
        k = 0
        beta = self.beta_0
        while True:
            der = self.__derivative(beta)
            beta_t = beta - self.rate * der
            loss = self.__loss(beta_t)
            losses.append(loss)
            if np.abs(loss - loss_0) < self.tolerance:
                break
            else:
                k += 1
                if loss > loss_0:
                    self.rate *= 0.5
                loss_0 = loss
                beta = beta_t
        return k, beta, losses


if __name__ == '__main__':
    # 获取数据，默认为正例500，负例500，且满足朴素贝叶斯假设（互不相关）
    # x, y = Data()
    x, y = Data(naive=False)
    xPlus = x2xPlus(x)
    Train_x, Train_y, Test_x, Test_y = SplitData(xPlus, y)
    beta_0 = np.zeros(xPlus.shape[1])

    hyper = np.exp(-6)
    # 无惩罚项（正则项）的梯度下降法
    gradient_descent = GradientDescent(Train_x, Train_y, beta_0)
    k_gradient, beta_gradient, losses_gradient = gradient_descent.fit()
    accuracy_gradient = accuracy(Test_x, Test_y, beta_gradient)
    # 带惩罚项（正则项）的梯度下降法
    gradient_descent_penalty = GradientDescent(Train_x, Train_y, beta_0, hyper=hyper)
    k_gradient_penalty, beta_gradient_penalty, losses_penalty = gradient_descent_penalty.fit()
    accuracy_gradient_penalty = accuracy(Test_x, Test_y, beta_gradient_penalty)

    # 画出二维参数的样本
    type1_x = Train_x[Train_y==1][:,1]
    type1_y = Train_x[Train_y==1][:,2]
    type0_x = Train_x[Train_y==0][:,1]
    type0_y = Train_x[Train_y==0][:,2]

    plt.scatter(type1_x, type1_y, marker="x", c="b", label="Positive")
    plt.scatter(type0_x, type0_y, marker="x", c="r", label="Negative")

    # 无惩罚项的结果图
    x_results = np.linspace(-3, 3)
    y_results = - (beta_gradient[0] +beta_gradient[1] * x_results) / beta_gradient[2]
    plt.plot(x_results, y_results, color="k", label='Gradient Descent without Penalty')
    # 带惩罚项的结果图
    y_results_penalty = - (beta_gradient_penalty[0] +beta_gradient_penalty[1] * x_results) / beta_gradient_penalty[2]
    plt.plot(x_results, y_results_penalty, color="r", label='Gradient Descent with Penalty')

    plt.xlabel("$X_0$")
    plt.ylabel("$X_1$")
    plt.legend(loc='best')
    plt.show()

    tb = pt.PrettyTable()
    tb.field_names = ["Methods", "Number of iterations", "beta(b, w0, w1)", "Accuracy"]
    tb.add_row(["Gradient Descent without Penalty", k_gradient, beta_gradient, accuracy_gradient])
    tb.add_row(["Gradient Descent with Penalty", k_gradient_penalty, beta_gradient_penalty, accuracy_gradient_penalty])
    print(tb)
