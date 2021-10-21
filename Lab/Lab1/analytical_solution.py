"""
    解析解求解两种loss的最优解（无正则项和有正则项）
"""
import numpy as np
import matplotlib.pyplot as plt
from generate_data import *

class AnalyticalSolution(object):
    def __init__(self, X, T, lamda=0):
        self.X = X
        self.T = T
        self. lamda = lamda

    def fit(self):
        """ 带正则项的解析解
        Args: 
            X: len(row) * (degree + 1) 的矩阵，每一行是每个元素的0到degree次幂
            T: 目标值的列向量
            lamda: 正则项系数的二倍，若不带正则项
        Returns: 
            w* = (X'X+lanmda I)^(-1)X'T     带正则项
            w* = (X'X)^(-1)X'T              不带正则项
        """
        # return np.linalg.inv(self.lamda * np.identity(len(self.X.T)) 
        #                     + self.X.T @ self.X) @ self.X.T @ self.T
        # wrong answer
        return np.linalg.solve(self.lamda * np.identity(len(self.X.T)) + 
                                self.X.T @ self.X, self.X.T @ self.T)

    def E_rms(self, y_true, y_pred):
        """ 计算E_RMS
        """
        return np.sqrt(np.mean(np.square(y_true - y_pred)))


if __name__ == '__main__':
    """
        增加训练样本数量来解决过拟合问题
    """
    degree = 9
    for index in range(1, 7):
        number_train = 10 + 5 * (index - 1)  # 训练样本的数量
        number_test = 100  # 测试样本的数量
        # 训练样本数据生成
        xn_train, T_train = geneData(number_train, 0.0, 0.2)
        # 测试样本数据生成
        xn_test = np.linspace(0, 1, number_test)
        T_test = np.sin(2 * np.pi * xn_test)
        # 生成训练、测试样本相关X
        X_train = generateX(xn_train, degree)
        X_test = generateX(xn_test, degree)
        plt.subplot(2, 3, index)
        # 训练数据点图
        plt.scatter(xn_train, T_train, marker="+", color="b")
        # 测试数据图
        plt.plot(xn_test, T_test, color="k")
        # 无惩罚项（正则项）的解析解
        fit_without_regular_term = AnalyticalSolution(X_train, T_train)
        w = fit_without_regular_term.fit()
        # 拟合结果图
        plt.plot(xn_test, np.dot(X_test, w), "r")
        plt.legend(labels=["train data","$\sin(2\pi x)$","analytical solution"], loc='best')
        plt.title("degree = " + str(degree) + ", train number = "+
                    str(number_train)+ ", test number = 100", fontsize= "medium")
    plt.show()
    

    """
        无惩罚项，N=10，阶数从1到9的拟合结果
    """
    number_train = 10  # 训练样本的数量
    number_test = 100  # 测试样本的数量

    # 训练样本数据生成
    xn_train, T_train = geneData(number_train, 0.0, 0.2)
    # 测试样本数据生成
    xn_test = np.linspace(0, 1, number_test)
    T_test = np.sin(2 * np.pi * xn_test)

    rms = []
    # 从最高次数1到9进行拟合
    for degree in range(1, 10):
        # 生成训练、测试样本相关X
        X_train = generateX(xn_train, degree)
        X_test = generateX(xn_test, degree)
        plt.subplot(3, 3, degree)
        # 训练数据点图
        plt.scatter(xn_train, T_train, marker="+", color="b")
        # 测试数据图
        plt.plot(xn_test, T_test, color="k")
        # 无惩罚项（正则项）的解析解
        fit_without_regular_term = AnalyticalSolution(X_train, T_train)
        w = fit_without_regular_term.fit()
        Y = np.dot(X_test, w)
        rms.append(fit_without_regular_term.E_rms(Y, T_test))
        # 拟合结果图
        plt.plot(xn_test, np.dot(X_test, w), "r")
        plt.legend(labels=["train data","$\sin(2\pi x)$","analytical solution"], loc='best')
        plt.title("degree = " + str(degree) + ", train number = 10, test number = 100", fontsize= "medium")
        print(w)

    print(rms)
    plt.show()


    """
        对于带正则项（惩罚项）的解析解，统计10000次实验得到最优lambda
        此处所加的高斯噪音均值为0，方差为0.2
    """
    number_train = 20  # 训练样本的数量
    number_test = 100  # 测试样本的数量
    degree = 9
    # 收集lamda可能大小的量
    buckets = np.zeros(51, dtype=np.int32)

    # 解多次，得到可能的最优lamda
    for i in range(10000):
        # 训练样本数据生成
        xn_train, T_train = geneData(number_train, 0.0, 0.2)
        # 测试样本数据生成
        xn_test = np.linspace(0, 1, number_test)
        T_test = np.sin(2 * np.pi * xn_test)
        # 生成训练、测试样本相关X
        X_train = generateX(xn_train, degree)
        X_test = generateX(xn_test, degree)
        # 测试lamda为e^(-50)到e^0的效果
        lamda_to_test = range(-50, 1)
        RMS_lamda = []
        for ex in lamda_to_test:
            fit_with_regular_term = AnalyticalSolution(X_train, T_train, np.exp(ex))
            w = fit_with_regular_term.fit()
            Y = np.dot(X_test, w)
            RMS_lamda.append(fit_with_regular_term.E_rms(Y, T_test))
        temp = np.array(lamda_to_test) # 将range转化为array类型，以求取最值索引
        best_ln_lamda = temp[np.where(RMS_lamda == np.min(RMS_lamda))]
        buckets[abs(best_ln_lamda)] = buckets[abs(best_ln_lamda)] + 1
    print(buckets)
    

    """
        多次随机训练，寻找最优超参数
    """
    for index in range(1, 5):
        number_train = 20  # 训练样本的数量
        number_test = 100  # 测试样本的数量
        degree = 9
        # 训练样本数据生成
        xn_train, T_train = geneData(number_train, 0.0, 0.2)
        # 测试样本数据生成
        xn_test = np.linspace(0, 1, number_test)
        T_test = np.sin(2 * np.pi * xn_test)
        # 生成训练、测试样本相关X
        X_train = generateX(xn_train, degree)
        X_test = generateX(xn_test, degree)

        plt.subplot(2, 2, index)
        # 测试lamda为e^(-50)到e^0的效果
        lamda_to_test = range(-50, 1)
        RMS_lamda = []
        for ex in lamda_to_test:
            fit_with_regular_term = AnalyticalSolution(X_train, T_train, np.exp(ex))
            w = fit_with_regular_term.fit()
            Y = np.dot(X_test, w)
            RMS_lamda.append(fit_with_regular_term.E_rms(Y, T_test))
        temp = np.array(lamda_to_test) # 将range转化为array类型，以求取最值索引
        best_ln_lamda = temp[np.where(RMS_lamda == np.min(RMS_lamda))]
        annotate = "$\lambda = e^{" + str(best_ln_lamda) + "}$"
        plt.ylabel("$E_{RMS}$")
        plt.xlabel("$ln \lambda$")
        plt.annotate(annotate, xy=(-30, 0.35))
        plt.plot(lamda_to_test, RMS_lamda, '+-', label="Test")
        plt.title("$E_{RMS}$ for $ln \lambda$ from -50 to 0")
        plt.legend()
    plt.show()


    """
        对比有无惩罚项的拟合结果
    """
    number_train = 10  # 训练样本的数量
    number_test = 100  # 测试样本的数量

    # 训练样本数据生成
    xn_train, T_train = geneData(number_train, 0.0, 0.2)
    # 测试样本数据生成
    xn_test = np.linspace(0, 1, number_test)
    T_test = np.sin(2 * np.pi * xn_test)

    # 从最高次数1到9进行拟合
    for degree in range(1, 10):
        # 生成训练、测试样本相关X
        X_train = generateX(xn_train, degree)
        X_test = generateX(xn_test, degree)
        plt.subplot(3, 3, degree)
        # 训练数据点图
        plt.scatter(xn_train, T_train, marker="+", color="b", label="Train data")
        # 测试数据图
        plt.plot(xn_test, T_test, color="k", label="$\sin(2\pi x)$")
        # 无惩罚项（正则项）的解析解
        fit_without_regular_term = AnalyticalSolution(X_train, T_train)
        fit_with_regular_term = AnalyticalSolution(X_train, T_train, np.exp(-8))
        w = fit_without_regular_term.fit()
        w_lamda = fit_with_regular_term.fit()
        # Y = np.dot(X_test, w)
        # rms.append(fit_without_regular_term.E_rms(Y, T_test))
        # 拟合结果图
        plt.plot(xn_test, np.dot(X_test, w), "r", label="WITHOUT regulation")
        plt.plot(xn_test, np.dot(X_test, w_lamda), "g", label="WITH regulation")
        plt.legend(loc='best')
        plt.title("degree = " + str(degree) + ", train number = 10, test number = 100", fontsize= "medium")
        print(w)
    plt.show()
