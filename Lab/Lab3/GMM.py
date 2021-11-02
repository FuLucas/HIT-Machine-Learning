import numpy as np
import random
from scipy.stats import multivariate_normal
import collections
import matplotlib.pyplot as plt
from operations import Data, Accuracy


class GaussianMixtureModel(object):
    def __init__(self, data, k, delta=1e-12, max_iteration=1000):
        """初始化GMM
        Args:
            data (array): 数据数组，每行表示一个样本地特征，行数表示样本数
            k (int): 要分成的类数
            delta (double, optional): tolerance，当类别中心的变化的二范数小于这个值就认为分类结束. Defaults to 1e-12.
            max_iteration (int, optional): 最大迭代次数，超过这个值就不再进行迭代. Defaults to 1000.
        """
        self.data = data
        self.k = k
        self.delta = delta
        self.max_iteration = max_iteration
        self.__alpha = np.ones(self.k) * (1.0 / self.k)
        self.__mu = np.array(self.data[random.sample(range(data.shape[0]), k)])
        self.__Sigma = self.__init_Sigma()
        self.label = None
        self.__gamma = None
        self.__LL = -np.inf

    def __init_Sigma(self):
        """初始化Sigma值，对角矩阵，且对角上每个值都设为0.1
        Returns:
            array: Sigma矩阵
        """
        Sigma = collections.defaultdict(list)
        for i in range(self.k):
            Sigma[i] = np.eye(self.data.shape[1], dtype=float) * 0.1
        return Sigma

    def __ProbabilityDensity(self):
        """根据分布计算每个样本属于某个类别的概率
        Returns:
            array: 每个样本属于每个类别的概率
        """
        ProbabilityDensity = np.zeros((self.data.shape[0], self.k))
        for i in range(self.k):
            ProbabilityDensity[:, i] = multivariate_normal.pdf(
                self.data, self.__mu[i], self.__Sigma[i])
        return ProbabilityDensity

    def __expectation(self):
        """E步，计算后验概率
        """
        # Gaussian mixture distribution
        Mixture_distribution = self.__ProbabilityDensity() * self.__alpha
        sum_ProbabilityDensity = np.sum(Mixture_distribution, axis=1).reshape(-1, 1)
        self.__LL = np.sum(np.log(sum_ProbabilityDensity))
        print(self.__LL)
        self.__gamma = Mixture_distribution / sum_ProbabilityDensity
        self.label = self.__gamma.argmax(axis=1)

    def __maximization(self):
        """M步，更新mu、Sigma和alpha
        """
        for i in range(self.k):
            gamma = self.__gamma[:, i].reshape(-1, 1)
            mu_i = np.sum(gamma * self.data, axis=0) / np.sum(gamma)
            covariance = (self.data - mu_i).T.dot((self.data -
                                                   mu_i) * gamma) / np.sum(gamma)
            self.__mu[i], self.__Sigma[i] = mu_i, covariance
        self.__alpha = self.__gamma.sum(axis=0) / self.data.shape[0]

    def GMM(self):
        last_LL = self.__LL
        iter = 0
        LLs = []
        while iter < self.max_iteration:
            self.__expectation()
            self.__maximization()
            iter += 1
            if self.__LL - last_LL <= self.delta:
                break
            last_LL = self.__LL
            LLs.append(last_LL)
        self.__expectation()
        return self.__mu, self.label, iter, LLs


if __name__ == '__main__':
    data0 = Data([1, 1])
    data1 = Data([4, 4])
    data2 = Data([12, 1])
    data3 = Data([8, 3])
    data4 = Data([9, 6])
    classes = 5
    data = np.vstack((data0, data1, data2, data3, data4))
    real_label = np.concatenate((np.zeros(len(data0)), np.ones(len(
        data1)), 2 * np.ones(len(data2)), 3 * np.ones(len(data3)), 4 * np.ones(len(data4))))

    plt.subplot(121)
    plt.title("My Data")
    plt.scatter(data[:, 0].tolist(), data[:, 1].tolist(), marker="x", c=np.array(
        real_label), cmap='rainbow', label="Generated Data")
    plt.legend()

    GMM = GaussianMixtureModel(data, classes)
    mu, label, iter, LLs = GMM.GMM()
    # show result
    plt.subplot(122)
    plt.title("GMM")
    plt.scatter(data[:, 0].tolist(), data[:, 1].tolist(),
                marker="x", c=np.array(label), cmap='rainbow', label="GMM")
    plt.scatter(mu[:, 0], mu[:, 1], color="k", label="Centers")
    plt.legend()
    plt.show()
    accuracy = Accuracy(real_label, label, classes)
    print("Accuracy: " + str(accuracy))
    print("Iterations: " + str(iter))

    plt.plot(LLs, color="k", label="Log Likelihood Change")
    plt.xlabel("Number of iterations")
    plt.ylabel("LL")
    plt.legend(loc='best')
    plt.show()
