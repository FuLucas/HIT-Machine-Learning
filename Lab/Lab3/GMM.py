import numpy as np
import random
from scipy.stats import multivariate_normal
import collections
# import pandas as pd
import matplotlib.pyplot as plt
from operations import Data, Accuracy


class GaussianMixtureModel(object):
    """ 高斯混合聚类EM算法 """

    def __init__(self, data, k, delta=1e-12, max_iteration=1000):
        self.data = data
        self.k = k
        self.delta = delta
        self.max_iteration = max_iteration
        self.__alpha = np.ones(self.k) * (1.0 / self.k)
        self.__mu = np.array(self.data[random.sample(range(data.shape[0]), k)])
        # self.__mu = self.__initial_center_not_random()
        self.__Sigma = self.__init_Sigma()
        self.label = None
        self.__gamma = None
        self.__LL = -np.inf

    def __initial_center_not_random(self):
        """ 选择彼此距离尽可能远的K个点 """
        mu = np.array(self.data[random.sample(range(data.shape[0]), 1)])
        for times in range(self.k - 1):
            distance = []
            for i in range(self.data.shape[0]):
                sum = np.sum([np.linalg.norm(self.data[i] - mu[j]) for j in range(times+1)])
                distance.append(sum)
            mu = np.vstack((mu, self.data[np.argmax(distance)]))
        print(mu)
        return mu

    def __init_Sigma(self):
        Sigma = collections.defaultdict(list)
        for i in range(self.k):
            Sigma[i] = np.eye(self.data.shape[1], dtype=float) * 0.1
        return Sigma

    def __likelihoods(self):
        likelihoods = np.zeros((self.data.shape[0], self.k))
        for i in range(self.k):
            likelihoods[:, i] = multivariate_normal.pdf(
                self.data, self.__mu[i], self.__Sigma[i])
        return likelihoods

    def __expectation(self):
        # 求期望 E
        # Gaussian mixture distribution
        Mixture_distribution = self.__likelihoods() * self.__alpha    # (m,k)
        sum_likelihoods = np.sum(Mixture_distribution, axis=1).reshape(-1, 1)
        self.__LL = np.sum(np.log(sum_likelihoods))
        print(self.__LL)
        self.__gamma = Mixture_distribution / sum_likelihoods    # (m,k)
        self.label = self.__gamma.argmax(axis=1)    # (m,)

    def __maximization(self):
        # 最大化 M
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
    data2 = Data([4, 1])
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
