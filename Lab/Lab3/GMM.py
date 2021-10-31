import numpy as np
import random
from scipy.stats import multivariate_normal
import collections
import pandas as pd


class GaussianMixtureModel(object):
    """ 高斯混合聚类EM算法 """

    def __init__(self, data, k=3, delta=1e-12, max_iteration=1000):
        self.data = data
        self.k = k
        self.delta = delta
        self.max_iteration = max_iteration
        self.__alpha = np.ones(self.k) * (1.0 / self.k)
        self.mu = np.array(self.data[random.sample(range(data.shape[0]), k)])
        # self.__mu = self.__initial_center_not_random()
        self.__Sigma = self.__init_params()
        self.sample_assignments = None
        self.c = collections.defaultdict(list)
        self.__gamma = None

    def __initial_center_not_random(self):
        """ 选择彼此距离尽可能远的K个点 """
        # 随机选第1个初始点
        mu_0 = np.random.randint(0, self.k) + 1
        mu = [self.data[mu_0]]
        # 依次选择与当前mu中样本点距离最大的点作为初始簇中心点
        for times in range(self.k-1):
            temp_ans = []
            for i in range(self.data.shape[0]):
                temp_ans.append(np.sum([np.linalg.norm(
                    self.data[i], mu[j]) for j in range(len(mu))]))
            mu.append(self.data[np.argmax(temp_ans)])
        return np.array(mu)

    def __init_params(self):
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
        weighted_likelihoods = self.__likelihoods() * self.__alpha    # (m,k)
        sum_likelihoods = np.expand_dims(
            np.sum(weighted_likelihoods, axis=1), axis=1)  # (m,1)
        print(np.log(np.prod(sum_likelihoods)))     # 输出似然值
        self.__gamma = weighted_likelihoods / sum_likelihoods    # (m,k)
        self.sample_assignments = self.__gamma.argmax(axis=1)    # (m,)
        for i in range(self.data.shape[0]):
            self.c[self.sample_assignments[i]].append(self.data[i].tolist())

    def __maximization(self):
        # 最大化 M
        for i in range(self.k):
            # 提取每一列 作为列向量 (m, 1)
            gamma = np.expand_dims(self.__gamma[:, i], axis=1)
            mean = (gamma * self.data).sum(axis=0) / gamma.sum()
            covariance = (self.data - mean).T.dot((self.data -
                                                   mean) * gamma) / gamma.sum()
            self.__mu[i], self.__Sigma[i] = mean, covariance    # 更新参数
        self.__alpha = self.__gamma.sum(axis=0) / self.data.shape[0]

    def GMM(self):
        print("GMM")
        last_alpha = self.__alpha
        last_mu = self.__mu
        last_Sigma = self.__Sigma
        for i in range(self.max_iteration):
            print(i)
            self.__expectation()
            self.__maximization()
            # Termination condition, Sigma, mu and alpha hardly change
            diff = np.linalg.norm(last_alpha - self.__alpha) + \
                np.linalg.norm(last_mu - self.__mu) + \
                np.sum([np.linalg.norm(last_Sigma[i] - self.__Sigma[i])
                    for i in range(self.k)])
            if diff > self.delta:
                last_Sigma = self.__Sigma
                last_mu = self.__mu
                last_alpha = self.__alpha
            if diff <= self.delta:
                break
        self.__expectation()
        return self.__mu, self.c
