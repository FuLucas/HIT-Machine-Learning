import random
import matplotlib.pyplot as plt
# import prettytable as pt
import numpy as np


class K_Means(object):
    def __init__(self, data, k, delta=1e-6):
        self.data = data
        self.k = k
        self.delta = delta
        # self.__data_rows, self.__data_columns = data.shape
        # 随机选择k个顶点作为初始簇中心点
        self.mu = self.data[random.sample(range(data.shape[0]), k)]
        # self.mu = self.__initial_center_not_random()
        self.sample_label = np.zeros(data.shape[0])

    def __k_means(self):
        iter = 0
        while True:
            iter += 1
            distance = np.zeros(self.k)
            # Determine the cluster label of each vector
            # according to the nearest cluster center
            for i in range(self.data.shape[0]):
                for j in range(self.k):
                    distance[j] = np.linalg.norm(
                        self.data[i] - self.mu[j])
                self.sample_label[i] = np.argmin(distance)
            # Calculate new k centers based on the new labels of all points
            new_mu = np.zeros((self.k, self.data.shape[1]))
            count = np.zeros(self.k)
            for i in range(self.data.shape[0]):
                new_mu[self.sample_label[i]] += self.data[i]
                count[self.sample_label[i]] += 1
            # for i in range(self.k):
            #     new_mu[i, :] = new_mu[i, :] / count[i]
            new_mu = [new_mu[i] / count[i] for i in range(self.k)]
            # Use the two-norm of the difference to express precision
            if np.linalg.norm(new_mu - self.mu) < self.delta:
                break
            else:
                self.mu = new_mu
        return self.mu, self.sample_label, iter

    # def k_means_not_random_center(self):
    #     """ 随机选择第一个簇中心点 再选择彼此距离最大的k个顶点作为初始簇中心点 """
    #     self.mu = self.__initial_center_not_random()
    #     return self.__k_means()

    # def __initial_center_not_random(self):
    #     """ 选择彼此距离尽可能远的K个点 """
    #     # 随机选第1个初始点
    #     mu_0 = np.random.randint(0, self.k) + 1
    #     mu = [self.data[mu_0]]
    #     # 依次选择与当前mu中样本点距离最大的点作为初始簇中心点
    #     for times in range(self.k - 1):
    #         temp_ans = []
    #         for i in range(self.__data_rows):
    #             temp_ans.append(np.sum([self.__euclidean_distance(
    #                 self.data[i], mu[j]) for j in range(len(mu))]))
    #         mu.append(self.data[np.argmax(temp_ans)])
    #     return np.array(mu)


if __name__ == "__main__":
    watermelon = np.array([[0.697, 0.46],
                           [0.774, 0.376],
                           [0.634, 0.264],
                           [0.608, 0.318],
                           [0.556, 0.215],
                           [0.403, 0.237],
                           [0.481, 0.149],
                           [0.437, 0.211],
                           [0.666, 0.091],
                           [0.243, 0.267],
                           [0.245, 0.057],
                           [0.343, 0.099],
                           [0.639, 0.161],
                           [0.657, 0.198],
                           [0.36, 0.37],
                           [0.593, 0.042],
                           [0.719, 0.103],
                           [0.359, 0.188],
                           [0.339, 0.241],
                           [0.282, 0.257],
                           [0.748, 0.232],
                           [0.714, 0.346],
                           [0.483, 0.312],
                           [0.478, 0.437],
                           [0.525, 0.369],
                           [0.751, 0.489],
                           [0.532, 0.472],
                           [0.473, 0.376],
                           [0.725, 0.445],
                           [0.446, 0.459]])
    kmeans = K_Means(watermelon, 3)
    mu, sample_label, iter = kmeans.__k_means()
    # result
    for i in range(3):
        plt.scatter(np.array(c_random[i])[:, 0], np.array(c_random[i])[:, 1], marker="x", label=str(i + 1))
    # type1_x = Train_x[Train_y==1][:,1]
    # type1_y = Train_x[Train_y==1][:,2]
    # type0_x = Train_x[Train_y==0][:,1]
    # type0_y = Train_x[Train_y==0][:,2]
    # plt.scatter(type1_x, type1_y, marker="x", c="b", label="Positive")
    # plt.scatter(type0_x, type0_y, marker="x", c="r", label="Negative")
