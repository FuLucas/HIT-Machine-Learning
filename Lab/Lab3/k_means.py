import random
import matplotlib.pyplot as plt
import numpy as np
from operations import Data, Accuracy


class K_Means(object):
    def __init__(self, data, k, delta=1e-6):
        """ 初始化数据集
        Args:
            data (array): 数据数组，每行表示一个样本地特征，行数表示样本数
            k (int): 要分成的类数
            delta (double, optional): tolerance，当类别中心的变化的二范数
                                    小于这个值就认为分类结束. Defaults to 1e-6.
        """
        self.data = data
        self.k = k
        self.delta = delta
        # Randomly select k vertices as the initial cluster center point
        self.mu = np.array(self.data[random.sample(range(data.shape[0]), k)])
        self.label = np.zeros(data.shape[0])

    def k_means(self):
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
                self.label[i] = np.argmin(distance)
            # Calculate new k centers based on the new labels of all points
            new_mu = np.zeros((self.k, self.data.shape[1]))
            count = np.zeros(self.k)
            for i in range(self.data.shape[0]):
                new_mu[int(self.label[i])] += self.data[i]
                count[int(self.label[i])] += 1
            new_mu = new_mu / count[:, None]
            # Use the two-norm of the difference to express precision
            if np.linalg.norm(new_mu - self.mu) < self.delta:
                break
            else:
                self.mu = new_mu
        return self.mu, self.label, iter


if __name__ == "__main__":
    data0 = Data([1, 1])
    data1 = Data([4, 4])
    data2 = Data([4, 1])
    classes = 3
    data = np.vstack((data0, data1, data2))
    real_label = np.concatenate(
        (np.zeros(len(data0)), np.ones(len(data1)), 2 * np.ones(len(data2))))
    plt.subplot(121)
    plt.title("My Data")
    plt.scatter(data[:, 0].tolist(), data[:, 1].tolist(), marker="x", c=np.array(
        real_label), cmap='rainbow', label="Generated Data")
    plt.legend()

    kmeans = K_Means(data, classes)
    mu, label, iter = kmeans.k_means()
    # show result
    plt.subplot(122)
    plt.title("K-Means")
    plt.scatter(data[:, 0].tolist(), data[:, 1].tolist(),
                marker="x", c=np.array(label), cmap='rainbow', label="K-Means")
    plt.legend()

    plt.show()
    accuracy = Accuracy(real_label, label, classes)
    print("Accuracy: " + str(accuracy))
    print("Iterations: " + str(iter))
