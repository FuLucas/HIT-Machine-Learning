import random
import matplotlib.pyplot as plt
import numpy as np
from operations import Data, Accuracy


class K_Means(object):
    def __init__(self, data, k, delta=1e-6):
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
    data1 = Data([3, 3])
    data2 = Data([3, 1])
    data = np.vstack((data0, data1, data2))
    real_lable = np.zeros(data.shape[0]).astype(np.int32)
    real_lable[:data0.shape[0]] = 0
    real_lable[data0.shape[0]:data0.shape[0]+data1.shape[0]] = 1
    real_lable[data0.shape[0]+data1.shape[0]:] = 2
    plt.subplot(121)
    plt.title("My Data")
    plt.scatter(data0[:, 0], data0[:, 1], marker="x", c="b", label="My Data 1")
    plt.scatter(data1[:, 0], data1[:, 1], marker="x", c="r", label="My Data 2")
    plt.scatter(data2[:, 0], data2[:, 1], marker="x", c="k", label="My Data 2")
    plt.legend()

    kmeans = K_Means(data, 3)
    mu, label, iter = kmeans.k_means()
    print(iter)
    # show result
    plt.subplot(122)
    plt.title("K-Means")
    type0 = data[np.where(label == 0)]
    type1 = data[np.where(label == 1)]
    type2 = data[np.where(label == 2)]
    plt.scatter(type0[:, 0], type0[:, 1], marker="x", c="b", label="Mean 1")
    plt.scatter(type1[:, 0], type1[:, 1], marker="x", c="r", label="Mean 2")
    plt.scatter(type2[:, 0], type2[:, 1], marker="x", c="k", label="Mean 3")
    plt.legend()

    plt.show()
    accuracy = Accuracy(real_lable, label, 3)
    print(accuracy)
