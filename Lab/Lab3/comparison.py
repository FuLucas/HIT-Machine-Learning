import numpy as np
import matplotlib.pyplot as plt
from operations import Data, Accuracy
from operations import *
from GMM import *
from k_means import *
import prettytable as pt


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

    k_means = K_Means(data, classes)
    kmeans_mu, kmeans_label, kmeans_iter = k_means.k_means()
    gmm = GaussianMixtureModel(data, classes)
    gmm_mu, gmm_label, gmm_iter, gmm_LLs = gmm.GMM()

    plt.subplot(131)
    plt.title("My Data")
    plt.scatter(data[:, 0].tolist(), data[:, 1].tolist(), marker="x", c=np.array(
        real_label), cmap='rainbow', label="Generated Data")
    plt.legend()

    plt.subplot(132)
    plt.title("K-Means")
    plt.scatter(data[:, 0].tolist(), data[:, 1].tolist(),
                marker="x", c=np.array(kmeans_label), cmap='rainbow', label="K-Means")
    plt.scatter(kmeans_mu[:, 0], kmeans_mu[:, 1], color="k", label="Centers")
    plt.legend()

    plt.subplot(133)
    plt.title("GMM")
    plt.scatter(data[:, 0].tolist(), data[:, 1].tolist(),
                marker="x", c=np.array(gmm_label), cmap='rainbow', label="GMM")
    plt.scatter(gmm_mu[:, 0], gmm_mu[:, 1], color="k", label="Centers")
    plt.legend()
    plt.show()

    plt.plot(gmm_LLs, color="k", label="Log Likelihood Change")
    plt.xlabel("Number of iterations")
    plt.ylabel("LL")
    plt.legend(loc='best')
    plt.show()

    kmeans_accuracy = Accuracy(real_label, kmeans_label, classes)
    gmm_accuracy = Accuracy(real_label, gmm_label, classes)
    tb = pt.PrettyTable()
    tb.field_names = ["Methods", "Accuracy", "Iterations"]
    tb.add_row(["K-Means", kmeans_accuracy, kmeans_iter])
    tb.add_row(["GMM", gmm_accuracy, gmm_iter])
    print(tb)