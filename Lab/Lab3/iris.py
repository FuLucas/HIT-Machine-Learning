import numpy as np
import pandas as pd
from operations import *
from GMM import *
from k_means import *

# 鸢尾花

def GetData():
    data_set = pd.read_csv("./data/iris.csv")
    x = data_set.drop('class', axis=1)
    y = data_set['class']
    y = y.replace('Iris-setosa', 0).replace('Iris-versicolor', 1).replace('Iris-virginica', 2)
    return x, y

if __name__ == '__main__':
    x, y = GetData()
    data = np.array(x)
    real_label = np.array(y)
    classes = np.unique(real_label).shape[0]

    k_means = K_Means(data, classes)
    kmeans_mu, kmeans_label, kmeans_iter = k_means.k_means()

    gmm = GaussianMixtureModel(data, classes)
    gmm_mu, gmm_label, gmm_iter, gmm_LLs = gmm.GMM()

    plt.plot(gmm_LLs, color="k", label="Log Likelihood Change")
    plt.xlabel("Number of iterations")
    plt.ylabel("LL")
    plt.legend(loc='best')
    plt.show()

    kmeans_accuracy = Accuracy(real_label, kmeans_label, classes)
    gmm_accuracy = Accuracy(real_label, gmm_label, classes)
    print("K-Means Accuracy: ", kmeans_accuracy)
    print("GMM Accuracy", gmm_accuracy)
