import numpy as np
from itertools import permutations


def Data(mean, N=100, naive=True):
    cov_naive = [[0.5, 0], [0, 0.5]]
    cov_NOT_naive = [[2, 1], [1, 2]]
    # 满足朴素贝叶斯假设
    if naive:
        cov = cov_naive
    # 不满足朴素贝叶斯假设
    else:
        cov = cov_NOT_naive
    x = np.random.multivariate_normal(mean, cov, size=N)
    return x

def Accuracy(real_label, class_label, k):
    """计算聚类准确率
    """
    classes = list(permutations(range(k), k))
    counts = np.zeros(len(classes))
    for i in range(len(classes)):
        for j in range(real_label.shape[0]):
            if int(real_label[j]) == classes[i][int(class_label[j])]:
                counts[i] += 1
    return np.max(counts) / real_label.shape[0]
