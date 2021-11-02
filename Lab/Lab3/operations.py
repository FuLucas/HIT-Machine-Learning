import numpy as np
from itertools import permutations


def Data(mean, N=100, naive=True):
    """随机生成一组高斯数据，特征是二维的，可以选择条件是否满足朴素贝叶斯
    Args:
        mean (array): 生成数据的均值中心，是一个一行二列的行向量.
        N (int, optional): 正例数量. Defaults to 100.
        naive (bool, optional): 是否满足朴素贝叶斯条件，True表示满足（即条件独立）. Defaults to True.
    Returns:
        array: 返回生成的特征数组x和标签y（1或0）
    """
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
    """计算分类标签的准确性（与源标签对比）
    Args:
        real_label (array): 样本数量长的列向量，由生成数据决定
        class_label (array): 样本数量长的列向量，由分类结果决定
        k (int): 样本类别数，标签是从 0 到 k-1
    Returns:
        [type]: [description]
    """
    # Full Permutation of labels 
    # The highest accuracy is taken as the result
    classes = list(permutations(range(k), k))
    counts = np.zeros(len(classes))
    for i in range(len(classes)):
        for j in range(real_label.shape[0]):
            if int(real_label[j]) == classes[i][int(class_label[j])]:
                counts[i] += 1
    return np.max(counts) / real_label.shape[0]
