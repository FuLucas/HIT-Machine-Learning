import math
import numpy as np

def sigmoid(x):
    "Numerically stable sigmoid function."
    if x <= 0:
        return 1.0 / (1.0 + np.exp(x))
    else:
        z = np.exp(-x)
        return z / (1.0 + z)

# def sigmoid(x):
#     return 1.0 / (1.0 + np.exp(x))


def Data(naive=True, N_1=500, N_0=500):
    """随机生成一组正例和一组负例，特征是二维的，可以选择条件是否满足朴素贝叶斯
    Args:
        naive (bool, optional): 是否满足朴素贝叶斯条件，True表示满足（即条件独立）. Defaults to True.
        N_1 (int, optional): 正例数量. Defaults to 500.
        N_0 (int, optional): 负例数量. Defaults to 500.
    Returns:
        array: 返回生成的特征数组x和标签y（1或0）
    """
    mean_1 = [1, 1]
    mean_0 = [-1, -1]
    cov_naive = [[0.5, 0], [0, 0.5]]
    cov_NOT_naive = [[2, 1], [1, 2]]
    y = np.zeros(N_1+N_0).astype(np.int32)
    # 满足朴素贝叶斯假设
    if naive:
        cov = cov_naive
    # 不满足朴素贝叶斯假设
    else:
        cov = cov_NOT_naive
    x_1 = np.random.multivariate_normal(mean_1, cov, size=N_1)
    x_0 = np.random.multivariate_normal(mean_0, cov, size=N_0)
    x = np.vstack((x_1, x_0))
    y = np.zeros(N_1+N_0).astype(np.int32)
    y[:N_1] = 1
    y[N_1:] = 0
    return x, y

def SplitData(x, y, trainRate=0.8):
    """划分测试集和训练集
    Args:
        x (array): 全部数据的特征集
        y (array): 一维数组，全部数据的标签，1或0
        trainRate (float, optional): 训练样本要占全部数据的比例. Defaults to 0.8.
    Returns:
        array: 训练样本和测试样本的特征和标签矩阵
    """
    N_1 = x[y == 1].shape[0]
    trainNum_1 = int(math.ceil(N_1 * trainRate))
    N_0 = x[y == 0].shape[0]
    trainNum_0 = int(math.ceil(N_0 * trainRate))
    # 训练集
    Train_x = np.vstack((x[:trainNum_1], x[N_1:N_1+trainNum_0]))
    Train_y = np.concatenate((y[:trainNum_1], y[N_1:N_1+trainNum_0]))
    # 测试集
    Test_x = np.vstack((x[trainNum_1:N_1], x[N_1+trainNum_0:]))
    Test_y = np.concatenate((y[trainNum_1:N_1], y[N_1+trainNum_0:]))

    return Train_x, Train_y, Test_x, Test_y

def x2xPlus(x):
    """在初始数据集之前加上一列1，使其符合beta的计算要求
    Args:
        x (array): 从数据中直接获取的特征，每一行都代表一个数据的各个特征
    Returns:
        array: 相较于x在前面多出了一列1
    """
    xPlus = np.column_stack((np.ones(x.shape[0]).T, x))
    return xPlus


def accuracy(Test_x, Test_y, beta):
    """ 计算训练结果的准确度，查看测试数据在分类面的哪一边与标签是否符合
    Args:
        Test_x (array): 特征数据，在这里Test_x的第一列包含1，这主要是为了配合beta的计算而设置的
        Test_y (array): 一维标签，正例为1，负例为0
        beta (array): 一维数组，(b, w0, w1, ... ,wn)
    Returns:
        [float]: 测试集中分类成功比例，即准确度
    """
    columns = len(Test_x)
    count = 0
    for i in range(columns):
        if sigmoid(beta @ Test_x[i]) < 0.5 and Test_y[i] == 1:
            count += 1
        elif sigmoid(beta @ Test_x[i]) > 0.5 and Test_y[i] == 0:
            count += 1
    return count / columns

