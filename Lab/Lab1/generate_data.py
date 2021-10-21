import numpy as np

def geneData(number, loc, scale):
    """ 生成训练样本数据
    Args:
        number: 在[0,1]范围内选取的数据个数
        loc: 高斯分布均值
        scale: 高斯分布的标准偏差
    Returns:
        Xn: 在[0,1]范围内均匀分布的一维数组
        T: sin(2Πx)，高斯噪音标准差为scale，均值为loc
    """
    Xn = np.linspace(0, 1, num=number)
    noise = np.random.normal(loc=loc, scale=scale, size=len(Xn))
    T = np.sin(2 * np.pi * Xn) + noise
    return Xn, T


def generateX(row, degree):
    """ 将一维数组 row 转化为len(row) * (degree + 1)矩阵X
    Args:
        row: 一维数组，样本横坐标
        degree: 多项式最高度数
    Returns:
        len(row) * (degree + 1) 的矩阵，每一行表示 row 中每个元素的0到degree次幂
    """
    X = np.empty((len(row), degree + 1), dtype=np.double)
    pow = np.arange(0, degree+1)
    for i in range(len(row)):
        row_i = np.ones(degree + 1) * row[i]
        row_i = np.power(row_i, pow)
        X[i] = row_i
    return X
