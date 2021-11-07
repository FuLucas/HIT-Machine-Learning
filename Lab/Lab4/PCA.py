import numpy as np
import os
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image


def pca(data, reduced_dimension):
    rows, columns = data.shape
    x_mean = np.sum(data, axis=0) / rows
    decentralized_x = data - x_mean  # 去中心化
    cov = decentralized_x.T.dot(decentralized_x)  # 计算协方差
    eigenvalues, feature_vectors = np.linalg.eig(cov)  # 特征值分解
    index = np.argsort(eigenvalues) # 特征值从小到大排序后的下标序列
    # 选取最大的特征值对应的特征向量
    feature_vectors = np.delete(feature_vectors, index[:columns - reduced_dimension], axis=1)
    return feature_vectors, x_mean

def generate_data(data_dimension, number=100):
    # Generate 2D or 3D data
    if data_dimension == 2:
        mean = [-2, 2]
        cov = [[1, 0], [0, 0.01]]
    elif data_dimension == 3:
        mean = [1, 2, 3]
        cov = [[0.01, 0, 0], [0, 1, 0], [0, 0, 1]]
    x = np.random.multivariate_normal(mean, cov, number)
    return x

def draw_data(dimension_draw, origin_data, pca_data):
    """ 将PCA前后的数据进行可视化对比 """
    if dimension_draw == 2:
        plt.scatter(origin_data[:, 0], origin_data[:, 1], marker="x", color="b", label="Origin Data")
        plt.scatter(pca_data[:, 0], pca_data[:, 1], color='r', label='PCA Data')
        plt.plot(pca_data[:, 0], pca_data[:, 1], c="k", label="vector", alpha=0.5)
    elif dimension_draw == 3:
        fig = plt.figure()
        ax = Axes3D(fig, auto_add_to_figure=False)
        fig.add_axes(ax)
        ax.scatter(origin_data[:, 0], origin_data[:, 1], origin_data[:, 2], marker="x", color="b", label='Origin Data')
        ax.scatter(pca_data[:, 0], pca_data[:, 1], pca_data[:, 2], color='r', label='PCA Data')
        ax.plot_trisurf(pca_data[:, 0], pca_data[:, 1], pca_data[:, 2], color="y", alpha=0.3)
    plt.legend(loc="best")
    plt.show()

def psnr(source, target):
    # Peak signal-to-noise ratio 峰值噪音比
    rmse = np.sqrt(np.mean((source - target) ** 2))
    return 20 * np.log10(255.0 / rmse)


if __name__ == "__main__":

    # test for generation data
    dimension = 3
    x = generate_data(dimension)
    w, mu_x = pca(x, dimension - 1)
    pca_data = (x - mu_x).dot(w).dot(w.T) + mu_x
    draw_data(dimension, x, pca_data)

    # psnr of one picture
    pic = Image.open('./data/61853_2019.jpg')
    pic_array = np.asarray(pic)
    psnrS = list()
    for k in range(50):
        w, mu_x = pca(pic_array, k)  # PCA降维
        pca_data = (pic_array - mu_x).dot(w).dot(w.T) + mu_x  # 重建数据
        psnrS.append(psnr(pic_array, pca_data))
    plt.plot(psnrS)
    plt.show()


    # picture data
    k_list = [30, 10, 5, 3, 1]
    size = len(k_list) + 1
    # list of all pictures
    file_list = os.listdir('data')

    for file in file_list:
        file_path = os.path.join('data', file)
        # Import picture
        pic = Image.open(file_path)
        # convert to ndarray
        pic_array = np.asarray(pic)

        # figure size setting
        plt.figure(figsize=(15,5))
        plt.subplot(1, size, 1)
        plt.title("Original Image")
        plt.imshow(pic_array)
        # Do not draw the axis
        plt.axis("off")

        for k in k_list:
            w, mu_x = pca(pic_array, k)  # PCA降维
            pca_data = (pic_array - mu_x).dot(w).dot(w.T) + mu_x  # 重建数据
            index = k_list.index(k)
            plt.subplot(1, size, index + 2)
            plt.title("k = " + str(k) + ", PSNR = " + "{:.2f}".format(psnr(pic_array, pca_data)) + "dB")
            plt.imshow(pca_data)
            plt.axis("off")

        plt.savefig('./images/' + str(file_list[file_list.index(file)]))
        plt.close()
