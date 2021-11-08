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

def psnr(source, target):
    # Peak signal-to-noise ratio 峰值噪音比
    rmse = np.sqrt(np.mean((source - target) ** 2))
    return 20 * np.log10(255.0 / rmse)


if __name__ == "__main__":

    number = 100
    # test for generation data
    mean_2 = [-2, 2]
    cov_2 = [[1, 0], [0, 0.01]]
    x = np.random.multivariate_normal(mean_2, cov_2, number)
    w, mu_x = pca(x, 1)
    pca_data = (x - mu_x).dot(w).dot(w.T) + mu_x
    # draw result
    plt.scatter(x[:, 0], x[:, 1], marker="x", color="b", label="Origin Data")
    plt.scatter(pca_data[:, 0], pca_data[:, 1], color='r', label='PCA Data')
    plt.plot(pca_data[:, 0], pca_data[:, 1], c="k", label="Vector", alpha=0.5)
    plt.legend(loc="best")
    plt.savefig('./images_result/2D21D.svg')
    plt.show()

    mean_3 = [1, 2, 3]
    cov_3 = [[0.01, 0, 0], [0, 1, 0], [0, 0, 1]]
    x = np.random.multivariate_normal(mean_3, cov_3, number)
    # PCA降维
    w, mu_x = pca(x, 2)
    # 重建数据
    pca_data = (x - mu_x).dot(w).dot(w.T) + mu_x
    # draw result
    fig = plt.figure()
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.scatter(x[:, 0], x[:, 1], x[:, 2], marker="x", color="b", label='Origin Data')
    ax.scatter(pca_data[:, 0], pca_data[:, 1], pca_data[:, 2], color='r', label='PCA Data')
    ax.plot_trisurf(pca_data[:, 0], pca_data[:, 1], pca_data[:, 2], color="y", alpha=0.3)
    plt.legend(loc="best")
    plt.savefig('./images_result/3D22D.svg')
    plt.show()

    # psnr of one picture
    pic = Image.open('./data/61853_2019.jpg')
    pic_array = np.array(pic)
    psnrS = list()
    for k in range(50):
        # PCA降维
        w, mu_x = pca(pic_array, k)
        # 重建数据
        pca_data = (pic_array - mu_x).dot(w).dot(w.T) + mu_x
        psnrS.append(psnr(pic_array, pca_data))
    # print(np.array(psnrS))
    plt.plot(list(np.arange(50)), psnrS)
    plt.savefig('./images_result/psnr.svg')
    plt.show()

    # picture data
    k_list = [30, 10, 5, 3, 1]
    size = len(k_list) + 1
    # list of all pictures
    file_list = os.listdir('data')
    f = open('PSNR.txt', 'w')
    for file in file_list:
        file_path = os.path.join('data', file)
        # Import picture
        pic = Image.open(file_path)
        # convert to ndarray
        pic_array = np.asarray(pic)
        x = pic_array
        res = file + " : "

        for k in k_list:
            w, mu_x = pca(pic_array, k)
            pca_data = (pic_array - mu_x).dot(w).dot(w.T) + mu_x
            x = np.concatenate((x, pca_data), axis=1)
            psnr_k = psnr(pic_array, pca_data)
            res = res + "k=" + str(k) + ", PSNR=" + "{:.2f}".format(psnr_k) + "dB; "
        im = Image.fromarray(x)
        im = im.convert('L')
        im.save('./images/' + str(file_list[file_list.index(file)]))
        f.write(res + '\n')
