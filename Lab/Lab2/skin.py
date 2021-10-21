import numpy as np
import pandas as pd
from operation import *
from newton import *
from gradient_descent import *

def GetData():
    data_set = pd.read_csv("./Skin_NonSkin.csv")
    x = data_set.drop('y', axis=1)
    y = data_set['y']
    new_y = np.copy(y)
    new_y[y==2] = 0
    return x, new_y

if __name__ == '__main__':
    x, y = GetData()
    xPlus = x2xPlus(x)
    Train_x, Train_y, Test_x, Test_y = SplitData(xPlus, y)
    # 无惩罚项（正则项）的梯度下降
    beta_0 = np.zeros(xPlus.shape[1])
    # newton = Newton(Train_x, Train_y, beta_0, hyper=np.exp(-6))
    # k, beta = newton.fit()
    # print(accuracy(Test_x, Test_y, beta))
    # 带惩罚项（正则项）的梯度下降
    # gradient_descent = GradientDescent(Train_x, Train_y, beta_0, hyper=np.exp(-6))
    # k, beta, losses = gradient_descent.fit()
    # print(k, beta)
    # accuracy_gradient = accuracy(Test_x, Test_y, beta)
    # 0.9199959192001632

    newton = Newton(Train_x, Train_y, beta_0)
    k, beta = newton.fit()

    accuracy_newton = accuracy(Test_x, Test_y, beta)
    print(k, accuracy_newton)
    
    x_results = np.arange(0, 250, 0.25)
    y_results = np.arange(0, 250, 0.25)
    x_results, y_results = np.meshgrid(x_results, y_results)
    z_results = - (beta[0] + beta[1] * x_results + beta[2] * y_results) / beta[3]

    data_1 = x[y==1]
    data_0 = x[y==0]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data_1['B'],data_1['G'],data_1['R'],c='r',marker='^')
    ax.scatter(data_0['B'],data_0['G'],data_0['R'],c='g',marker='*')

    ax.plot_surface(x_results, y_results, z_results, rstride = 1, cstride = 1, cmap = plt.get_cmap('coolwarm'))

    ax.set_xlabel('X label')
    ax.set_ylabel('Y label')
    ax.set_zlabel('Z label')
    # plt.savefig('./skin.jpg')
    plt.show()
