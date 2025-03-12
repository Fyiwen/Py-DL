import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def plot_linear_regression():
    # 创建随机数据集
    np.random.seed(0)
    X = np.sort(np.random.rand(100, 1), axis=0)
    y = np.sin(X).ravel()  # 真实的非线性关系

    # 添加一些噪声
    y[::5] += np.random.normal(0, 0.1, y[::5].shape)

    # 绘制散点图
    plt.scatter(X, y, color='blue', label='Data points')

    # 线性回归模型
    model = LinearRegression()
    model.fit(X, y)

    # 绘制回归直线
    X_curve = np.linspace(X.min(), X.max(), 100).reshape(100, 1)
    y_curve = model.predict(X_curve)
    plt.plot(X_curve, y_curve, color='red', label='Linear regression line')

    # 添加图例和标题
    plt.legend()
    plt.title('1023041004 Fanwenyi')
    plt.xlabel('X', fontsize=18)  # 设置x轴标签字体大小为16
    plt.ylabel('y', fontsize=18)  # 设置y轴标签字体大小为16

    # 设置坐标轴上的数值字体大小
    plt.xticks(fontsize=18)  # 设置x轴数值字体大小为14
    plt.yticks(fontsize=18)  # 设置y轴数值字体大小为14

    plt.show()

# 调用函数绘制图表
plot_linear_regression()