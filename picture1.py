import matplotlib.pyplot as plt

# 数据准备
x = [1, 2, 3, 4, 5]  # x轴数据
y = [2, 3, 5, 7, 11] # y轴数据

# 创建折线图
plt.figure(figsize=(10, 5))  # 设置图形的大小
plt.plot(x, y, marker='o', linestyle='-', color='b')  # 绘制折线图，使用蓝色，点标记

# 添加标题和标签
plt.title('1023041004Fanwenyi')  # 图形标题
plt.xlabel('X')      # x轴标签
plt.ylabel('Y')      # y轴标签

# 显示图形
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# 数据准备
categories = ['A', 'B', 'C', 'D']
values = [10, 20, 15, 25]

# 创建柱状图
plt.figure(figsize=(8, 6))  # 设置图形的大小
bar_plot = plt.bar(categories, values, color='skyblue')  # 绘制柱状图，设置颜色

# 添加标题和标签
plt.title('1023041004Fanwenyi')  # 图形标题
plt.xlabel('Categories')         # x轴标签
plt.ylabel('Values')             # y轴标签

# 显示数值标签
for bar in bar_plot:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 1), ha='center', va='bottom')

# 显示图形
plt.show()