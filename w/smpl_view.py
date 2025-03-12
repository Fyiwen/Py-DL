import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2 

# 示例：SMPL模型的顶点坐标 (6890个顶点，每个顶点有x, y, z三个坐标)
# 假设你的顶点坐标存储在一个 (6890, 3) 的数组中
# 示例数据: 生成随机点代替实际SMPL模型顶点
import pickle


model_paths ='F:\\GitHub\\Py-DL\\w\\posevertice.pkl'
with open(model_paths, 'rb') as f:
    posevertice = pickle.load(f, encoding='latin1')
v_template = posevertice['frame_000373']['posevertice']



def visualize_smpl_vertices(vertices):
    """
    可视化SMPL模型的6890个顶点坐标。

    Args:
        vertices: np.ndarray, (N, 3)，SMPL模型的顶点坐标。
    """
    # 提取x, y, z坐标
    #vertices = vertices.numpy()


    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]
    
    # 创建一个3D图形
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制散点图
    ax.scatter(x, y, z, c='b', s=1, marker='.', alpha=0.8)
    
    # 设置图像标题和坐标轴标签
    ax.set_title("SMPL Model Vertices Visualization")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_zlabel("Z Coordinate")
    
    # 显示图像
    plt.show()


visualize_smpl_vertices(v_template)
