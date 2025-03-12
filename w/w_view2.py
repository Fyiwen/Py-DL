import numpy as np
import matplotlib.pyplot as plt

import pickle
from mpl_toolkits.mplot3d import Axes3D
import warnings
#用于可视化代码实现的gaussian volume权重分布
# 示例顶点坐标（6890, 3）
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)






model_paths = r'F:\GitHub\Py-DL\w\weightVolume.pkl'
with open(model_paths, 'rb') as f:
    volume = pickle.load(f, encoding='latin1')
g_volumes = volume['volume']#(25,32,32,32)
print(g_volumes)
bbox_min_xyz=volume['min']
bbox_max_xyz=volume['max']

# 选择一个特定的骨骼或背景体积进行可视化（例如，第0个骨骼）
selected_volume = g_volumes[0] #(32,32,32)

# 提取网格坐标
grid_size = 32

min_x, min_y, min_z = bbox_min_xyz
max_x, max_y, max_z = bbox_max_xyz
zgrid, ygrid, xgrid = np.meshgrid(
    np.linspace(min_z, max_z, grid_size),
    np.linspace(min_y, max_y, grid_size),
    np.linspace(min_x, max_x, grid_size),
    indexing='ij') 


# 展平坐标和权重
x_coords = xgrid.flatten()
y_coords = ygrid.flatten()
z_coords = zgrid.flatten()
weights = selected_volume.flatten()

# 归一化权重
#weights_normalized = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))

# 过滤权重值小于 0.5 的点
mask = weights >= 0.5
x_coords = x_coords[mask]
y_coords = y_coords[mask]
z_coords = z_coords[mask]
weights = weights[mask]

# 创建3D散点图
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# 绘制散点图，颜色根据权重值
sc = ax.scatter(x_coords, y_coords, z_coords, c=weights, cmap='viridis', s=10)

# 添加颜色条
plt.colorbar(sc, label='Weight Value')

# 设置轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 设置标题
ax.set_title('3D Grid Visualization with Weighted Vertices')

# 显示图形
plt.show()