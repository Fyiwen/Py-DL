import numpy as np
import matplotlib.pyplot as plt

import pickle
from mpl_toolkits.mplot3d import Axes3D
import warnings
# 用于可视化预定义文件中给出的权重情况

# 示例顶点坐标（6890, 3）
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

model_path =  r'F:\GitHub\Py-DL\w\Tvertice.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f, encoding='latin1')
vertices = model['frame_000000']['Tvertice']




model_paths = r'F:\GitHub\Py-DL\w\basicmodel_m_lbs_10_207_0_v1.0.0.pkl'
with open(model_paths, 'rb') as f:
    smpl_model = pickle.load(f, encoding='latin1')
weights = smpl_model['weights']
print(weights)

# 示例蒙皮权重（6890, 24）
#weights = np.random.rand(6890, 24)  # 随机生成蒙皮权重

# 绘制3D散点图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 对于每个顶点的24个维度，你可以选择其中一个维度作为颜色指示
# 这里假设我们选择第0维作为颜色指示
dimension = 0
selected_weights = weights[:, dimension]

# 归一化权重以适应颜色映射
#normalized_weights = (selected_weights - np.min(selected_weights)) / (np.max(selected_weights) - np.min(selected_weights))

# 绘制散点图
sc = ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c=selected_weights, cmap='viridis', s=10)

# 添加颜色条
plt.colorbar(sc, label=f'Weight Dimension {dimension}')

# 设置标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 显示图形
plt.show()