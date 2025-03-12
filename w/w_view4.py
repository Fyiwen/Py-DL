import numpy as np
import matplotlib.pyplot as plt

import pickle
from mpl_toolkits.mplot3d import Axes3D
import warnings
import torch.nn.functional as F
import torch
# 用于查看基准顶点能从，网络+gau学习后的volumn中查询到的权重情况
# 示例顶点坐标（6890, 3）
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


model_paths = r'F:\GitHub\Py-DL\w\FinalVolume.pkl'
with open(model_paths, 'rb') as f:
    volume = pickle.load(f, encoding='latin1')
g_volumes = volume['volume']#(25,32,32,32)

bbox_min_xyz=volume['min']
bbox_max_xyz=volume['max']

cnl_bbox_scale_xyz=2.0 / (bbox_max_xyz - bbox_min_xyz)

model_path =  r'F:\GitHub\Py-DL\w\Tvertice.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f, encoding='latin1')
vertices = model['frame_000000']['Tvertice']
vertices = torch.tensor(vertices, dtype=torch.float32)

vertices = (vertices - bbox_min_xyz[None, :]) \
                            * cnl_bbox_scale_xyz[None, :] - 1.0 # 这一步确实时需要的，不做这个胸口坐标系的顶点是去这个网格不对应的地方查询
i=24
weights = F.grid_sample(input=g_volumes[None, i:i+1, :, :, :], 
                        grid=vertices[None, None, None, :, :],           
                        padding_mode='zeros', align_corners=True) #torch.Size([1, 1, 1, 1, 614400]) 采样点去当前关节点的 motion_weights_vol中查询，获得对应当前关节点的蒙皮权重
weights = weights[0, 0, 0, 0, :, None] #torch.Size([614400, 1])每个采样点关于当前关节点i的蒙皮权重



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# 归一化权重以适应颜色映射
#normalized_weights = (selected_weights - np.min(selected_weights)) / (np.max(selected_weights) - np.min(selected_weights))

# 绘制散点图
sc = ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c=weights, cmap='viridis', s=10)

# 添加颜色条
plt.colorbar(sc, label=f'Weight Dimension {i}')

# 设置标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 显示图形
plt.show()