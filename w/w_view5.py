import numpy as np
import matplotlib.pyplot as plt

import pickle
from mpl_toolkits.mplot3d import Axes3D
import warnings
import torch.nn.functional as F
import torch
# 使用网络+gaussian学习到的权重信息,使用原文的提取权重方式，和转换方式将贴合当前人形状的pose vertice形变到基准姿势，然后可视化
# 示例顶点坐标（6890, 3）
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


model_paths = r'F:\GitHub\Py-DL\w\FinalVolume.pkl'
with open(model_paths, 'rb') as f:
    volume = pickle.load(f, encoding='latin1')
g_volumes = volume['volume']#(25,32,32,32)

bbox_min_xyz=volume['min']
bbox_max_xyz=volume['max']
motion_scale_Rs=volume['motion_scale_Rs']
motion_Ts=volume['motion_Ts']

motion_scale_Rs=motion_scale_Rs[0]
motion_Ts=motion_Ts[0]


cnl_bbox_scale_xyz=2.0 / (bbox_max_xyz - bbox_min_xyz)

model_path =  r'F:\GitHub\Py-DL\w\posevertice.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f, encoding='latin1')
vertices = model['frame_000300']['posevertice']
vertices = torch.tensor(vertices, dtype=torch.float32)

# 获取每个顶点对应的24个权重
weights_list=[]
for i in range(24): # 24次循环
    pos = torch.matmul(motion_scale_Rs[i, :, :], vertices.T).T + motion_Ts[i, :] # torch.Size([614400, 3])把采样点从目标空间转换到基准空间（这么形容这一步并不完整，所以需要后面的权重再处理）。左乘3*3，再加上平移。所有采样点按照当前这个关节点的旋转矩阵和平移完成相应变化。
    pos = (pos - bbox_min_xyz[None, :]) \
                        * cnl_bbox_scale_xyz[None, :] - 1.0 # 将变化后的采样点标准化到 [-1, 1] 范围内，以适应 下面motion_weights 的采样网格。按照bbox的尺寸缩小
        

    weights = F.grid_sample(input=g_volumes[None, i:i+1, :, :, :], 
                                grid=pos[None, None, None, :, :],           
                                padding_mode='zeros', align_corners=True) #torch.Size([1, 1, 1, 1, 614400]) 采样点去当前关节点的 motion_weights_vol中查询，获得对应当前关节点的蒙皮权重
    weights = weights[0, 0, 0, 0, :, None] #torch.Size([614400, 1])每个采样点关于当前关节点的蒙皮权重
    weights_list.append(weights) # 汇聚了每个采样点关于所有关节点的蒙皮权重
backwarp_motion_weights = torch.cat(weights_list, dim=-1) #[N, 24])




backwarp_motion_weights_sum = torch.sum(backwarp_motion_weights, 
                                                dim=-1, keepdim=True) # torch.Size([614400, 1])将24个关节点的权重合并
weighted_motion_fields = []
for i in range(24): # 为每个采样点加权
            pos = torch.matmul(motion_scale_Rs[i, :, :], vertices.T).T + motion_Ts[i, :] # 如果采样点根据当前关节点完成变形后的采样点位置
            weighted_pos = backwarp_motion_weights[:, i:i+1] * pos # torch.Size([614400, 3])所有采样点新位置乘上当前这个关节点对应他的权重wi*x
            weighted_motion_fields.append(weighted_pos) #  24个，每个orch.Size([614400, 3])w1*x，w2*x，w3*x。。。。。。
x_skel = torch.sum(
                    torch.stack(weighted_motion_fields, dim=0), dim=0
                    ) / backwarp_motion_weights_sum.clamp(min=0.0001)  #torch.Size([614400, 3])



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# 归一化权重以适应颜色映射
#normalized_weights = (selected_weights - np.min(selected_weights)) / (np.max(selected_weights) - np.min(selected_weights))

# 绘制散点图
sc = ax.scatter(x_skel[:, 0], x_skel[:, 1], x_skel[:, 2], c='b', cmap='viridis', s=10)

# 添加颜色条
plt.colorbar(sc, label=f'Weight Dimension {i}')

# 设置标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 显示图形
plt.show()