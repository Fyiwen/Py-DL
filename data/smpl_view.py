import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2 

# 示例：SMPL模型的顶点坐标 (6890个顶点，每个顶点有x, y, z三个坐标)
# 假设你的顶点坐标存储在一个 (6890, 3) 的数组中
# 示例数据: 生成随机点代替实际SMPL模型顶点
import pickle
'''
model_paths = 'F:\\GitHub\\Py-DL\\basicmodel_m_lbs_10_207_0_v1.0.0.pkl'
with open(model_paths, 'rb') as f:
    smpl_model = pickle.load(f, encoding='latin1')
v_template = smpl_model['v_template']
'''

'''
model_paths='F:\\GitHub\\Py-DL\\mesh_infos.pkl'
with open(model_paths, 'rb') as f:
    smpl_model = pickle.load(f, encoding='latin1')
v_template = smpl_model['000008']['tpose_joints']
'''
'''
model_paths='F:\\GitHub\\Py-DL\\zmesh_infos.pkl'
with open(model_paths, 'rb') as f:
    smpl_model = pickle.load(f, encoding='latin1')
v_template = smpl_model['frame_000023']['joints'] # (24,3)
#v_template = smpl_model['000013']['joints'] # (24,3)
'''

'''
model_paths='F:\\GitHub\\Py-DL\\tvertices.npy'
with open(model_paths, 'rb') as f:
    smpl_model = np.load(f, allow_pickle=True)
v_template = smpl_model
'''
#R_h = smpl_model['000073']['Rh']
#R_h = np.array(R_h, dtype=np.float32)  # 确保 R_h 是正确的数组
#R, _ = cv2.Rodrigues(R_h)  # 转换为 (3, 3) 旋转矩阵
# 对 v_template 应用旋转变换
#v_template = np.dot(v_template, R.T) 

'''
model_paths ='F:\\GitHub\\Py-DL\\cameras.pkl'
with open(model_paths, 'rb') as f:
    camera_model = pickle.load(f, encoding='latin1')
K = camera_model['000073']['extrinsics']
K = np.array(K, dtype=np.float32) 
K=K[:3,:3]
v_template = np.dot(v_template, K.T) 
'''
''''
model_paths ='F:\\GitHub\\Py-DL\\posevertice.pkl'
with open(model_paths, 'rb') as f:
    posevertice = pickle.load(f, encoding='latin1')
v_template = posevertice['frame_000073']['posevertice']
'''




model_paths='F:\\GitHub\\Py-DL\\zju\\C.pkl'
with open(model_paths, 'rb') as f:
    smpl_model = pickle.load(f, encoding='latin1')

v_template = torch.cat(smpl_model, dim=0)
v_template=v_template.reshape(-1,3)
print(v_template.shape)

step = 128
v_template = v_template[80::step]
print(v_template)
#v_template=v_template[2558694-10000:2558694+10000]



model_paths2='F:\\GitHub\\Py-DL\\zju\\rgb.pkl'
with open(model_paths2, 'rb') as f:
    color = pickle.load(f, encoding='latin1')
v_template_color = torch.cat(color, dim=0)  # (24,3)

v_template_color = torch.clamp(v_template_color, min=0.0, max=1.0)
v_template_color = v_template_color[70::step]
#v_template_color=v_template_color[2558694-10000:2558694+10000]
#print(v_template_color)





def visualize_smpl_vertices3(vertices,color):
    """
    可视化SMPL模型的6890个顶点坐标。

    Args:
        vertices: np.ndarray, (N, 3)，SMPL模型的顶点坐标。
    """
    # 提取x, y, z坐标
    vertices = vertices.numpy()
    color = color.numpy()

    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]
    
    # 创建一个3D图形
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制散点图
    ax.scatter(x, y, z, c=color, s=1, marker='.', alpha=0.8)
    
    # 设置图像标题和坐标轴标签
    ax.set_title("SMPL Model Vertices Visualization")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_zlabel("Z Coordinate")
    
    # 显示图像
    plt.show()
'''
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
'''
# 调用函数可视化顶点
visualize_smpl_vertices3(v_template,v_template_color)
#visualize_smpl_vertices(v_template)
