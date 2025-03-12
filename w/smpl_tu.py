import numpy as np
import matplotlib.pyplot as plt
import pickle

# 加载模型顶点数据
model_paths = 'F:\\GitHub\\Py-DL\\w\\posevertice.pkl'
with open(model_paths, 'rb') as f:
    posevertice = pickle.load(f, encoding='latin1')
v_template = posevertice['frame_000073']['posevertice']

# 定义可视化函数
def visualize_smpl_vertices(vertices, output_prefix):
    """
    可视化SMPL模型的6890个顶点，并生成四个视角的图片。

    Args:
        vertices: np.ndarray, (6890, 3)，SMPL模型的顶点坐标。
        output_prefix: str, 输出图像的文件名前缀。
    """
    # 提取x, y, z坐标
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]
    
    # 定义四种视角
    view_angles = [
        {'elev': 0, 'azim': 0},    # 左侧
        {'elev': 0, 'azim': 180},  # 右侧
        {'elev': 90, 'azim': 0},   # 正面
        {'elev': 270, 'azim': 0}   # 背面
    ]
    
    # 生成四个视角的图像
    for i, angle in enumerate(view_angles):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制顶点
        ax.scatter(x, y, z, c='b', s=1, marker='.', alpha=1)  # 蓝色顶点，不带透明度
        
        # 设置视角
        ax.view_init(elev=angle['elev'], azim=angle['azim'])
        
        # 设置背景颜色为白色
        ax.set_facecolor('white')
        
        # 去掉网格线
        ax.grid(False)
        
        # 去掉坐标轴
        ax.set_axis_off()
        
        # 保存图像
        plt.savefig(f"{output_prefix}_view_{i+1}.png", bbox_inches='tight', pad_inches=0)
        plt.close()

# 调用函数生成图像
visualize_smpl_vertices(v_template, 'smpl_vertices')