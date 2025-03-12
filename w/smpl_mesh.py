import numpy as np
import open3d as o3d
import pyrender
import cv2
import os
from tqdm import tqdm

def render_smpl_mesh(vertices, faces, output_path, num_views=7, fps=60):
    # 准备 Open3D 渲染器
    render = o3d.visualization.rendering.OffscreenRenderer(800, 600)
    render.scene.set_background([1, 1, 1, 1])  # 白色背景
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = 'defaultUnlit'

    # 准备渲染的网格
    mesh = o3d.geometry.TriangleMesh()
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.paint_uniform_color([0.3, 0.3, 0.3])

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (800, 600))

    # 计算整个序列的边界框
    min_bound = np.min(vertices, axis=0)
    max_bound = np.max(vertices, axis=0)
    center = (min_bound + max_bound) / 2
    extent = max_bound - min_bound
    diagonal = np.linalg.norm(extent)
    distance = diagonal * 1.0  # 增加观察距离

    # 生成多角度视图
    for i in tqdm(range(num_views)):
        theta = 2 * np.pi * i / num_views
        phi = np.pi / 6  # 30 度
        eye = center + distance * np.array([
            np.cos(phi) * np.cos(theta),
            np.cos(phi) * np.sin(theta),
            np.sin(phi)
        ])

        # 设置相机参数
        render.setup_camera(
            vertical_field_of_view=45.0,
            center=center,
            eye=eye,
            up=[0, 0, 1],
            near_clip=0.1,
            far_clip=100.0
        )

        # 渲染并保存帧
        img = render.render_to_image()
        img = np.asarray(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        video.write(img)

    # 确保视频正确关闭
    video.release()
    print("\n视频保存完成!")

# 示例用法
vertices = np.load(r'F:\GitHub\Py-DL\w\tvertices.npy')  # 加载顶点数据
faces = np.load(r'F:\GitHub\Py-DL\w\faces.npy')  # 加载面数据
output_path = "F:\GitHub\Py-DL\w\output.mp4"
render_smpl_mesh(vertices, faces, output_path)