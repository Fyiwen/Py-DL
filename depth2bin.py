import os
import numpy as np
from PIL import Image

# 定义路径
image_path = 'F:\\pycode\\Depth-Anything\\assets\\image'
depth_path = 'F:\\pycode\\Depth-Anything\\depth_vis'
output_path = 'F:\\pycode\\Depth-Anything\\transd2boutput'

# 确保输出目录存在
os.makedirs(output_path, exist_ok=True)

# 获取图像文件列表
image_files = [f for f in os.listdir(image_path) if f.endswith('.png')]
depth_files = [f for f in os.listdir(depth_path) if f.endswith('_depth.png')]

# 排序确保匹配
image_files.sort()
depth_files.sort()

# 检查数量是否一致
assert len(image_files) == len(depth_files), "数量不匹配"

# 处理每个文件
for idx, (img_file, depth_file) in enumerate(zip(image_files, depth_files)):
    # 读取深度图像
    depth_img_path = os.path.join(depth_path, depth_file)
    depth_image = Image.open(depth_img_path)
    
    # 将图像转换为灰度图像（单通道）
    depth_image = depth_image.convert('L')
    depth_array = np.array(depth_image).astype(np.float32)
    
     # 调整深度图大小到 (600, 800)
    depth_image_resized = depth_image.resize((800, 600), Image.NEAREST)
    depth_array_resized = np.array(depth_image_resized).astype(np.float32)

    # 获取图像尺寸
    height, width = depth_array_resized.shape
    channels = 1
    print(height,width)
    # 生成文件头
    header = f"{width}&{height}&{channels}&"
    
    # 创建输出文件路径
    output_file = os.path.join(output_path, f"{idx}.jpg.geometric.bin")
    
    # 写入文件
    with open(output_file, 'wb') as f:
        # 写入头部信息
        f.write(header.encode('ascii'))
        
        # 写入深度数据，按照 row-major 顺序展平并转换为 4-byte 浮点数
        f.write(depth_array_resized.tobytes())

print("所有文件已成功生成")
