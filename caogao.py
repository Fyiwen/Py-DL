def find_nearest_vertices(self, points, k=3):
    """ 找到距离多个点最近的k个顶点 """
    # 计算每个点到所有顶点的欧氏距离
    # points 是一个 (n, 3) 的张量，self.vertices 是一个 (m, 3) 的张量，n是点的数量，m是顶点的数量
    distances = torch.norm(self.vertices[None, :, :] - points[:, None, :], dim=2)  # 计算每个点到所有顶点的距离

    # 获取每个点最近的k个顶点的索引和距离
    _, indices = torch.topk(distances, k, largest=False, dim=1)
    nearest_distances = torch.gather(distances, 1, indices)

    return indices, nearest_distances



def get_uv_for_3d_points(self, points):
    """根据3D点的坐标计算其对应的UV坐标，并生成UV图"""
    image_size = (512, 512)
    uv_image = torch.zeros((image_size[0], image_size[1], 3), dtype=torch.float32)  # 创建空白的UV图
    uv_indices = []  # 记录每个3D点对应的UV坐标

    uvs = torch.tensor(self.uvs, device=self.vertices.device)
    points = torch.tensor(points, device=self.vertices.device)

    # 批量获取最近的3个顶点
    indices, distances = self.find_nearest_vertices(points)  # 假设该方法已经批量化

    # 获取顶点坐标，避免在循环内逐个查找
    v1 = self.vertices[indices[:, 0]]
    v2 = self.vertices[indices[:, 1]]
    v3 = self.vertices[indices[:, 2]]

    # 批量判断三点是否共线，如果是，跳过该点的UV计算（可以通过向量叉积判断）
    non_collinear_mask = ~self.are_points_collinear(v1, v2, v3)  # 批量判断哪些三点不共线

    # 处理不共线的点
    valid_indices = non_collinear_mask.nonzero(as_tuple=True)[0]  # 获取有效的点索引

    

    # 对于有效的点，批量计算UV
    if valid_indices.numel() > 0:
        valid_points = points[valid_indices]
        valid_indices_batch = indices[valid_indices]

        # 批量计算每个点的重心坐标
        barycentric_coords = self.compute_barycentric_coords_batch(valid_points, v1[valid_indices], v2[valid_indices], v3[valid_indices])

        # 计算每个点的UV坐标
        uv_batch = torch.stack([
            barycentric_coords[:, 0].unsqueeze(1) * uvs[self.v_uv[valid_indices_batch[:, 0]]],
            barycentric_coords[:, 1].unsqueeze(1) * uvs[self.v_uv[valid_indices_batch[:, 1]]],
            barycentric_coords[:, 2].unsqueeze(1) * uvs[self.v_uv[valid_indices_batch[:, 2]]]
        ], dim=2)

        uv_indices.append(uv_batch.sum(dim=2))  # 计算最终的UV坐标

    # 进一步处理不构成面片的情况（即无效的面片）
    invalid_faces_mask = ~self.find_face_with_vertices(valid_indices_batch[:, 0], valid_indices_batch[:, 1], valid_indices_batch[:, 2])
    invalid_face_indices = valid_indices[invalid_faces_mask]  # 获取不构成面片
    if invalid_face_indices.numel() > 0:
        invalid_face_pints = points[invalid_face_indices]
        
        # 对于共线的点，可以使用其他方法计算UV（此处可以根据实际需要处理，暂时用None表示）
        uv_indices.append(None)  # 或者使用某个默认的UV值



    # 处理共线的点（共线的点无法用重心坐标来计算UV）
    collinear_indices = (~non_collinear_mask).nonzero(as_tuple=True)[0]
    if collinear_indices.numel() > 0:
        collinear_points = points[collinear_indices]
        # 对于共线的点，可以使用其他方法计算UV（此处可以根据实际需要处理，暂时用None表示）
        uv_indices.append(None)  # 或者使用某个默认的UV值

    # 返回最终计算的UV索引或UV图
    return uv_indices
