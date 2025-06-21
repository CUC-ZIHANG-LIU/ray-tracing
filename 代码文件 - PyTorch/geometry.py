import numpy as np
import torch
from vector import Vector3
from material import Material

# 几何体基类
class Geometry:
    def __init__(self, material):
        if not isinstance(material, Material):
            raise TypeError("material必须是Material类型")
        self.material = material
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 球体
class Sphere(Geometry):
    def __init__(self, center, radius, material):
        super().__init__(material)
        if not isinstance(center, Vector3):
            raise TypeError("center必须是Vector3类型")
        if not isinstance(radius, (int, float)) or radius <= 0:
            raise TypeError("radius必须是正数")
        self.center = center
        self.radius = radius
    
    def intersect(self, ray):
        if not isinstance(ray, dict) or 'origin' not in ray or 'direction' not in ray:
            raise TypeError("ray必须是包含'origin'和'direction'的字典")
        if not isinstance(ray['origin'], Vector3) or not isinstance(ray['direction'], Vector3):
            raise TypeError("ray的origin和direction必须是Vector3类型")
            
        # 将射线转换为张量
        origin = torch.tensor([ray['origin'].x, ray['origin'].y, ray['origin'].z], 
                            dtype=torch.float32, device=self.device)
        direction = torch.tensor([ray['direction'].x, ray['direction'].y, ray['direction'].z], 
                               dtype=torch.float32, device=self.device)
        center = torch.tensor([self.center.x, self.center.y, self.center.z], 
                            dtype=torch.float32, device=self.device)
        
        # 计算射线到球心的向量
        oc = origin - center
        
        # 计算二次方程的系数
        a = torch.dot(direction, direction)
        b = 2.0 * torch.dot(oc, direction)
        c = torch.dot(oc, oc) - self.radius * self.radius
        
        # 计算判别式
        discriminant = b * b - 4 * a * c
        
        if discriminant < 0:
            return None
            
        # 计算交点
        t = (-b - torch.sqrt(discriminant)) / (2.0 * a)
        if t < 0:
            t = (-b + torch.sqrt(discriminant)) / (2.0 * a)
            if t < 0:
                return None
                
        # 计算交点和法线
        point = origin + t * direction
        normal = (point - center) / self.radius
        
        return {
            'distance': t.item(),
            'point': Vector3(point[0].item(), point[1].item(), point[2].item()),
            'normal': Vector3(normal[0].item(), normal[1].item(), normal[2].item())
        }

    def intersect_batch(self, rays_batch):
        batch_size = rays_batch['origins'].shape[0]
        origins = rays_batch['origins']
        directions = rays_batch['directions']
        center = torch.tensor([self.center.x, self.center.y, self.center.z], dtype=torch.float32, device=self.device)
        oc = origins - center
        a = torch.sum(directions * directions, dim=1)
        b = 2.0 * torch.sum(oc * directions, dim=1)
        c = torch.sum(oc * oc, dim=1) - self.radius * self.radius
        discriminant = b * b - 4 * a * c
        t = torch.full((batch_size,), float('inf'), device=self.device)
        valid_mask = discriminant >= 0
        if torch.any(valid_mask):
            sqrt_disc = torch.sqrt(discriminant[valid_mask])
            t1 = (-b[valid_mask] - sqrt_disc) / (2.0 * a[valid_mask])
            t2 = (-b[valid_mask] + sqrt_disc) / (2.0 * a[valid_mask])
            t_valid = torch.where(t1 >= 0, t1, t2)
            t_valid = torch.where(t_valid >= 0, t_valid, torch.full_like(t_valid, float('inf')))
            t[valid_mask] = t_valid
        points = origins + t.unsqueeze(-1) * directions
        normals = (points - center) / self.radius
        colors = torch.zeros((batch_size, 3), device=self.device)
        diffuse_color = self.material.diffuse
        hit_mask = t < float('inf')
        colors[hit_mask] = diffuse_color.expand(hit_mask.sum(), -1)
        return {
            'distances': t,
            'points': points,
            'normals': normals,
            'colors': colors
        }

# 立方体
class Cube(Geometry):
    def __init__(self, center, size, material, rotation=None):
        super().__init__(material)
        if not isinstance(center, Vector3):
            raise TypeError("center必须是Vector3类型")
        if not isinstance(size, (int, float)) or size <= 0:
            raise TypeError("size必须是正数")
        self.center = center
        self.size = size
        self.rotation = rotation if rotation is not None else torch.eye(3, device=self.device)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def intersect(self, ray):
        if not isinstance(ray, dict) or 'origin' not in ray or 'direction' not in ray:
            raise TypeError("ray必须是包含'origin'和'direction'的字典")
        if not isinstance(ray['origin'], Vector3) or not isinstance(ray['direction'], Vector3):
            raise TypeError("ray的origin和direction必须是Vector3类型")
            
        # 将射线转换为张量
        origin = torch.tensor([ray['origin'].x, ray['origin'].y, ray['origin'].z], 
                            dtype=torch.float32, device=self.device)
        direction = torch.tensor([ray['direction'].x, ray['direction'].y, ray['direction'].z], 
                               dtype=torch.float32, device=self.device)
        center = torch.tensor([self.center.x, self.center.y, self.center.z], 
                            dtype=torch.float32, device=self.device)
        
        # 将射线转换到立方体的局部坐标系
        local_origin = torch.matmul(self.rotation.T, origin - center)
        local_direction = torch.matmul(self.rotation.T, direction)
        
        # 计算立方体的边界
        half_size = self.size / 2
        bounds = torch.tensor([[-half_size, half_size],
                             [-half_size, half_size],
                             [-half_size, half_size]], device=self.device)
        
        # 计算射线与边界的交点
        t_min = torch.tensor(float('-inf'), device=self.device)
        t_max = torch.tensor(float('inf'), device=self.device)
        normal = torch.zeros(3, device=self.device)
        
        for i in range(3):
            if abs(local_direction[i]) < 1e-6:
                if local_origin[i] < bounds[i, 0] or local_origin[i] > bounds[i, 1]:
                    return None
            else:
                t1 = (bounds[i, 0] - local_origin[i]) / local_direction[i]
                t2 = (bounds[i, 1] - local_origin[i]) / local_direction[i]
                if t1 > t2:
                    t1, t2 = t2, t1
                if t1 > t_min:
                    t_min = t1
                    normal = torch.zeros(3, device=self.device)
                    normal[i] = -1 if local_direction[i] > 0 else 1
                if t2 < t_max:
                    t_max = t2
                if t_min > t_max:
                    return None
                if t_max < 0:
                    return None
        
        if t_min < 0:
            t_min = t_max
            if t_min < 0:
                return None
        
        # 计算交点和法线
        intersection_point = local_origin + t_min * local_direction
        world_normal = torch.matmul(self.rotation, normal)
        
        return {
            'distance': t_min.item(),
            'point': Vector3(intersection_point[0].item() + self.center.x,
                           intersection_point[1].item() + self.center.y,
                           intersection_point[2].item() + self.center.z),
            'normal': Vector3(world_normal[0].item(),
                            world_normal[1].item(),
                            world_normal[2].item())
        }

    def intersect_batch(self, rays_batch):
        """批量计算射线与立方体的交点
        Args:
            rays_batch: 包含批量射线起点和方向的字典
        Returns:
            dict: 包含交点和颜色的信息
        """
        origins = rays_batch['origins']
        directions = rays_batch['directions']
        batch_size = origins.shape[0]
        center = torch.tensor([self.center.x, self.center.y, self.center.z], dtype=torch.float32, device=self.device)
        local_origins = torch.matmul(self.rotation.T, (origins - center).unsqueeze(-1)).squeeze(-1)
        local_directions = torch.matmul(self.rotation.T, directions.unsqueeze(-1)).squeeze(-1)
        half_size = self.size / 2
        bounds = torch.tensor([[-half_size, half_size], [-half_size, half_size], [-half_size, half_size]], device=self.device)
        t_min = torch.full((batch_size,), float('-inf'), device=self.device)
        t_max = torch.full((batch_size,), float('inf'), device=self.device)
        normals = torch.zeros((batch_size, 3), device=self.device)
        for i in range(3):
            parallel_mask = torch.abs(local_directions[:, i]) < 1e-6
            if torch.any(parallel_mask):
                invalid_mask = (local_origins[parallel_mask, i] < bounds[i, 0]) | (local_origins[parallel_mask, i] > bounds[i, 1])
                if torch.any(invalid_mask):
                    t_min[parallel_mask] = float('inf')
                    t_max[parallel_mask] = float('-inf')
            t1 = (bounds[i, 0] - local_origins[:, i]) / local_directions[:, i]
            t2 = (bounds[i, 1] - local_origins[:, i]) / local_directions[:, i]
            mask = t1 > t2
            t1[mask], t2[mask] = t2[mask], t1[mask]
            t_min = torch.maximum(t_min, t1)
            t_max = torch.minimum(t_max, t2)
            normal_mask = t1 > t_min
            normals[normal_mask, i] = -1 if local_directions[0, i] > 0 else 1
        valid_mask = (t_min <= t_max) & (t_max >= 0)
        neg_mask = valid_mask & (t_min < 0)
        t_min[neg_mask] = t_max[neg_mask]
        points = local_origins + t_min.unsqueeze(-1) * local_directions
        world_normals = torch.matmul(self.rotation, normals.unsqueeze(-1)).squeeze(-1)
        colors = torch.zeros((batch_size, 3), device=self.device)
        diffuse_color = self.material.diffuse
        hit_mask = (t_min < float('inf')) & (t_min > 0)
        colors[hit_mask] = diffuse_color.expand(hit_mask.sum(), -1)
        points[~hit_mask] = 0
        world_normals[~hit_mask] = 0
        return {
            'distances': t_min,
            'points': points,
            'normals': world_normals,
            'colors': colors
        }

# 圆锥体
class Cone(Geometry):
    def __init__(self, position, radius, height, material):
        super().__init__(material)
        self.position = position
        self.radius = radius
        self.height = height
    
    def intersect(self, ray):
        # 射线与圆锥体的求交算法 (简化版)
        # 假设圆锥体的底面中心在 position 的 z 轴，高度沿 z 轴向上。
        # 圆锥体方程涉及到 x^2 + y^2 = (k*z)^2 的形式，其中 k = radius / height。
        
        # 将射线原点和方向平移，使圆锥体底面中心在原点。
        # local_origin = ray.origin - self.position
        # local_direction = ray.direction
        
        # 构建二次方程 Ax^2 + Bx + C = 0 的系数
        k = (self.radius / self.height) ** 2
        a = ray['direction'].x * ray['direction'].x + ray['direction'].y * ray['direction'].y - k * ray['direction'].z * ray['direction'].z
        b = 2 * (ray['origin'].x * ray['direction'].x + ray['origin'].y * ray['direction'].y - k * ray['origin'].z * ray['direction'].z)
        c = ray['origin'].x * ray['origin'].x + ray['origin'].y * ray['origin'].y - k * ray['origin'].z * ray['origin'].z
        
        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            # 无实数解，不相交
            return None
        
        t1 = (-b - np.sqrt(discriminant)) / (2 * a)
        t2 = (-b + np.sqrt(discriminant)) / (2 * a)
        
        # 筛选有效交点
        if t1 < 0 and t2 < 0:
            return None # 两个交点都在射线后面
        
        t = t1 if t1 > 0 else t2 # 取第一个在射线前方的交点
        point = ray['origin'] + ray['direction'] * t
        
        # 检查交点是否在圆锥体的高度范围内
        if point.z < self.position.z or point.z > self.position.z + self.height:
            return None
        
        # 计算交点处的法线
        # 圆锥体侧面的法线计算比较复杂，这里是简化版本，可能不够精确。
        # 通常需要考虑顶点和底面的法线。
        normal = Vector3(
            point.x,
            point.y,
            -k * (point.z - self.position.z) # Z分量是根据圆锥体的斜率和高度计算的
        ).normalize()
        
        return {
            'distance': t,
            'point': point,
            'normal': normal
        }

    def intersect_batch(self, rays_batch):
        origins = rays_batch['origins']
        directions = rays_batch['directions']
        batch_size = origins.shape[0]
        position = torch.tensor([self.position.x, self.position.y, self.position.z], dtype=torch.float32, device=self.device)
        k = (self.radius / self.height) ** 2
        a = directions[:, 0] * directions[:, 0] + directions[:, 1] * directions[:, 1] - k * directions[:, 2] * directions[:, 2]
        b = 2 * (origins[:, 0] * directions[:, 0] + origins[:, 1] * directions[:, 1] - k * origins[:, 2] * directions[:, 2])
        c = origins[:, 0] * origins[:, 0] + origins[:, 1] * origins[:, 1] - k * origins[:, 2] * origins[:, 2]
        discriminant = b * b - 4 * a * c
        t = torch.full((batch_size,), float('inf'), device=self.device)
        valid_mask = discriminant >= 0
        if torch.any(valid_mask):
            sqrt_disc = torch.sqrt(discriminant[valid_mask])
            t1 = (-b[valid_mask] - sqrt_disc) / (2 * a[valid_mask])
            t2 = (-b[valid_mask] + sqrt_disc) / (2 * a[valid_mask])
            t_valid = torch.where(t1 >= 0, t1, t2)
            t_valid = torch.where(t_valid >= 0, t_valid, torch.full_like(t_valid, float('inf')))
            t[valid_mask] = t_valid
        points = origins + t.unsqueeze(-1) * directions
        height_mask = (points[:, 2] >= position[2]) & (points[:, 2] <= position[2] + self.height)
        hit_mask = (t < float('inf')) & height_mask
        normals = torch.zeros_like(points)
        normals[hit_mask, 0] = points[hit_mask, 0]
        normals[hit_mask, 1] = points[hit_mask, 1]
        normals[hit_mask, 2] = -k * (points[hit_mask, 2] - position[2])
        normals[hit_mask] = normals[hit_mask] / torch.norm(normals[hit_mask], dim=1, keepdim=True)
        colors = torch.zeros((batch_size, 3), device=self.device)
        diffuse_color = self.material.diffuse
        colors[hit_mask] = diffuse_color.expand(hit_mask.sum(), -1)
        points[~hit_mask] = 0
        normals[~hit_mask] = 0
        return {
            'distances': t,
            'points': points,
            'normals': normals,
            'colors': colors
        }

# 平面（棋盘格地板）
class Plane(Geometry):
    def __init__(self, point, normal, material):
        super().__init__(material)
        if not isinstance(point, Vector3) or not isinstance(normal, Vector3):
            raise TypeError("point和normal必须是Vector3类型")
        self.point = point
        self.normal = normal.normalize()

    def intersect(self, ray):
        if not isinstance(ray, dict) or 'origin' not in ray or 'direction' not in ray:
            raise TypeError("ray必须是包含'origin'和'direction'的字典")
        if not isinstance(ray['origin'], Vector3) or not isinstance(ray['direction'], Vector3):
            raise TypeError("ray的origin和direction必须是Vector3类型")
            
        # 将射线转换为张量
        origin = torch.tensor([ray['origin'].x, ray['origin'].y, ray['origin'].z], 
                            dtype=torch.float32, device=self.device)
        direction = torch.tensor([ray['direction'].x, ray['direction'].y, ray['direction'].z], 
                               dtype=torch.float32, device=self.device)
        normal = torch.tensor([self.normal.x, self.normal.y, self.normal.z], 
                            dtype=torch.float32, device=self.device)
        point = torch.tensor([self.point.x, self.point.y, self.point.z], 
                           dtype=torch.float32, device=self.device)
        
        # 计算分母
        denom = torch.dot(normal, direction)
        
        # 如果射线与平面平行，则没有交点
        if abs(denom) < 1e-6:
            return None
            
        # 计算交点参数
        t = torch.dot(point - origin, normal) / denom
        
        # 如果交点在射线后方，则没有交点
        if t < 0:
            return None
            
        # 计算交点和法线
        intersection_point = origin + t * direction
        
        return {
            'distance': t.item(),
            'point': Vector3(intersection_point[0].item(), 
                           intersection_point[1].item(), 
                           intersection_point[2].item()),
            'normal': self.normal
        }
    
    # 棋盘格着色辅助函数
    def get_checker_color(self, point, color1, color2, scale=1.0):
        # 根据交点在平面上的X和Z坐标，计算棋盘格颜色
        # 通过对坐标进行缩放和取整，判断其所在的"格子"是奇数还是偶数，从而决定颜色。
        x, z = point.x, point.z
        if int((x * scale) % 2) == int((z * scale) % 2):
            return color1 # 如果X和Z坐标的"奇偶性"相同，则使用颜色1
        else:
            return color2 # 否则使用颜色2

    def intersect_batch(self, rays_batch):
        batch_size = rays_batch['origins'].shape[0]
        origins = rays_batch['origins']
        directions = rays_batch['directions']
        normal = torch.tensor([self.normal.x, self.normal.y, self.normal.z], dtype=torch.float32, device=self.device)
        point = torch.tensor([self.point.x, self.point.y, self.point.z], dtype=torch.float32, device=self.device)
        denom = torch.sum(directions * normal, dim=1)
        t = torch.full((batch_size,), float('inf'), device=self.device)
        valid_mask = torch.abs(denom) >= 1e-6
        t[valid_mask] = torch.sum((point - origins[valid_mask]) * normal, dim=1) / denom[valid_mask]
        valid_mask &= t >= 0
        points = origins + t.unsqueeze(-1) * directions
        normals = normal.expand(batch_size, -1)
        colors = torch.zeros((batch_size, 3), device=self.device)
        for i in range(batch_size):
            if valid_mask[i]:
                point_vec = Vector3(points[i, 0].item(), points[i, 1].item(), points[i, 2].item())
                color1 = torch.tensor([0.9, 0.9, 0.9], dtype=torch.float32, device=self.device)
                color2 = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float32, device=self.device)
                checker_color = self.get_checker_color(point_vec, color1, color2, scale=1.5)
                colors[i] = checker_color
            else:
                points[i] = 0
                normals[i] = 0
        return {
            'distances': t,
            'points': points,
            'normals': normals,
            'colors': colors
        }

# 正四面体
class Tetrahedron(Geometry):
    def __init__(self, v0, v1, v2, v3, material):
        super().__init__(material)
        if not all(isinstance(v, Vector3) for v in [v0, v1, v2, v3]):
            raise TypeError("所有顶点必须是Vector3类型")
        self.vertices = [v0, v1, v2, v3]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def intersect(self, ray):
        if not isinstance(ray, dict) or 'origin' not in ray or 'direction' not in ray:
            raise TypeError("ray必须是包含'origin'和'direction'的字典")
        if not isinstance(ray['origin'], Vector3) or not isinstance(ray['direction'], Vector3):
            raise TypeError("ray的origin和direction必须是Vector3类型")
            
        # 将射线转换为张量
        origin = torch.tensor([ray['origin'].x, ray['origin'].y, ray['origin'].z], 
                            dtype=torch.float32, device=self.device)
        direction = torch.tensor([ray['direction'].x, ray['direction'].y, ray['direction'].z], 
                               dtype=torch.float32, device=self.device)
        
        # 将顶点转换为张量
        vertices = torch.tensor([[v.x, v.y, v.z] for v in self.vertices], 
                              dtype=torch.float32, device=self.device)
        
        # 计算四面体的面
        faces = [
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 1],
            [1, 3, 2]
        ]
        
        closest_intersection = None
        min_distance = float('inf')
        
        for face in faces:
            # 获取面的顶点
            v0 = vertices[face[0]]
            v1 = vertices[face[1]]
            v2 = vertices[face[2]]
            
            # 计算面的法线
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = torch.cross(edge1, edge2)
            normal = normal / torch.norm(normal)
            
            # 计算射线与面的交点
            denom = torch.dot(normal, direction)
            if abs(denom) < 1e-6:
                continue
                
            t = torch.dot(v0 - origin, normal) / denom
            if t < 0:
                continue
                
            # 计算交点
            intersection = origin + t * direction
            
            # 检查交点是否在三角形内
            edge0 = v1 - v0
            edge1 = v2 - v1
            edge2 = v0 - v2
            
            c0 = intersection - v0
            c1 = intersection - v1
            c2 = intersection - v2
            
            if (torch.dot(normal, torch.cross(edge0, c0)) > 0 and
                torch.dot(normal, torch.cross(edge1, c1)) > 0 and
                torch.dot(normal, torch.cross(edge2, c2)) > 0):
                
                if t < min_distance:
                    min_distance = t
                    closest_intersection = {
                        'distance': t.item(),
                        'point': Vector3(intersection[0].item(),
                                       intersection[1].item(),
                                       intersection[2].item()),
                        'normal': Vector3(normal[0].item(),
                                        normal[1].item(),
                                        normal[2].item())
                    }
        
        return closest_intersection

    def intersect_batch(self, rays_batch):
        """批量计算射线与四面体的交点
        Args:
            rays_batch: 包含批量射线起点和方向的字典
        Returns:
            dict: 包含交点和颜色的信息
        """
        # 将射线转换为张量
        origins = rays_batch['origins']  # shape: (batch_size, 3)
        directions = rays_batch['directions']  # shape: (batch_size, 3)
        
        # 将顶点转换为张量
        vertices = torch.tensor([[v.x, v.y, v.z] for v in self.vertices], 
                              dtype=torch.float32, device=self.device)
        
        # 定义四面体的面
        faces = [
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 1],
            [1, 3, 2]
        ]
        
        # 初始化结果
        closest_distances = torch.full((len(origins),), float('inf'), device=self.device)
        closest_points = torch.zeros((len(origins), 3), device=self.device)
        closest_normals = torch.zeros((len(origins), 3), device=self.device)
        
        # 对每个面计算交点
        for face in faces:
            # 获取面的顶点
            v0 = vertices[face[0]]
            v1 = vertices[face[1]]
            v2 = vertices[face[2]]
            
            # 计算面的法线
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = torch.cross(edge1, edge2)
            normal = normal / torch.norm(normal)
            
            # 计算射线与面的交点
            denom = torch.sum(directions * normal, dim=1)
            valid_mask = torch.abs(denom) >= 1e-6
            
            if not torch.any(valid_mask):
                continue
            
            t = torch.sum((v0 - origins) * normal, dim=1) / denom
            valid_mask &= t >= 0
            
            if not torch.any(valid_mask):
                continue
            
            # 计算交点
            points = origins + t.unsqueeze(-1) * directions
            
            # 检查交点是否在三角形内
            edge0 = v1 - v0
            edge1 = v2 - v1
            edge2 = v0 - v2
            
            # 修改这部分代码，正确处理向量叉积
            c0 = points - v0.unsqueeze(0)  # shape: (batch_size, 3)
            c1 = points - v1.unsqueeze(0)  # shape: (batch_size, 3)
            c2 = points - v2.unsqueeze(0)  # shape: (batch_size, 3)
            
            # 计算叉积
            cross0 = torch.cross(edge0.unsqueeze(0).expand(len(points), -1), c0)
            cross1 = torch.cross(edge1.unsqueeze(0).expand(len(points), -1), c1)
            cross2 = torch.cross(edge2.unsqueeze(0).expand(len(points), -1), c2)
            
            # 计算点积
            dot0 = torch.sum(normal.unsqueeze(0) * cross0, dim=1)
            dot1 = torch.sum(normal.unsqueeze(0) * cross1, dim=1)
            dot2 = torch.sum(normal.unsqueeze(0) * cross2, dim=1)
            
            # 判断点是否在三角形内
            in_triangle = (dot0 > 0) & (dot1 > 0) & (dot2 > 0)
            
            valid_mask &= in_triangle
            
            if not torch.any(valid_mask):
                continue
            
            # 更新最近交点
            update_mask = valid_mask & (t < closest_distances)
            closest_distances[update_mask] = t[update_mask]
            closest_points[update_mask] = points[update_mask]
            closest_normals[update_mask] = normal
        
        # 检查是否有有效交点
        valid_mask = closest_distances < float('inf')
        if not torch.any(valid_mask):
            return None
        
        # 计算颜色
        colors = torch.zeros((len(valid_mask), 3), device=self.device)
        diffuse_color = self.material.diffuse
        colors[valid_mask] = diffuse_color.expand(torch.sum(valid_mask), -1)
        
        return {
            'distances': closest_distances,
            'points': closest_points,
            'normals': closest_normals,
            'colors': colors
        }