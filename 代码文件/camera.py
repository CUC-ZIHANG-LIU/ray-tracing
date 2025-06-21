from vector import Vector3
import numpy as np
import torch

class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction.normalize()

class Camera:
    def __init__(self, position, look_at, up, fov):
        self.position = position
        self.look_at = look_at
        self.up = up
        self.fov = fov
        
        # 计算相机坐标系
        self.forward = (look_at - position).normalize()
        self.right = self.forward.cross(up).normalize()
        self.up = self.right.cross(self.forward).normalize()
    
    def get_ray(self, x, y, width, height):
        # 计算像素在相机平面上的位置
        aspect_ratio = width / height
        fov_rad = self.fov * 3.14159 / 180.0
        scale = np.tan(fov_rad / 2.0)
        
        # 将像素坐标转换为[-1, 1]范围
        px = (2.0 * ((x + 0.5) / width) - 1.0) * aspect_ratio * scale
        py = (1.0 - 2.0 * ((y + 0.5) / height)) * scale
        
        # 计算光线方向
        direction = (self.forward + 
                    self.right * px + 
                    self.up * py).normalize()
        
        return Ray(self.position, direction)

    def get_rays_batch(self, X, Y, width, height):
        # 批量计算光线方向
        aspect_ratio = width / height
        fov_rad = self.fov * 3.14159 / 180.0
        scale = torch.tan(torch.tensor(fov_rad / 2.0))
        
        # 将像素坐标转换为[-1, 1]范围
        px = (2.0 * ((X + 0.5) / width) - 1.0) * aspect_ratio * scale
        py = (1.0 - 2.0 * ((Y + 0.5) / height)) * scale
        
        # 将相机参数转换为张量
        forward = torch.tensor([self.forward.x, self.forward.y, self.forward.z])
        right = torch.tensor([self.right.x, self.right.y, self.right.z])
        up = torch.tensor([self.up.x, self.up.y, self.up.z])
        position = torch.tensor([self.position.x, self.position.y, self.position.z])
        
        # 扩展维度以进行广播
        px = px.unsqueeze(-1)
        py = py.unsqueeze(-1)
        
        # 计算光线方向
        directions = (forward + 
                     right * px + 
                     up * py)
        
        # 归一化方向向量
        directions = directions / torch.norm(directions, dim=-1, keepdim=True)
        
        # 创建光线批次
        origins = position.expand(directions.shape)
        
        return {'origins': origins, 'directions': directions} 