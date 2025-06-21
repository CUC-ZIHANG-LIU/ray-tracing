from vector import Vector3
import cupy as cp

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
        self.forward = (look_at - position).normalize().to_gpu()
        self.right = self.forward.cross(up).normalize().to_gpu()
        self.up = self.right.cross(self.forward).normalize().to_gpu()
    
    def get_ray(self, x, y, width, height):
        # 计算像素在相机平面上的位置
        aspect_ratio = width / height
        fov_rad = self.fov * cp.pi / 180.0
        scale = cp.tan(fov_rad / 2.0)
        
        # 将像素坐标转换为[-1, 1]范围
        px = (2.0 * ((x + 0.5) / width) - 1.0) * aspect_ratio * scale
        py = (1.0 - 2.0 * ((y + 0.5) / height)) * scale
        
        # 计算光线方向
        direction = (self.forward + 
                    self.right * px + 
                    self.up * py).normalize()
        
        return Ray(self.position, direction)

    def get_rays_batch(self, X, Y, width, height):
        # 使用CuPy批量计算光线方向
        aspect_ratio = width / height
        fov_rad = self.fov * cp.pi / 180.0
        scale = cp.tan(fov_rad / 2.0)
        
        # 将像素坐标转换为[-1, 1]范围
        px = (2.0 * ((X + 0.5) / width) - 1.0) * aspect_ratio * scale
        py = (1.0 - 2.0 * ((Y + 0.5) / height)) * scale
        
        # 将相机参数转换为CuPy数组
        forward = cp.array([self.forward.x, self.forward.y, self.forward.z], dtype=cp.float32)
        right = cp.array([self.right.x, self.right.y, self.right.z], dtype=cp.float32)
        up = cp.array([self.up.x, self.up.y, self.up.z], dtype=cp.float32)
        position = cp.array([self.position.x, self.position.y, self.position.z], dtype=cp.float32)
        
        # 扩展维度以进行广播
        px = px[..., cp.newaxis]
        py = py[..., cp.newaxis]
        
        # 计算光线方向
        directions = forward + right * px + up * py
        
        # 归一化方向向量
        directions = directions / cp.linalg.norm(directions, axis=-1, keepdims=True)
        
        # 创建光线批次
        origins = cp.broadcast_to(position, directions.shape)
        
        return {'origins': origins, 'directions': directions}