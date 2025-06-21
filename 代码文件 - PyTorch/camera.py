from vector import Vector3
import torch

class Ray:
    def __init__(self, origin, direction):
        if not isinstance(origin, Vector3) or not isinstance(direction, Vector3):
            raise TypeError("Ray的origin和direction必须是Vector3类型")
        self.origin = origin
        self.direction = direction.normalize()

class Camera:
    def __init__(self, position, look_at, up, fov):
        if not all(isinstance(v, Vector3) for v in [position, look_at, up]):
            raise TypeError("position、look_at和up必须是Vector3类型")
        if not isinstance(fov, (int, float)) or fov <= 0 or fov >= 180:
            raise ValueError("fov必须是0到180之间的数值")
            
        self.position = position
        self.look_at = look_at
        self.up = up
        self.fov = fov
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 计算相机坐标系
        self.forward = (look_at - position).normalize()
        self.right = self.forward.cross(up).normalize()
        self.up = self.right.cross(self.forward).normalize()
        
        # 将向量转换为张量
        self.forward_tensor = torch.tensor([self.forward.x, self.forward.y, self.forward.z], 
                                         dtype=torch.float32, device=self.device)
        self.right_tensor = torch.tensor([self.right.x, self.right.y, self.right.z], 
                                       dtype=torch.float32, device=self.device)
        self.up_tensor = torch.tensor([self.up.x, self.up.y, self.up.z], 
                                    dtype=torch.float32, device=self.device)
        self.position_tensor = torch.tensor([self.position.x, self.position.y, self.position.z], 
                                          dtype=torch.float32, device=self.device)

    def normalize_tensor(self, tensor):
        """使用 PyTorch 函数对张量进行归一化"""
        norm = torch.norm(tensor, dim=-1, keepdim=True)
        return tensor / (norm + 1e-8)  # 添加小量防止除零

    def get_ray(self, x, y, width, height):
        if not all(isinstance(v, int) and v > 0 for v in [width, height]):
            raise ValueError("width和height必须是正整数")
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise TypeError("x和y必须是数值类型")
            
        # 将输入转换为张量
        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        y = torch.tensor(y, dtype=torch.float32, device=self.device)
        width = torch.tensor(width, dtype=torch.float32, device=self.device)
        height = torch.tensor(height, dtype=torch.float32, device=self.device)
        
        # 计算像素坐标到NDC坐标的映射
        aspect_ratio = width / height
        fov_rad = torch.tensor(self.fov * torch.pi / 180.0, device=self.device)
        scale = torch.tan(fov_rad / 2.0)
        
        # 计算像素在相机空间中的方向
        px = (2.0 * ((x + 0.5) / width) - 1.0) * aspect_ratio * scale
        py = (1.0 - 2.0 * ((y + 0.5) / height)) * scale
        
        # 计算射线方向
        direction = self.forward_tensor + px * self.right_tensor + py * self.up_tensor
        direction = self.normalize_tensor(direction)
        
        # 将张量转换为Vector3
        origin = Vector3(self.position.x, self.position.y, self.position.z)
        direction = Vector3(direction[0].item(), direction[1].item(), direction[2].item())
        
        # 返回射线
        return {
            'origin': origin,
            'direction': direction
        }

    def get_rays_batch(self, X, Y, width, height):
        device = self.device
        X = X.to(device)
        Y = Y.to(device)
        aspect_ratio = width / height
        fov_rad = self.fov * torch.pi / 180.0
        scale = torch.tan(torch.tensor(fov_rad / 2.0, device=device))  # 保证scale为tensor，和X/px同类型
        px = (2.0 * ((X + 0.5) / width) - 1.0) * aspect_ratio * scale
        py = (1.0 - 2.0 * ((Y + 0.5) / height)) * scale
        forward = torch.tensor([self.forward.x, self.forward.y, self.forward.z], device=device)
        right = torch.tensor([self.right.x, self.right.y, self.right.z], device=device)
        up = torch.tensor([self.up.x, self.up.y, self.up.z], device=device)
        position = torch.tensor([self.position.x, self.position.y, self.position.z], device=device)
        directions = forward + right * px.unsqueeze(-1) + up * py.unsqueeze(-1)
        directions = directions / torch.norm(directions, dim=-1, keepdim=True)
        origins = position.expand_as(directions)
        return {'origins': origins, 'directions': directions}