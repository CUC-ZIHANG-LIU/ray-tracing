import torch

class Vector3:
    def __init__(self, x, y, z, device='cuda'):
        self.device = device
        if device == 'cuda':
            self.data = torch.tensor([x, y, z], dtype=torch.float32, device='cuda')
        else:
            self.data = torch.tensor([x, y, z], dtype=torch.float32)
        self.x = self.data[0]
        self.y = self.data[1]
        self.z = self.data[2]
        
    def __add__(self, other):
        if not isinstance(other, Vector3):
            raise TypeError("只能与Vector3类型相加")
        if self.device == 'cuda':
            return Vector3(*(self.data + other.data), device='cuda')
        return Vector3(*(self.data + other.data), device='cpu')
    
    def __sub__(self, other):
        if not isinstance(other, Vector3):
            raise TypeError("只能与Vector3类型相减")
        if self.device == 'cuda':
            return Vector3(*(self.data - other.data), device='cuda')
        return Vector3(*(self.data - other.data), device='cpu')
    
    def __mul__(self, scalar):
        if not isinstance(scalar, (int, float, torch.Tensor)):
            raise TypeError("只能与标量或张量相乘")
        # 如果是张量，确保它是标量
        if isinstance(scalar, torch.Tensor):
            if scalar.numel() != 1:
                raise ValueError("张量必须是标量")
            scalar = scalar.item()
        if self.device == 'cuda':
            return Vector3(*(self.data * scalar), device='cuda')
        return Vector3(*(self.data * scalar), device='cpu')
    
    def dot(self, other):
        if not isinstance(other, Vector3):
            raise TypeError("点积运算需要Vector3类型")
        if self.data.shape != other.data.shape:
            raise ValueError("向量维度不匹配")
        if self.device == 'cuda':
            return torch.dot(self.data, other.data)
        return torch.dot(self.data, other.data).item()
    
    def cross(self, other):
        if not isinstance(other, Vector3):
            raise TypeError("叉积运算需要Vector3类型")
        if self.data.shape != other.data.shape:
            raise ValueError("向量维度不匹配")
        cross = torch.cross(self.data, other.data)
        if self.device == 'cuda':
            return Vector3(*cross, device='cuda')
        return Vector3(*cross.cpu(), device='cpu')
    
    def length(self):
        if self.device == 'cuda':
            return torch.sqrt(torch.sum(self.data ** 2))
        return torch.sqrt(torch.sum(self.data ** 2)).item()
    
    def normalize(self):
        length = self.length()
        if length == 0:
            return Vector3(0, 0, 0, device=self.device)
        if self.device == 'cuda':
            return Vector3(*(self.data / length), device='cuda')
        return Vector3(*(self.data / length), device='cpu')
    
    def reflect(self, normal):
        """计算反射向量
        Args:
            normal: Vector3, 法线向量
        Returns:
            Vector3: 反射向量
        """
        if not isinstance(normal, Vector3):
            raise TypeError("法线必须是Vector3类型")
        # 确保法线是单位向量
        normal = normal.normalize()
        # 计算反射向量: R = I - 2(N·I)N
        dot_product = self.dot(normal)
        reflection = self - normal * (2 * dot_product)
        return reflection.normalize()
    
    def to_array(self):
        if self.device == 'cuda':
            return self.data
        return self.data.cpu()
    
    def to_cpu(self):
        if self.device == 'cpu':
            return self
        return Vector3(*self.data.cpu(), device='cpu')
    
    def to_gpu(self):
        if self.device == 'cuda':
            return self
        return Vector3(*self.data, device='cuda')