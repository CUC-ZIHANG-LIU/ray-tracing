import torch

class Material:
    def __init__(self, ambient, diffuse, specular, shininess):
        if not isinstance(ambient, torch.Tensor) or ambient.shape != (3,):
            raise TypeError("ambient必须是3维PyTorch张量")
        if not isinstance(diffuse, torch.Tensor) or diffuse.shape != (3,):
            raise TypeError("diffuse必须是3维PyTorch张量")
        if not isinstance(specular, torch.Tensor) or specular.shape != (3,):
            raise TypeError("specular必须是3维PyTorch张量")
        if not isinstance(shininess, (int, float)) or shininess < 0:
            raise ValueError("shininess必须是非负数")
            
        self.ambient = ambient.to(torch.float32)  # 环境光反射系数 (GPU)
        self.diffuse = diffuse.to(torch.float32)  # 漫反射系数 (GPU)
        self.specular = specular.to(torch.float32)  # 镜面反射系数 (GPU)
        self.shininess = shininess  # 镜面反射指数