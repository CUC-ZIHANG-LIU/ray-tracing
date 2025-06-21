import cupy as cp

class Material:
    def __init__(self, ambient, diffuse, specular, shininess):
        self.ambient = ambient.astype(cp.float32)  # 环境光反射系数 (GPU)
        self.diffuse = diffuse.astype(cp.float32)  # 漫反射系数 (GPU)
        self.specular = specular.astype(cp.float32)  # 镜面反射系数 (GPU)
        self.shininess = shininess  # 镜面反射指数