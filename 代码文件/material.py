import numpy as np

class Material:
    def __init__(self, ambient, diffuse, specular, shininess):
        self.ambient = ambient.astype(np.float32)  # 环境光反射系数
        self.diffuse = diffuse.astype(np.float32)  # 漫反射系数
        self.specular = specular.astype(np.float32)  # 镜面反射系数
        self.shininess = shininess  # 镜面反射指数 