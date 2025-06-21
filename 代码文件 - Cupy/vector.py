import cupy as cp

class Vector3:
    def __init__(self, x, y, z, device='gpu'):
        self.device = device
        if device == 'gpu':
            self.data = cp.array([x, y, z], dtype=cp.float32)
        else:
            self.data = cp.array([x, y, z], dtype=cp.float32).get()
        self.x = self.data[0]
        self.y = self.data[1]
        self.z = self.data[2]
        
    def __add__(self, other):
        if self.device == 'gpu':
            return Vector3(*(self.data + other.data), device='gpu')
        return Vector3(*(self.data + other.data), device='cpu')
    
    def __sub__(self, other):
        if self.device == 'gpu':
            return Vector3(*(self.data - other.data), device='gpu')
        return Vector3(*(self.data - other.data), device='cpu')
    
    def __mul__(self, scalar):
        if self.device == 'gpu':
            return Vector3(*(self.data * scalar), device='gpu')
        return Vector3(*(self.data * scalar), device='cpu')
    
    def dot(self, other):
        if self.device == 'gpu':
            return cp.dot(self.data, other.data)
        return cp.dot(self.data, other.data).get()
    
    def cross(self, other):
        cross = cp.cross(self.data, other.data)
        if self.device == 'gpu':
            return Vector3(*cross, device='gpu')
        return Vector3(*cross.get(), device='cpu')
    
    def length(self):
        if self.device == 'gpu':
            return cp.sqrt(cp.sum(self.data ** 2))
        return cp.sqrt(cp.sum(self.data ** 2)).get()
    
    def normalize(self):
        length = self.length()
        if length == 0:
            return Vector3(0, 0, 0, device=self.device)
        if self.device == 'gpu':
            return Vector3(*(self.data / length), device='gpu')
        return Vector3(*(self.data / length), device='cpu')
    
    def to_array(self):
        if self.device == 'gpu':
            return self.data
        return self.data.get()
    
    def to_cpu(self):
        if self.device == 'cpu':
            return self
        return Vector3(*self.data.get(), device='cpu')
    
    def to_gpu(self):
        if self.device == 'gpu':
            return self
        return Vector3(*self.data, device='gpu')