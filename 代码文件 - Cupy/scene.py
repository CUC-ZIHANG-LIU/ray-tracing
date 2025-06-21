import cupy as cp
from vector import Vector3
from camera import Ray
from geometry import Plane

class Scene:
    def __init__(self):
        self.objects = []
        self.lights = []
        self.ambient_light = cp.array([0.08, 0.08, 0.08], dtype=cp.float32)
        self.max_depth = 3
    
    def add_object(self, obj):
        self.objects.append(obj)
    
    def add_light(self, position, color):
        self.lights.append({'position': position, 'color': color})
    
    def trace_ray(self, ray, depth):
        if depth > self.max_depth:
            return cp.zeros(3, dtype=cp.float32)
        
        # 找到最近的交点
        closest_intersection = None
        closest_object = None
        
        for obj in self.objects:
            intersection = obj.intersect(ray)
            if intersection and (closest_intersection is None or intersection['t'] < closest_intersection['t']):
                closest_intersection = intersection
                closest_object = obj
        
        if closest_intersection is None:
            return cp.zeros(3, dtype=cp.float32)
        
        # 计算光照
        point = closest_intersection['point']
        normal = closest_intersection['normal']
        material = closest_object.material
        
        # 棋盘格地板 
        if isinstance(closest_object, Plane):
            # 棋盘格两种颜色
            color1 = cp.array([0.9, 0.9, 0.9], dtype=cp.float32)
            color2 = cp.array([0.1, 0.1, 0.1], dtype=cp.float32)
            checker_color = closest_object.get_checker_color(point, color1, color2, scale=1.5)
            base_color = checker_color
        else:
            base_color = material.diffuse
        
        # 环境光
        color = material.ambient * self.ambient_light
        
        # 漫反射和镜面反射
        for light in self.lights:
            light_dir = (light['position'] - point).normalize()
            shadow_ray = Ray(point + normal * 1e-4, light_dir)
            in_shadow = False
            for obj in self.objects:
                if obj.intersect(shadow_ray):
                    in_shadow = True
                    break
            ndotl = normal.dot(light_dir)
            if not in_shadow and ndotl > 0:
                # 漫反射
                diffuse = ndotl
                color += base_color * light['color'] * diffuse
                # 镜面反射
                view_dir = (ray.origin - point).normalize()
                reflect_dir = normal * 2 * ndotl - light_dir
                specular = max(0, view_dir.dot(reflect_dir)) ** material.shininess
                color += material.specular * light['color'] * specular
        
        # 递归计算反射
        if depth < self.max_depth and not isinstance(closest_object, Plane):
            reflect_dir = ray.direction - normal * 2 * normal.dot(ray.direction)
            reflect_ray = Ray(point + normal * 1e-4, reflect_dir)
            reflect_color = self.trace_ray(reflect_ray, depth + 1)
            color += material.specular * reflect_color
        
        return cp.clip(color, 0, 1)