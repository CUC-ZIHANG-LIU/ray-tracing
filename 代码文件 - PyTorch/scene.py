import torch
from vector import Vector3
from camera import Ray
from geometry import Plane

class Scene:
    def __init__(self):
        self.objects = []
        self.lights = []
        self.ambient_light = torch.tensor([0.2, 0.2, 0.2], dtype=torch.float32, device='cuda')
        self.max_depth = 3
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def add_object(self, obj):
        if not hasattr(obj, 'intersect'):
            raise TypeError("对象必须实现intersect方法")
        self.objects.append(obj)
    
    def add_light(self, position, color):
        if not isinstance(position, Vector3):
            raise TypeError("position必须是Vector3类型")
        if not isinstance(color, torch.Tensor) or color.shape != (3,):
            raise TypeError("color必须是形状为(3,)的tensor")
        self.lights.append({'position': position, 'color': color})
    
    def trace_ray(self, ray, depth):
        if not isinstance(ray, dict) or 'origin' not in ray or 'direction' not in ray:
            raise TypeError("ray必须是包含'origin'和'direction'的字典")
        if not isinstance(depth, int) or depth < 0:
            raise TypeError("depth必须是非负整数")
            
        if depth >= self.max_depth:
            return torch.zeros(3, dtype=torch.float32, device=self.device)
        
        # 找到最近的交点
        closest_intersection = None
        closest_object = None
        min_distance = float('inf')
        
        for obj in self.objects:
            intersection = obj.intersect(ray)
            if intersection and intersection['distance'] < min_distance:
                min_distance = intersection['distance']
                closest_intersection = intersection
                closest_object = obj
        
        if not closest_intersection:
            return torch.zeros(3, dtype=torch.float32, device=self.device)
        
        # 计算光照
        point = closest_intersection['point']
        normal = closest_intersection['normal']
        material = closest_object.material
        
        # 棋盘格地板 
        if isinstance(closest_object, Plane):
            # 棋盘格两种颜色
            color1 = torch.tensor([0.9, 0.9, 0.9], dtype=torch.float32, device='cuda')
            color2 = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float32, device='cuda')
            checker_color = closest_object.get_checker_color(point, color1, color2, scale=1.5)
            base_color = checker_color
        else:
            base_color = material.diffuse
        
        # 环境光
        color = material.ambient * self.ambient_light
        
        # 漫反射和镜面反射
        for light in self.lights:
            light_pos = torch.tensor([light['position'].x, light['position'].y, light['position'].z], 
                                   dtype=torch.float32, device=self.device)
            light_dir = (light_pos - torch.tensor([point.x, point.y, point.z], 
                                                dtype=torch.float32, device=self.device)).normalize()
            light_distance = torch.norm(light_pos - torch.tensor([point.x, point.y, point.z], 
                                                              dtype=torch.float32, device=self.device))
            
            # 检查阴影
            shadow_ray = {
                'origin': point + normal * 1e-4,  # 避免自阴影
                'direction': light_dir
            }
            in_shadow = False
            for obj in self.objects:
                shadow_intersection = obj.intersect(shadow_ray)
                if shadow_intersection and shadow_intersection['distance'] < light_distance:
                    in_shadow = True
                    break
            
            if not in_shadow:
                # 漫反射
                diffuse = max(0, normal.dot(light_dir))
                color += base_color * light['color'] * diffuse
                
                # 镜面反射
                view_dir = (torch.tensor([ray['origin'].x, ray['origin'].y, ray['origin'].z], 
                                       dtype=torch.float32, device=self.device) - 
                           torch.tensor([point.x, point.y, point.z], 
                                      dtype=torch.float32, device=self.device)).normalize()
                reflect_dir = light_dir.reflect(normal)
                specular = max(0, view_dir.dot(reflect_dir)) ** material.shininess
                color += material.specular * light['color'] * specular
        
        # 递归计算反射
        if material.shininess > 0:
            reflect_dir = ray['direction'].reflect(normal)
            reflect_ray = {
                'origin': point + normal * 1e-4,
                'direction': reflect_dir
            }
            reflect_color = self.trace_ray(reflect_ray, depth + 1)
            color += material.specular * reflect_color
        
        return torch.clamp(color, 0, 1)
    
    def trace_rays_batch(self, rays_batch, depth=0):
        batch_size = rays_batch['origins'].shape[0]
        if depth >= self.max_depth:
            return torch.zeros((batch_size, 3), dtype=torch.float32, device=self.device)
        closest_distances = torch.full((batch_size,), float('inf'), device=self.device)
        closest_colors = torch.zeros((batch_size, 3), device=self.device)
        closest_normals = torch.zeros((batch_size, 3), device=self.device)
        closest_points = torch.zeros((batch_size, 3), device=self.device)
        closest_materials = [None] * batch_size
        for obj in self.objects:
            intersections = obj.intersect_batch(rays_batch)
            if intersections is None:
                continue
            mask = intersections['distances'] < closest_distances
            closest_distances[mask] = intersections['distances'][mask]
            closest_colors[mask] = intersections['colors'][mask]
            closest_normals[mask] = intersections['normals'][mask]
            closest_points[mask] = intersections['points'][mask]
            for i in range(batch_size):
                if mask[i]:
                    closest_materials[i] = obj.material
        final_colors = torch.zeros((batch_size, 3), device=self.device)
        for i, material in enumerate(closest_materials):
            if material is not None:
                final_colors[i] += material.ambient * self.ambient_light
        for light in self.lights:
            light_pos = torch.tensor([light['position'].x, light['position'].y, light['position'].z], dtype=torch.float32, device=self.device)
            light_color = light['color']
            light_dirs = light_pos - closest_points
            light_dirs = light_dirs / torch.norm(light_dirs, dim=1, keepdim=True)
            shadow_rays = {
                'origins': closest_points + closest_normals * 1e-4,
                'directions': light_dirs
            }
            in_shadow = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
            for obj in self.objects:
                shadow_intersections = obj.intersect_batch(shadow_rays)
                if shadow_intersections is not None:
                    in_shadow |= shadow_intersections['distances'] < torch.norm(light_pos - closest_points, dim=1)
            diffuse = torch.clamp(torch.sum(closest_normals * light_dirs, dim=1), 0, 1).unsqueeze(-1)
            view_dirs = (rays_batch['origins'] - closest_points)
            view_dirs = view_dirs / torch.norm(view_dirs, dim=1, keepdim=True)
            reflect_dirs = 2 * torch.sum(closest_normals * light_dirs, dim=1, keepdim=True) * closest_normals - light_dirs
            specular = torch.clamp(torch.sum(view_dirs * reflect_dirs, dim=1), 0, 1).unsqueeze(-1)
            for i, material in enumerate(closest_materials):
                if material is not None and not in_shadow[i]:
                    final_colors[i] += material.diffuse * light_color * diffuse[i]
                    final_colors[i] += material.specular * light_color * (specular[i] ** material.shininess)
        if depth < self.max_depth - 1:
            reflect_dirs = 2 * torch.sum(closest_normals * rays_batch['directions'], dim=1, keepdim=True) * closest_normals - rays_batch['directions']
            reflect_rays = {
                'origins': closest_points + closest_normals * 1e-4,
                'directions': reflect_dirs
            }
            reflect_colors = self.trace_rays_batch(reflect_rays, depth + 1)
            for i, material in enumerate(closest_materials):
                if material is not None and material.shininess > 0:
                    final_colors[i] += material.specular * reflect_colors[i]
        return torch.clamp(final_colors, 0, 1)