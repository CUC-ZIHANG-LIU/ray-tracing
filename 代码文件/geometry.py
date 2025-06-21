import numpy as np
from vector import Vector3

# 几何体基类
class Geometry:
    def __init__(self, position, material):
        self.position = position
        self.material = material

# 球体
class Sphere(Geometry):
    def __init__(self, position, radius, material):
        super().__init__(position, material)
        self.radius = radius
    
    def intersect(self, ray):
        # 射线与球体的求交算法
        # 球体方程: (p - C) . (p - C) = R^2, 其中 p 是球面上一点，C是球心，R是半径
        # 射线方程: p = O + tD, 其中 O 是射线原点，D是射线方向，t是距离参数
        # 将射线方程代入球体方程，得到关于t的二次方程：
        # (O + tD - C) . (O + tD - C) = R^2
        # 令 oc = O - C (从球心指向射线原点的向量)
        # 则 (oc + tD) . (oc + tD) = R^2
        # 展开: oc.oc + 2*t*(oc.D) + t^2*(D.D) = R^2
        # 整理成标准二次方程: (D.D)t^2 + 2*(oc.D)t + (oc.oc - R^2) = 0
        # 即 At^2 + Bt + C = 0
        
        oc = ray.origin - self.position # oc = O - C
        a = ray.direction.dot(ray.direction) # A = D.D
        b = 2.0 * oc.dot(ray.direction)      # B = 2*(oc.D)
        c = oc.dot(oc) - self.radius * self.radius # C = oc.oc - R^2
        
        discriminant = b * b - 4 * a * c # 计算判别式 (delta)
        
        # 判断射线与球体是否相交
        if discriminant < 0:
            return None
        
        # 计算最近的交点距离 t
        # 二次方程解: t = (-B +- sqrt(delta)) / (2A)
        t = (-b - np.sqrt(discriminant)) / (2.0 * a) # 尝试第一个解（较小的t，即最近的交点）
        if t < 0:
            # 如果第一个解小于0，说明交点在射线原点后面，尝试第二个解
            t = (-b + np.sqrt(discriminant)) / (2.0 * a)
            if t < 0:
                # 如果第二个解也小于0，说明两个交点都在射线原点后面，不相交
                return None
        
        # 计算交点坐标和法线
        point = ray.origin + ray.direction * t # 交点 = 射线原点 + t * 射线方向
        normal = (point - self.position).normalize() # 法线 = (交点 - 球心) 的单位向量
        return {'t': t, 'point': point, 'normal': normal}

# 立方体
class Cube(Geometry):
    def __init__(self, position, size, material, rotation=None):
        super().__init__(position, material)
        self.size = size
        if rotation is None:
            self.rotation = np.eye(3, dtype=np.float32)  # 单位矩阵，无旋转
        else:
            self.rotation = rotation
    
    def intersect(self, ray):
        # 射线与旋转立方体（AABB）的求交算法
        # 由于立方体可能被旋转，直接在世界坐标系求交复杂。
        # 策略：将射线变换到立方体的"本地坐标系"（即立方体未旋转时的坐标系），
        # 在本地坐标系中进行轴对齐包围盒 (AABB) 求交，最后将结果转换回世界坐标系。
        
        # === 1. 将射线变换到立方体本地坐标系 ===
        # 首先，计算旋转矩阵的逆矩阵，用于将世界坐标变换到本地坐标。
        inv_rot = np.linalg.inv(self.rotation)
        # 将射线原点平移到立方体中心，然后进行逆旋转。
        local_origin = Vector3(*inv_rot.dot((ray.origin - self.position).to_array()))
        # 将射线方向直接进行逆旋转（方向向量不涉及平移）。
        local_direction = Vector3(*inv_rot.dot(ray.direction.to_array()))
        
        # === 2. 在本地坐标系中进行轴对齐包围盒 (AABB) 求交 ===
        # AABB立方体的边界是 [-size/2, size/2] 在X, Y, Z轴上。
        # 射线与AABB的求交算法通常通过计算射线与每个轴平行面（共6个面）的交点，
        # 然后找到所有有效交点中 t_min 的最大值和 t_max 的最小值。
        t_min = float('-inf') # 射线进入AABB的最远t值
        t_max = float('inf')  # 射线离开AABB的最近t值
        
        for i in range(3): # 遍历X, Y, Z三个轴
            # 获取当前轴的射线原点分量和方向分量
            orig_comp = local_origin.to_array()[i]
            dir_comp = local_direction.to_array()[i]
            
            # 计算当前轴的AABB边界
            min_bound = -self.size / 2
            max_bound = self.size / 2
            
            if abs(dir_comp) < 1e-6: # 如果射线方向分量接近0（射线平行于当前轴的面）
                if orig_comp < min_bound or orig_comp > max_bound:
                    # 如果射线原点在此轴方向上在AABB之外，则无交点
                    return None
                # 如果射线原点在此轴方向上在AABB之内，则此轴不限制t_min/t_max，跳过
            else:
                # 计算射线与当前轴的两个平行面的交点t值
                t1 = (min_bound - orig_comp) / dir_comp
                t2 = (max_bound - orig_comp) / dir_comp
                
                # 确保t1是较小的进入点，t2是较大的离开点
                t_enter = min(t1, t2)
                t_exit = max(t1, t2)
                
                # 更新全局的t_min和t_max
                t_min = max(t_min, t_enter)
                t_max = min(t_max, t_exit)
        
        # 检查是否有效交点
        if t_max < t_min or t_max < 0:
            # 如果t_max小于t_min，说明射线没有穿过AABB
            # 如果t_max小于0，说明AABB在射线后面
            return None
        
        # 确定最终的交点t值（取最近的那个有效交点）
        t = t_min if t_min > 0 else t_max # 如果t_min在射线前方，则取t_min；否则取t_max
        
        # 计算本地坐标系下的交点
        local_point = local_origin + local_direction * t
        
        # === 3. 计算本地法线并变换回世界坐标系 ===
        normal = Vector3(0, 0, 0) # 初始化法线向量
        epsilon = 1e-4 # 用于浮点数比较的容差
        
        # 根据交点在AABB的哪个面上，确定本地法线。
        # AABB的法线总是轴对齐的 (例如，+X, -X, +Y, -Y, +Z, -Z)
        for i in range(3): # 遍历X, Y, Z轴
            if abs(local_point.to_array()[i] + self.size/2) < epsilon:
                # 如果交点在当前轴的负方向边界上（例如：x = -size/2）
                arr = [0, 0, 0]
                arr[i] = -1 # 法线指向负方向
                normal = Vector3(*arr)
                break
            elif abs(local_point.to_array()[i] - self.size/2) < epsilon:
                # 如果交点在当前轴的正方向边界上（例如：x = +size/2）
                arr = [0, 0, 0]
                arr[i] = 1 # 法线指向正方向
                normal = Vector3(*arr)
                break
        
        # 将本地法线向量通过立方体的旋转矩阵变换回世界坐标系，并归一化。
        world_normal = Vector3(*self.rotation.dot(normal.to_array())).normalize()
        # 将本地交点通过立方体的旋转矩阵变换回世界坐标系，并加上立方体的平移量。
        world_point = self.position + Vector3(*self.rotation.dot(local_point.to_array()))
        
        # === 4. 计算世界坐标系下的射线距离 t_world ===
        # t_world 是从射线原点到世界交点的实际距离，确保与射线方向的长度一致。
        t_world = (world_point - ray.origin).length() / ray.direction.length()
        
        return {'t': t_world, 'point': world_point, 'normal': world_normal}

# 圆锥体
class Cone(Geometry):
    def __init__(self, position, radius, height, material):
        super().__init__(position, material)
        self.radius = radius
        self.height = height
    
    def intersect(self, ray):
        # 射线与圆锥体的求交算法 (简化版)
        # 假设圆锥体的底面中心在 position 的 z 轴，高度沿 z 轴向上。
        # 圆锥体方程涉及到 x^2 + y^2 = (k*z)^2 的形式，其中 k = radius / height。
        
        # 将射线原点和方向平移，使圆锥体底面中心在原点。
        # local_origin = ray.origin - self.position
        # local_direction = ray.direction
        
        # 构建二次方程 Ax^2 + Bx + C = 0 的系数
        k = (self.radius / self.height) ** 2
        a = ray.direction.x * ray.direction.x + ray.direction.y * ray.direction.y - k * ray.direction.z * ray.direction.z
        b = 2 * (ray.origin.x * ray.direction.x + ray.origin.y * ray.direction.y - k * ray.origin.z * ray.direction.z)
        c = ray.origin.x * ray.origin.x + ray.origin.y * ray.origin.y - k * ray.origin.z * ray.origin.z
        
        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            # 无实数解，不相交
            return None
        
        t1 = (-b - np.sqrt(discriminant)) / (2 * a)
        t2 = (-b + np.sqrt(discriminant)) / (2 * a)
        
        # 筛选有效交点
        if t1 < 0 and t2 < 0:
            return None # 两个交点都在射线后面
        
        t = t1 if t1 > 0 else t2 # 取第一个在射线前方的交点
        point = ray.origin + ray.direction * t
        
        # 检查交点是否在圆锥体的高度范围内
        if point.z < self.position.z or point.z > self.position.z + self.height:
            return None
        
        # 计算交点处的法线
        # 圆锥体侧面的法线计算比较复杂，这里是简化版本，可能不够精确。
        # 通常需要考虑顶点和底面的法线。
        normal = Vector3(
            point.x,
            point.y,
            -k * (point.z - self.position.z) # Z分量是根据圆锥体的斜率和高度计算的
        ).normalize()
        
        return {'t': t, 'point': point, 'normal': normal}

# 平面（棋盘格地板）
class Plane(Geometry):
    def __init__(self, position, normal, material):
        super().__init__(position, material)
        self.normal = normal.normalize() # 平面法线，用于定义平面的方向
    def intersect(self, ray):
        # 射线与平面的求交算法
        # 平面方程: (p - P0) . N = 0, 其中 P0 是平面上一点，N是平面法线
        # 射线方程: p = O + tD
        # 代入平面方程: (O + tD - P0) . N = 0
        # 展开: (O - P0) . N + t (D . N) = 0
        # 解t: t = - (O - P0) . N / (D . N)
        # t = (P0 - O) . N / (D . N)
        
        denom = self.normal.dot(ray.direction) # 计算分母 (D . N)
        if abs(denom) < 1e-6:
            # 如果分母接近0，表示射线平行于平面，或射线在平面内
            return None
        
        # 计算t值
        t = (self.position - ray.origin).dot(self.normal) / denom
        if t < 0:
            # 如果t小于0，表示交点在射线原点后面
            return None
        
        # 计算交点坐标
        point = ray.origin + ray.direction * t
        return {'t': t, 'point': point, 'normal': self.normal} # 平面法线处处相同
    
    # 棋盘格着色辅助函数
    def get_checker_color(self, point, color1, color2, scale=1.0):
        # 根据交点在平面上的X和Z坐标，计算棋盘格颜色
        # 通过对坐标进行缩放和取整，判断其所在的"格子"是奇数还是偶数，从而决定颜色。
        x, z = point.x, point.z
        if int((x * scale) % 2) == int((z * scale) % 2):
            return color1 # 如果X和Z坐标的"奇偶性"相同，则使用颜色1
        else:
            return color2 # 否则使用颜色2

# 正四面体
class Tetrahedron(Geometry):
    def __init__(self, v0, v1, v2, v3, material):
        super().__init__(v0, material) # 将第一个顶点作为"位置"基准
        self.vertices = [v0, v1, v2, v3] # 存储四面体的四个顶点
        # 定义四面体的四个三角面，每个面由三个顶点组成
        self.faces = [
            (v0, v1, v2), # 面1
            (v0, v1, v3), # 面2
            (v0, v2, v3), # 面3
            (v1, v2, v3)  # 面4
        ]

    def intersect(self, ray):
        # 射线与四面体的求交算法
        # 四面体由四个三角形面组成。射线与四面体的交点，就是射线与这四个三角形面的最近交点。
        closest_t = float('inf') # 记录最近的交点距离
        hit_info = None          # 存储最近交点的详细信息
        
        for v0, v1, v2 in self.faces: # 遍历四面体的每个三角形面
            # 调用射线与三角形的求交函数
            result = self.ray_triangle_intersect(ray, v0, v1, v2)
            if result and result['t'] < closest_t:
                # 如果有交点，并且这个交点比之前找到的交点更近，则更新最近交点信息
                closest_t = result['t']
                hit_info = result
        
        if hit_info:
            # 如果找到有效交点，将四面体的材质信息添加到命中结果中
            hit_info['material'] = self.material
        return hit_info # 返回最近的交点信息

    def ray_triangle_intersect(self, ray, v0, v1, v2):
        # 射线与三角形的求交算法 (Möller–Trumbore算法)
        # 这是一个高效的算法，直接计算重心坐标，避免了额外的平面求交步骤。
        
        epsilon = 1e-6 # 浮点数比较容差，避免除以零或非常小的数
        
        # 计算三角形的两条边向量
        edge1 = v1 - v0 
        edge2 = v2 - v0 
        
        # 步骤1: 判断射线是否与三角形平面平行
        # H = D x E2 (D: 射线方向, E2: 三角形边2)
        h = ray.direction.cross(edge2)
        # a = E1 . H (E1: 三角形边1)
        a = edge1.dot(h)
        
        if -epsilon < a < epsilon:
            # 如果a接近0，表示射线与三角形平面平行 (或共面)，没有唯一交点
            return None
        
        f = 1.0 / a
        
        # 步骤2: 计算重心坐标 u
        # S = O - V0 (O: 射线原点, V0: 三角形第一个顶点)
        s = ray.origin - v0
        # u = f * (S . H)
        u = f * s.dot(h)
        
        if u < 0.0 or u > 1.0:
            # 如果u不在[0, 1]范围内，表示交点在三角形外部（沿着E1方向）
            return None
        
        # 步骤3: 计算重心坐标 v
        # Q = S x E1
        q = s.cross(edge1)
        # v = f * (D . Q)
        v = f * ray.direction.dot(q)
        
        if v < 0.0 or u + v > 1.0:
            # 如果v不在[0, 1]范围内，或者u+v大于1，表示交点在三角形外部（沿着E2方向）
            return None
        
        # 步骤4: 计算射线距离 t
        # t = f * (E2 . Q)
        t = f * edge2.dot(q)
        
        if t > epsilon:
            # 如果t大于epsilon（确保交点在射线前方，避免自身相交问题）
            point = ray.origin + ray.direction * t # 计算交点坐标
            # 计算三角形法线（归一化）
            normal = edge1.cross(edge2).normalize()
            # 确保法线方向与射线方向相反（法线指向外部）
            if normal.dot(ray.direction) > 0:
                normal = normal * -1
            return {'t': t, 'point': point, 'normal': normal}
        
        return None # t小于等于epsilon，交点在射线原点处或后面，视为不相交 