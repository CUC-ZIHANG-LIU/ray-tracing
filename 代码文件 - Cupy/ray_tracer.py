import cupy as cp
from PIL import Image
import os
from geometry import Sphere, Cube, Cone, Plane, Tetrahedron
from vector import Vector3
from material import Material
from scene import Scene
from camera import Camera
import time
from tqdm import tqdm
    
def main():

    print("开始初始化场景...")
    scene = Scene()

    # 创建材质
    print("创建材质...")
    # 红色材质 - 球体
    red_material = Material(
        ambient=cp.array([0.1, 0.0, 0.0], dtype=cp.float32),
        diffuse=cp.array([0.7, 0.0, 0.0], dtype=cp.float32),
          specular=cp.array([1.0, 1.0, 1.0], dtype=cp.float32),
        shininess=50.0
    )
    # 绿色材质 - 四面体
    green_material = Material(
        ambient=cp.array([0.0, 0.1, 0.0], dtype=cp.float32),
        diffuse=cp.array([0.0, 0.7, 0.0], dtype=cp.float32),
        specular=cp.array([1.0, 1.0, 1.0], dtype=cp.float32),
        shininess=50.0
    )
    # 粉色材质 - 球体
    pink_material = Material(
        ambient=cp.array([0.1, 0.0, 0.1], dtype=cp.float32),      # 轻微环境光
    diffuse=cp.array([1.0, 0.4, 0.7], dtype=cp.float32),      # 纯粉色
    specular=cp.array([0.5, 0.5, 0.5], dtype=cp.float32),     # 适度镜面反射
        shininess=30.0                                            # 适度高光锐度
    )
    # 亮蓝色材质 - 立方体
    blue_material = Material(
        ambient=cp.array([0.0, 0.0, 0.2], dtype=cp.float32),
        diffuse=cp.array([0.3, 0.6, 1.0], dtype=cp.float32),
        specular=cp.array([0.8, 0.8, 1.0], dtype=cp.float32),
        shininess=80.0
    )
    # 地板材质
    floor_material = Material(
        ambient=cp.array([0.1, 0.1, 0.1], dtype=cp.float32),
        diffuse=cp.array([0.9, 0.9, 0.9], dtype=cp.float32),
        specular=cp.array([0.2, 0.2, 0.2], dtype=cp.float32),
        shininess=10.0
    )

    # 添加几何体到场景  
    print("添加几何体到场景...")
    # 地板
    scene.add_object(Plane(Vector3(0, -1, 0), Vector3(0, 1, 0), floor_material))

    # 红色球体
    scene.add_object(Sphere(Vector3(-2.2, 0, -4), 0.7, red_material))

    # 粉色球体
    scene.add_object(Sphere(Vector3(-1.1, -0.3, -8), 0.7, pink_material))
   
    # 立方体
    # 立方体旋转矩阵：先绕y轴45°，再绕x轴35.26°
    angle_y = cp.deg2rad(45)
    angle_x = cp.deg2rad(35.26438968)  # arccos(1/sqrt(3)) ≈ 54.74°，sin⁻¹(1/√3) ≈ 35.26°
    Ry = cp.array([
        [cp.cos(angle_y).item(), 0, cp.sin(angle_y).item()],
        [0, 1, 0],
        [-cp.sin(angle_y).item(), 0, cp.cos(angle_y).item()]
    ], dtype=cp.float32)
    Rx = cp.array([
        [1, 0, 0],
        [0, cp.cos(angle_x).item(), -cp.sin(angle_x).item()],
        [0, cp.sin(angle_x).item(), cp.cos(angle_x).item()]
    ], dtype=cp.float32)
    cube_rotation = Ry @ Rx
    # 立方体位置
    scene.add_object(Cube(Vector3(2.2, 0, -5), 1.2, blue_material, rotation=cube_rotation))

    # 正四面体
    # 四面体size
    tetra_size = 2.8
    a = tetra_size
    h = cp.sqrt(3) / 2 * a
    H = a * cp.sqrt(2/3)
    center_z = -8
    # 顶点位置
    v0 = Vector3(0.0, 0.0, float(center_z + 2*h/3))
    v1 = Vector3(float(a/2), 0.0, float(center_z - h/3))
    v2 = Vector3(float(-a/2), 0.0, float(center_z - h/3))
    v3 = Vector3(0.0, float(H), float(center_z - h/3))  # 使用H作为顶点高度
    # 四面体旋转矩阵
    angle_y = float(cp.deg2rad(-200))  # 顺时针200°
    angle_x = float(cp.pi + cp.deg2rad(-10))  # 上下颠倒后再逆时针旋转10°
    Ry = cp.array([
        [float(cp.cos(angle_y)), 0, float(cp.sin(angle_y))],
        [0, 1, 0],
        [float(-cp.sin(angle_y)), 0, float(cp.cos(angle_y))]
    ], dtype=cp.float32)
    Rx = cp.array([
        [1, 0, 0],
        [0, float(cp.cos(angle_x)), float(-cp.sin(angle_x))],
        [0, float(cp.sin(angle_x)), float(cp.cos(angle_x))]
    ], dtype=cp.float32)
    R = Ry @ Rx
    # 计算垂直抬升量，确保四面体最低点在地面y=-1之上
    y_offset = H - 1.0 + 0.01 # H 是四面体高度，1.0是地板y坐标(-1)的绝对值，0.01 是微小偏移量
    x_offset_tetra = 0.9 # 四面体向右移动0.9个单位
    def rotate(v):
        v_local = cp.array([v.x, v.y, v.z - center_z])
        v_rot = R @ v_local
        return Vector3(v_rot[0] + x_offset_tetra, v_rot[1] + y_offset, v_rot[2] + center_z)
    v0r = rotate(v0)
    v1r = rotate(v1)
    v2r = rotate(v2)
    v3r = rotate(v3)
    # 四面体位置
    scene.add_object(Tetrahedron(v0r, v1r, v2r, v3r, green_material))

    # 光源
    print("设置光源...")
    scene.add_light(Vector3(5, 8, 2), cp.array([0.5, 0.5, 0.5], dtype=cp.float32))
    scene.add_light(Vector3(-6, 6, 0), cp.array([0.2, 0.2, 0.2], dtype=cp.float32))
    scene.add_light(Vector3(4, 4, -4), cp.array([0.35, 0.35, 0.5], dtype=cp.float32))
    scene.add_light(Vector3(0, 4, 4), cp.array([0.4, 0.4, 0.5], dtype=cp.float32))

    # 相机
    print("设置相机...")
    camera = Camera(
        position=Vector3(0, 1.0, 2),
        look_at=Vector3(0, 0, -5),
        up=Vector3(0, 1, 0),
        fov=50
    )

    # 渲染
    print("开始渲染场景...")
    width, height = 800, 600
    # 创建GPU图像数组
    image_gpu = cp.zeros((height, width, 3), dtype=cp.float32)
    start_time = time.time()
    
    print(f"图像尺寸: {width}x{height}")
    print("初始化相机和场景...")
    
    try:
        # 创建CUDA流
        stream = cp.cuda.Stream()
        
        with stream:
            print("\n开始预计算所有射线...")
            # 预计算所有射线
            rays = []
            # 创建预计算进度条
            precalc_pbar = tqdm(total=height, desc="预计算射线", unit="行")
            for i in range(height):
                for j in range(width):
                    ray = camera.get_ray(j, i, width, height)
                    rays.append(ray)
                precalc_pbar.update(1)
                precalc_pbar.refresh()
            precalc_pbar.close()
            
            print(f"\n开始渲染 {len(rays)} 个像素...")
            # 创建渲染进度条
            render_pbar = tqdm(total=len(rays), desc="渲染进度", unit="像素")
            # 一次性处理所有像素
            for k, ray in enumerate(rays):
                row = k // width
                col = k % width
                color = scene.trace_ray(ray, 0)
                image_gpu[row, col] = cp.clip(color, 0, 1)
                render_pbar.update(1)
                render_pbar.refresh()
            render_pbar.close()
            
    except Exception as e:
        print(f"\n渲染过程中出现错误: {str(e)}")
        raise
    finally:
        # 确保所有GPU操作完成
        stream.synchronize()
    
    print("\n开始将结果从GPU复制到CPU...")
    # 将结果从GPU复制到CPU
    image = image_gpu.get()
    
    # 计算总时间
    total_time = time.time() - start_time
    print(f"\n渲染完成！总用时: {total_time:.1f}秒")

    # 保存图像
    print("渲染完成，正在保存图像...")
    image = (image * 255).astype(cp.uint8).get()
    # 获取当前脚本的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建完整的输出文件路径
    output_path = os.path.join(script_dir, 'output.png')
    Image.fromarray(image).save(output_path)
    total_time = time.time() - start_time
    print(f"渲染完成！总用时: {total_time:.1f}秒")
    print(f"图像已保存到: {output_path}")

if __name__ == "__main__":
    main()