import torch
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
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    print("开始初始化场景...")
    scene = Scene()

    # 创建材质
    print("创建材质...")
    # 红色材质 - 球体
    red_material = Material(
        ambient=torch.tensor([0.1, 0.0, 0.0], dtype=torch.float32, device=device),
        diffuse=torch.tensor([0.7, 0.0, 0.0], dtype=torch.float32, device=device),
        specular=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=device),
        shininess=50.0
    )
    # 绿色材质 - 四面体
    green_material = Material(
        ambient=torch.tensor([0.0, 0.1, 0.0], dtype=torch.float32, device=device),
        diffuse=torch.tensor([0.0, 0.7, 0.0], dtype=torch.float32, device=device),
        specular=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=device),
        shininess=50.0
    )
    # 粉色材质 - 球体
    pink_material = Material(
        ambient=torch.tensor([0.1, 0.0, 0.1], dtype=torch.float32, device=device),
        diffuse=torch.tensor([1.0, 0.4, 0.7], dtype=torch.float32, device=device),
        specular=torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, device=device),
        shininess=30.0
    )
    # 亮蓝色材质 - 立方体
    blue_material = Material(
        ambient=torch.tensor([0.0, 0.0, 0.2], dtype=torch.float32, device=device),
        diffuse=torch.tensor([0.3, 0.6, 1.0], dtype=torch.float32, device=device),
        specular=torch.tensor([0.8, 0.8, 1.0], dtype=torch.float32, device=device),
        shininess=80.0
    )
    # 地板材质
    floor_material = Material(
        ambient=torch.tensor([0.1, 0.1, 0.1], dtype=torch.float32, device=device),
        diffuse=torch.tensor([0.9, 0.9, 0.9], dtype=torch.float32, device=device),
        specular=torch.tensor([0.2, 0.2, 0.2], dtype=torch.float32, device=device),
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
    angle_y = torch.tensor(45.0, device=device).deg2rad()
    angle_x = torch.tensor(35.26438968, device=device).deg2rad()
    Ry = torch.tensor([
        [torch.cos(angle_y).item(), 0, torch.sin(angle_y).item()],
        [0, 1, 0],
        [-torch.sin(angle_y).item(), 0, torch.cos(angle_y).item()]
    ], dtype=torch.float32, device=device)
    Rx = torch.tensor([
        [1, 0, 0],
        [0, torch.cos(angle_x).item(), -torch.sin(angle_x).item()],
        [0, torch.sin(angle_x).item(), torch.cos(angle_x).item()]
    ], dtype=torch.float32, device=device)
    cube_rotation = torch.matmul(Ry, Rx)
    # 立方体位置
    scene.add_object(Cube(Vector3(2.2, 0, -5), 1.2, blue_material, rotation=cube_rotation))

    # 正四面体
    # 四面体size
    tetra_size = 2.8
    a = tetra_size
    h = torch.sqrt(torch.tensor(3.0, device=device)) / 2 * a
    H = a * torch.sqrt(torch.tensor(2.0/3.0, device=device))
    center_z = -8
    # 顶点位置
    v0 = Vector3(0.0, 0.0, float(center_z + 2*h/3))
    v1 = Vector3(float(a/2), 0.0, float(center_z - h/3))
    v2 = Vector3(float(-a/2), 0.0, float(center_z - h/3))
    v3 = Vector3(0.0, float(H), float(center_z - h/3))
    # 四面体旋转矩阵
    angle_y = float(torch.tensor(-200.0, device=device).deg2rad())
    angle_x = float(torch.pi + torch.tensor(-10.0, device=device).deg2rad())
    Ry = torch.tensor([
        [float(torch.cos(torch.tensor(angle_y, device=device))), 0, float(torch.sin(torch.tensor(angle_y, device=device)))],
        [0, 1, 0],
        [float(-torch.sin(torch.tensor(angle_y, device=device))), 0, float(torch.cos(torch.tensor(angle_y, device=device)))]
    ], dtype=torch.float32, device=device)
    Rx = torch.tensor([
        [1, 0, 0],
        [0, float(torch.cos(torch.tensor(angle_x, device=device))), float(-torch.sin(torch.tensor(angle_x, device=device)))],
        [0, float(torch.sin(torch.tensor(angle_x, device=device))), float(torch.cos(torch.tensor(angle_x, device=device)))]
    ], dtype=torch.float32, device=device)
    R = torch.matmul(Ry, Rx)
    # 计算垂直抬升量，确保四面体最低点在地面y=-1之上
    y_offset = H - 1.0 + 0.01
    x_offset_tetra = 0.9
    def rotate(v):
        v_local = torch.tensor([v.x, v.y, v.z - center_z], device=device)
        v_rot = torch.matmul(R, v_local)
        return Vector3(v_rot[0].item() + x_offset_tetra, v_rot[1].item() + y_offset, v_rot[2].item() + center_z)
    v0r = rotate(v0)
    v1r = rotate(v1)
    v2r = rotate(v2)
    v3r = rotate(v3)
    # 四面体位置
    scene.add_object(Tetrahedron(v0r, v1r, v2r, v3r, green_material))

    # 光源
    print("设置光源...")
    scene.add_light(Vector3(5, 8, 2), torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, device=device))
    scene.add_light(Vector3(-6, 6, 0), torch.tensor([0.2, 0.2, 0.2], dtype=torch.float32, device=device))
    scene.add_light(Vector3(4, 4, -4), torch.tensor([0.35, 0.35, 0.5], dtype=torch.float32, device=device))
    scene.add_light(Vector3(0, 4, 4), torch.tensor([0.4, 0.4, 0.5], dtype=torch.float32, device=device))

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
    # width, height = 800, 600
    width, height = 400, 300
    # 创建GPU图像数组
    image_gpu = torch.zeros((height, width, 3), dtype=torch.float32, device=device)
    start_time = time.time()
    
    print(f"图像尺寸: {width}x{height}")
    print("初始化相机和场景...")
    
    try:
        # 使用批处理方式渲染
        batch_size = 4096
        total_pixels = width * height
        
        print(f"\n开始渲染 {total_pixels} 个像素...")
        render_pbar = tqdm(total=total_pixels, desc="渲染进度", unit="像素")
        
        for batch_start in range(0, total_pixels, batch_size):
            batch_end = min(batch_start + batch_size, total_pixels)
            idxs = torch.arange(batch_start, batch_end, device=device)
            Y = torch.div(idxs, width, rounding_mode='floor')
            X = idxs % width
            rays_batch = camera.get_rays_batch(X, Y, width, height)
            colors = scene.trace_rays_batch(rays_batch)
            assert colors.shape[0] == X.shape[0] == Y.shape[0], "Batch渲染输出数量不一致"
            # 用一维索引写入，彻底避免条带错位
            image_gpu.view(-1, 3)[idxs] = torch.clamp(colors, 0, 1)
            render_pbar.update(len(X))
            render_pbar.refresh()
        
        render_pbar.close()
        
    except Exception as e:
        print(f"\n渲染过程中出现错误: {str(e)}")
        raise
    
    print("\n开始将结果从GPU复制到CPU...")
    # 将结果从GPU复制到CPU
    image = image_gpu.cpu()
    
    # 计算总时间
    total_time = time.time() - start_time
    print(f"\n渲染完成！总用时: {total_time:.1f}秒")

    # 保存图像
    print("渲染完成，正在保存图像...")
    image = (image * 255).to(torch.uint8).numpy()
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