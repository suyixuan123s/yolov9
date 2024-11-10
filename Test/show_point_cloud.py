import open3d as o3d

def visualize_point_cloud(file_path):
    # 加载点云文件
    pcd = o3d.io.read_point_cloud(file_path)

    # 检查点云是否成功加载
    if pcd.is_empty():
        print(f"无法加载点云文件: {file_path}")
        return

    # 可视化点云
    o3d.visualization.draw_geometries([pcd],
                                        window_name="Point Cloud Viewer",
                                        width=800,
                                        height=600,
                                        left=50,
                                        top=50,
                                        mesh_show_back_face=False)

if __name__ == '__main__':
    # 点云文件路径
    point_cloud_file = 'E:/ABB/AI/yolov9/data/Point_Cloud/point_cloud_1727143874.ply'  # 替换为你的文件路径
    visualize_point_cloud(point_cloud_file)
