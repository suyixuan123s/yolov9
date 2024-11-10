import os

import cv2
import numpy as np
import pyrealsense2 as rs
import open3d as o3d
from datetime import datetime  # 使用datetime来生成时间戳

# 创建保存目录
point_cloud_folder = 'E:/ABB/AI/yolov9/data/Point_Cloud'
os.makedirs(point_cloud_folder, exist_ok=True)

# 配置深度相机
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)
align = rs.align(rs.stream.color)

def capture_point_cloud():
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    if not depth_frame or not color_frame:
        return None

    # 获取深度图像数据
    depth_image = np.array(depth_frame.get_data())
    color_image = np.array(color_frame.get_data())

    # 获取相机内参
    intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

    # 生成点云
    width, height = depth_image.shape[1], depth_image.shape[0]
    i, j = np.meshgrid(np.arange(width), np.arange(height))
    z = depth_image / 1000.0  # 将深度值从毫米转换为米
    x = (j - intrinsics.ppy) * z / intrinsics.fy
    y = (i - intrinsics.ppx) * z / intrinsics.fx

    # 组合为点云
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    colors = color_image.reshape(-1, 3) / 255.0  # 颜色归一化

    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd

if __name__ == '__main__':
    try:
        while True:
            pcd = capture_point_cloud()
            if pcd is not None:
                # 获取当前时间戳并生成点云文件名
                timestamp = int(datetime.now().timestamp())
                point_cloud_path = os.path.join(point_cloud_folder, f'point_cloud_{timestamp}.ply')

                # 保存点云
                o3d.io.write_point_cloud(point_cloud_path, pcd)
                print(f'保存点云到 {point_cloud_path}')

            # 显示点云（可选）
            o3d.visualization.draw_geometries([pcd])

            # 等待1秒以获取下一帧
            key = cv2.waitKey(1000)
            if key & 0xFF == ord('q') or key == 27:
                break
    finally:
        # 停止管道流
        pipeline.stop()
