import pyrealsense2 as rs
import numpy as np
import cv2
import os

# 确保保存路径存在
save_path = r"E:\ABB\AI\yolov9\Camera\data"
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 配置RealSense相机流
pipeline = rs.pipeline()
config = rs.config()

# 配置彩色流和深度流
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# 启动RealSense相机
pipeline.start(config)

# 设置对齐方式，将深度帧对齐到彩色帧
align_to = rs.stream.color
align = rs.align(align_to)

# 图片保存计数
image_counter = 0

try:
    while True:
        # 等待一帧图像
        frames = pipeline.wait_for_frames()

        # 对齐深度帧到彩色帧
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()  # 对齐后的深度帧
        color_frame = aligned_frames.get_color_frame()  # 彩色帧

        # 检查是否成功获取帧
        if not aligned_depth_frame or not color_frame:
            continue

        # 将彩色帧转换为numpy数组
        color_image = np.asanyarray(color_frame.get_data())

        # 显示彩色图像
        cv2.imshow('RealSense Color Image', color_image)

        # 按 's' 键保存图片到指定目录
        key = cv2.waitKey(1)
        if key & 0xFF == ord('s'):
            image_counter += 1
            file_name = os.path.join(save_path, f"color_image_{image_counter}.jpg")
            cv2.imwrite(file_name, color_image)
            print(f"保存图片: {file_name}")

        # 按 'q' 键退出
        elif key & 0xFF == ord('q'):
            break

finally:
    # 停止相机
    pipeline.stop()
    cv2.destroyAllWindows()
