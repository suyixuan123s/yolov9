import os
import cv2
import numpy as np
import pyrealsense2 as rs
import time

# 创建保存目录
image_folder = 'E:/ABB/AI/yolov9/data/dataset/images1'
os.makedirs(image_folder, exist_ok=True)

# 配置深度相机
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)
align = rs.align(rs.stream.color)

if __name__ == '__main__':
    try:
        while True:
            # 获取对齐的图像帧
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # 将图像数据转换为NumPy数组
            color_image = np.array(color_frame.get_data())
            depth_image = np.array(depth_frame.get_data())

            # 获取当前时间戳并生成图像文件名
            timestamp = int(time.time())
            color_image_path = os.path.join(image_folder, f'color_image_{timestamp}.png')
            depth_image_path = os.path.join(image_folder, f'depth_image_{timestamp}.png')

            # 保存图像
            cv2.imwrite(color_image_path, color_image)
            cv2.imwrite(depth_image_path, depth_image)
            print(f'保存彩色图像到 {color_image_path}')
            print(f'保存深度图像到 {depth_image_path}')

            # 显示彩色图像
            cv2.imshow('RealSense Color', color_image)

            # 等待1秒以获取下一帧
            key = cv2.waitKey(1000)
            if key & 0xFF == ord('q') or key == 27:
                break
    finally:
        # 停止管道流
        pipeline.stop()
        cv2.destroyAllWindows()
