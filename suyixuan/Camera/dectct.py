import os
import cv2
import numpy as np
import pyrealsense2 as rs
from suyixuan.Test.YOLOv9_Detect_API import DetectAPI  # 假设你已经定义了YOLOv9的API
from suyixuan.Test.YOLOv9_Intel_Realsence_Test_copy2 import class_names

# 创建保存目录
save_path = r"/suyixuan/Camera/data"
detect_save_path = r"/suyixuan/Camera/data/detect"
os.makedirs(save_path, exist_ok=True)
os.makedirs(detect_save_path, exist_ok=True)

# 加载YOLOv9模型
model = DetectAPI(weights='E:/ABB/AI/yolov9/runs/train/exp19/weights/best.pt')  # 替换为你的YOLOv9模型路径

# 深度相机配置
pipeline = rs.pipeline()  # 定义流程pipeline，创建一个管道
config = rs.config()  # 定义配置config

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # 初始化摄像头深度流
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipe_profile = pipeline.start(config)  # 启用管段流
align = rs.align(rs.stream.color)  # 用于将深度图像与彩色图像对齐

# 图片计数器
image_counter = 0

def get_aligned_images():
    frames = pipeline.wait_for_frames()  # 等待获取图像帧
    aligned_frames = align.process(frames)  # 获取对齐帧，将深度框与颜色框对齐
    depth_frame = aligned_frames.get_depth_frame()  # 获取深度帧
    color_frame = aligned_frames.get_color_frame()  # 获取彩色帧

    depth_image = np.asanyarray(depth_frame.get_data())  # 将深度帧转换为NumPy数组
    color_image = np.asanyarray(color_frame.get_data())  # 将彩色帧转化为numpy数组

    return color_image, depth_image, depth_frame

if __name__ == '__main__':
    try:
        # 提示信息
        print("程序启动成功。按 'Enter' 键进行图像拍摄和检测任务，按 'Esc' 键退出程序。")

        while True:
            # 实时获取并显示摄像头画面
            color_image, depth_image, depth_frame = get_aligned_images()

            # 在窗口中显示实时彩色图像
            cv2.imshow('RealSense Camera', color_image)

            # 等待用户按键
            key = cv2.waitKey(1)  # 实时刷新显示画面，用户按键

            # 按下 'Esc' 键退出
            if key == 27:  # 27 是 'Esc' 键的 ASCII 值
                print("Exiting...")
                break

            # 按下 'Enter' 键进行拍摄和检测
            if key == 13:  # 13 是 'Enter' 键的 ASCII 值
                # 保存原始彩色图像
                image_counter += 1
                color_image_path = os.path.join(save_path, f"color_image_{image_counter}.jpg")
                cv2.imwrite(color_image_path, color_image)
                print(f"Image saved at: {color_image_path}")

                # YOLOv9 模型检测
                source = [color_image]  # 将彩色图像作为YOLOv9模型的输入
                im0, pred = model.detect(source)  # 使用YOLOv9 API进行检测，返回处理后的图像和预测结果

                # 处理检测结果，标注物体类别和三维坐标，并保存
                if pred is not None and len(pred):
                    for det in pred:  # 处理每个检测结果
                        if len(det):
                            for *xyxy, conf, cls in det:
                                # 转换边框坐标为整数
                                x1, y1, x2, y2 = map(int, xyxy)

                                # 计算物体中心点的坐标
                                center_x = int((x1 + x2) / 2)
                                center_y = int((y1 + y2) / 2)

                                # 获取深度图中的深度值（单位为米）
                                distance = depth_frame.get_distance(center_x, center_y)

                                # 将像素坐标和深度值转换为三维坐标
                                depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
                                xyz = rs.rs2_deproject_pixel_to_point(depth_intrin, [center_x, center_y], distance)
                                xyz = np.round(xyz, 3)  # 保留三位小数

                                # 获取物体的类别名称
                                class_name = class_names.get(int(cls), "Unknown")


                                # 打印类别名称和中心点的三维坐标
                                print(
                                    f"Detected object: {class_name},Confidence: {conf:.2f} ,3D coordinates (X: {xyz[0]}m, Y: {xyz[1]}m, Z: {xyz[2]}m)")

                                # # 打印中心点的三维坐标
                                # print(f"Object center 3D coordinates (X: {xyz[0]}m, Y: {xyz[1]}m, Z: {xyz[2]}m)")

                                # # 绘制检测框和标签
                                # label = f'{cls} {conf:.2f}'

                                # cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色矩形框
                                # cv2.putText(im0, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                                # 显示物体的三维坐标信息
                                text_3d = f"X: {xyz[0]}m Y: {xyz[1]}m Z: {xyz[2]}m"
                                cv2.putText(im0, text_3d, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # 保存带有检测标签和三维坐标的图片
                detect_image_path = os.path.join(detect_save_path, f"detect_image_{image_counter}.jpg")
                cv2.imwrite(detect_image_path, im0)
                print(f"Detection image saved at: {detect_image_path}")

    finally:
        # 停止相机流
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Camera stopped.")
