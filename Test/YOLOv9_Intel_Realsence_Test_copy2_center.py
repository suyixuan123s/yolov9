import os
import cv2
import numpy as np
import pandas as pd
import pyrealsense2 as rs
from YOLOv9_Detect_API import DetectAPI  # 假设你已经定义了YOLOv9的API

# 加载YOLOv9模型
model = DetectAPI(weights='E:/ABB/AI/yolov9/runs/train/exp19/weights/best.pt')  # 替换为你的YOLOv9模型路径

# 深度相机配置
pipeline = rs.pipeline()  # 定义流程pipeline，创建一个管道
config = rs.config()  # 定义配置config
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)  # 初始化摄像头深度流
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

pipe_profile = pipeline.start(config)  # 启用管段流
align = rs.align(rs.stream.color)  # 这个函数用于将深度图像与彩色图像对齐

# 创建保存目录
results_folder = 'E:/ABB/AI/yolov9/results'
os.makedirs(results_folder, exist_ok=True)  # 如果没有results文件夹就创建一个

# CSV 文件路径
csv_file_path = os.path.join(results_folder, 'center_3d_coordinates.csv')

# 初始化存储检测结果的列表
results_data = []

# 类别名称字典，将编号映射为具体的类别名称
class_names = {
    0: 'blood_tube',
    1: '5ML_centrifuge_tube',
    2: '10ML_centrifuge_tube',
    3: '5ML_sorting_tube_rack',
    4: '10ML_sorting_tube_rack',
    5: 'centrifuge_open',
    6: 'centrifuge_close',
    7: 'refrigerator_open',
    8: 'refrigerator_close',
    9: 'operating_desktop',
    10: 'tobe_sorted_tube_rack',
    11: 'dispensing_tube_rack',
    12: 'sorting_tube_rack_base',
    13: 'tube_rack_storage_cabinet'
}

def get_aligned_images():
    frames = pipeline.wait_for_frames()  # 等待获取图像帧
    aligned_frames = align.process(frames)  # 获取对齐帧，将深度框与颜色框对齐
    depth_frame = aligned_frames.get_depth_frame()  # 获取深度帧
    color_frame = aligned_frames.get_color_frame()  # 获取对齐帧中的color帧
    depth_image = np.asanyarray(depth_frame.get_data())  # 将深度帧转换为NumPy数组
    color_image = np.asanyarray(color_frame.get_data())  # 将彩色帧转化为numpy数组
    depth_intri = depth_frame.profile.as_video_stream_profile().intrinsics  # 获取相机内参
    return depth_intri, color_image, depth_frame

if __name__ == '__main__':
    try:
        while True:
            depth_intri, color_image, depth_frame = get_aligned_images()
            source = [color_image]  # 将彩色图像作为YOLOv9模型的输入

            # YOLOv9模型预测
            im0, pred = model.detect(source)  # 使用YOLOv9 API进行检测，返回处理后的图像和预测结果

            # 检查是否有预测结果
            if pred is not None and len(pred):
                for det in pred:  # 处理每个检测结果
                    if len(det):
                        for *xyxy, conf, cls in det:
                            # 获取类别名称
                            cls = int(cls)
                            class_name = class_names.get(cls, "Unknown")

                            # 获取目标中心点的像素坐标并计算3D坐标
                            ux = int((xyxy[0] + xyxy[2]) / 2)  # 计算x中心
                            uy = int((xyxy[1] + xyxy[3]) / 2)  # 计算y中心
                            dis_center = depth_frame.get_distance(ux, uy)

                            # 计算物体中心的3D坐标
                            camera_xyz_center = rs.rs2_deproject_pixel_to_point(depth_intri, (ux, uy), dis_center)
                            camera_xyz_center = np.round(np.array(camera_xyz_center), 3)  # 转成3位小数
                            camera_xyz_center = camera_xyz_center * 1000  # 单位转换为毫米

                            # 打印检测到的物体信息
                            print(f"Detected object: {class_name}")
                            print(f"Confidence: {conf:.2f}")
                            print(f"Center 3D coordinates: [{camera_xyz_center[0]:.2f}, {camera_xyz_center[1]:.2f}, {camera_xyz_center[2]:.2f}]")
                            print("-" * 50)

                            # 将检测结果存储到列表中，准备写入 CSV 文件
                            results_data.append({
                                'Class': class_name,  # 将类别编号转换为类别名称
                                'Confidence': conf.item(),
                                'Center 3D Coordinates': list(camera_xyz_center)
                            })

                            # 在图像上绘制中心点和3D坐标
                            cv2.circle(im0, (ux, uy), 3, (255, 255, 255), -1)  # 标出中心点

                            # 缩小物体类别的字体大小并显示置信度
                            cv2.putText(im0, class_name + f" {conf:.2f}", (ux + 10, uy - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                            # 只显示三维坐标的小数点后两位
                            cv2.putText(im0, f"[{camera_xyz_center[0]:.2f}, {camera_xyz_center[1]:.2f}, {camera_xyz_center[2]:.2f}]",
                                        (ux + 20, uy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [225, 255, 255], 1, cv2.LINE_AA)

            # 显示检测结果
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', im0)
            key = cv2.waitKey(1)  # 等待用户输入

            # 按下esc或'q'关闭图像窗口
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                pipeline.stop()
                break

    finally:
        # 将数据写入 CSV 文件
        if results_data:
            df = pd.DataFrame(results_data)
            df.to_csv(csv_file_path, index=False)
            print(f"Results saved to {csv_file_path}")

        # Stop streaming
        pipeline.stop()
