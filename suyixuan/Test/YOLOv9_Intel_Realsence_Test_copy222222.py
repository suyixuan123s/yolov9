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
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # 初始化深度流
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # 初始化彩色流
pipeline.start(config)  # 启用管道流
align = rs.align(rs.stream.color)  # 将深度图像与彩色图像对齐

# 创建保存目录
results_folder = 'E:/ABB/AI/yolov9/results'
os.makedirs(results_folder, exist_ok=True)  # 如果没有results文件夹就创建一个

# CSV 文件路径
csv_file_path = os.path.join(results_folder, 'detection_results.csv')

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

# 相机内外参
left_camera_matrix = np.array([[385.360809167600, 0, 326.439935124233],
                               [0, 385.348854118456, 241.596362585282],
                               [0, 0, 1]])
left_distortion = np.array([[0.00746036445182610, -0.00358232086751385, -0.000421311503155155,
                             0.00113998387662830, 0]])

right_camera_matrix = np.array([[385.778468661424, 0, 326.309739929908],
                                [0, 385.790097867250, 241.465659082747],
                                [0, 0, 1]])
right_distortion = np.array([[0.00899774214072054, -0.00661251061822051, -0.000714220760572046,
                              0.000771730834556187, 0]])

# 旋转矩阵R和平移向量T
R = np.array([[0.999999809513483, -8.22506698534670e-05, 0.000611725285347005],
              [8.20948311867422e-05, 0.999999964175465, 0.000254773444595701],
              [-0.000611746218718710, -0.000254723176580762, 0.999999780441310]])
T = np.array([-50.0626719811567, 0.0177521779707264, 0.0190872058108044])


def get_aligned_images():
    frames = pipeline.wait_for_frames()  # 等待获取图像帧
    aligned_frames = align.process(frames)  # 获取对齐帧
    depth_frame = aligned_frames.get_depth_frame()  # 获取深度帧
    color_frame = aligned_frames.get_color_frame()  # 获取彩色帧

    depth_image = np.asanyarray(depth_frame.get_data())  # 转换为NumPy数组
    color_image = np.asanyarray(color_frame.get_data())  # 转换为NumPy数组

    return depth_frame, color_image


if __name__ == '__main__':
    try:
        while True:
            depth_frame, color_image = get_aligned_images()  # 获取深度帧和彩色帧

            # YOLOv9模型预测
            im0, pred = model.detect([color_image])  # 使用YOLOv9 API进行检测

            # 检查是否有预测结果
            if pred is not None and len(pred):
                for det in pred:
                    if len(det):
                        for *xyxy, conf, cls in det:
                            # 只识别 'blood_tube'
                            if int(cls) != 0:
                                continue  # 跳过非 'blood_tube' 的目标

                            # 转换边框坐标为浮点数
                            x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])

                            # 获取目标中心点的像素坐标并计算3D坐标
                            ux = int((x1 + x2) / 2)  # 计算x中心
                            uy = int((y1 + y2) / 2)  # 计算y中心
                            dis_center = depth_frame.get_distance(ux, uy)

                            if dis_center == 0:  # 检查深度值是否为零
                                print("深度信息无效，跳过。")
                                continue  # 跳过无效的深度值

                            # 计算相机坐标
                            camera_xyz_center = rs.rs2_deproject_pixel_to_point(
                                depth_frame.profile.as_video_stream_profile().intrinsics, (ux, uy), dis_center)
                            camera_xyz_center = np.round(np.array(camera_xyz_center), 3) * 1000  # 单位转换为毫米

                            # 打印中心点的三维坐标
                            print(
                                f"Detected object: {class_names[int(cls)]}, Center 3D coordinates: {camera_xyz_center.tolist()}")

                            # 在图像上绘制中心点和3D坐标
                            cv2.circle(im0, (ux, uy), 4, (255, 255, 255), 5)  # 标出中心点
                            cv2.putText(im0, str(camera_xyz_center), (ux + 20, uy + 10), 0, 0.5, [225, 255, 255],
                                        thickness=1, lineType=cv2.LINE_AA)  # 标出坐标

            # 显示检测结果
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', im0)

            # 按'q'或'Esc'键退出
            if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
                break

    finally:
        # 清理和释放资源
        cv2.destroyAllWindows()
        pipeline.stop()
