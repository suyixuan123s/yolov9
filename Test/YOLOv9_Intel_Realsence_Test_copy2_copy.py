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

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # 初始化摄像头深度流
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipe_profile = pipeline.start(config)  # 启用管段流
align = rs.align(rs.stream.color)  # 这个函数用于将深度图像与彩色图像对齐

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


def get_aligned_images():
    frames = pipeline.wait_for_frames()  # 等待获取图像帧

    depth_frame_LL = frames.get_depth_frame()
    color_frame_ll = frames.get_color_frame()


    aligned_frames = align.process(frames)  # 获取对齐帧，将深度框与颜色框对齐

    depth_frame = aligned_frames.get_depth_frame()  # 获取深度帧
    color_frame = aligned_frames.get_color_frame()  # 获取对齐帧中的的color帧

    depth_image = np.asanyarray(depth_frame.get_data())  # 将深度帧转换为NumPy数组
    color_image = np.asanyarray(color_frame.get_data())  # 将彩色帧转化为numpy数组

    # 获取相机内参
    depth_intri = depth_frame.profile.as_video_stream_profile().intrinsics

    depth_frame_LL_intrinsic = depth_frame_LL.profile.as_video_stream_profile().intrinsics

    color_frame_ll_intri = color_frame_ll.profile.as_video_stream_profile().intrinsics

    print("----------------")
    print(f"depth_intri:{depth_intri}")
    color_intri = color_frame.profile.as_video_stream_profile().intrinsics
    print("----------------")
    print(f"color_intri:{color_intri}")

    print("----------------")
    print(f"depth_frame_LL_intrinsic:{depth_frame_LL_intrinsic}")

    print("----------------")
    print(f"color_frame_LL_intrinsic:{color_frame_ll_intri}")

    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.07), cv2.COLORMAP_JET)

    # return depth_intri, color_intri, color_image, depth_image, depth_frame, color_frame
    return depth_intri, color_intri, color_image, depth_image, depth_frame, color_frame, depth_frame_LL_intrinsic, depth_frame_LL


if __name__ == '__main__':
    try:
        while True:
            depth_intri, color_intri, color_image, depth_image, depth_frame, color_frame, depth_frame_LL_intrinsic, depth_frame_LL = get_aligned_images()

            # depth_intri, depth_frame, color_image = get_aligned_images()  # 获取深度帧和彩色帧
            source = [color_image]  # 将彩色图像作为YOLOv9模型的输入

            # YOLOv9模型预测
            im0, pred = model.detect(source)  # 使用YOLOv9 API进行检测，返回处理后的图像和预测结果

            # 初始化空列表来存储结果
            camera_xyz_list = []
            class_id_list = []
            xyxy_list = []
            conf_list = []

            # 检查是否有预测结果
            if pred is not None and len(pred):
                for det in pred:  # 处理每个检测结果1
                    if len(det):
                        for *xyxy, conf, cls in det:
                            xyxy_list.append(xyxy)
                            class_id_list.append(int(cls))
                            conf_list.append(float(conf))
                            #
                            # # 获取类别名称
                            # class_name = class_names.get(int(cls), "Unknown")

                            # cls 是类别编号，需要将其映射为类别名称

                            cls = int(cls)  # 确保类别编号是整数
                            class_name = class_names.get(cls, "Unknown")  # 从字典中获取类别名称，如果找不到则返回 "Unknown"

                            # # 只识别'blood_tube'
                            # if cls == 0:
                            #     class_name = class_names[cls]
                            # else:
                            #     class_name = "Unknown"

                            # 将Confidence从张量转换为浮点数
                            confidence = conf.item()

                            # 打印检测结果
                            print(f"Detected class: {class_name}, Confidence: {conf}")

                            # 转换边框坐标为浮点数
                            x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
                            print(f"Bounding box coordinates: [{x1}, {y1}, {x2}, {y2}]")

                            # 获取目标中心点的像素坐标并计算3D坐标
                            ux = int((xyxy[0] + xyxy[2]) / 2)  # 计算x中心
                            uy = int((xyxy[1] + xyxy[3]) / 2)  # 计算y中心
                            dis_center = depth_frame.get_distance(ux, uy)

                            camera_xyz_center = rs.rs2_deproject_pixel_to_point(depth_intri, (ux, uy), dis_center)
                            camera_xyz_center = np.round(np.array(camera_xyz_center), 3)  # 转成3位小数

                            camera_xyz_center = camera_xyz_center * 1000  # 将单位转换为毫米
                            camera_xyz_center = list(camera_xyz_center)

                            # 打印中心点的三维坐标
                            print(f"Center 3D coordinates: {camera_xyz_center}")

                            dis_center1 = depth_frame_LL.get_distance(ux, uy)

                            camera_xyz_center1 = rs.rs2_deproject_pixel_to_point(depth_frame_LL_intrinsic, (ux, uy),
                                                                                 dis_center1)
                            camera_xyz_center1 = np.round(np.array(camera_xyz_center1), 3)  # 转成3位小数

                            camera_xyz_center1 = camera_xyz_center1 * 1000  # 将单位转换为毫米
                            camera_xyz_center1 = list(camera_xyz_center1)

                            # 打印中心点的三维坐标
                            print(f"Center 3D coordinates1111: {camera_xyz_center1}")

                            # 在图像上绘制中心点和3D坐标
                            cv2.circle(im0, (ux, uy), 3, (255, 255, 255), -1)  # 标出中心点
                            cv2.putText(im0, str(camera_xyz_center), (ux + 20, uy + 10), 0, 0.5,
                                        [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)  # 标出坐标

                            # 获取深度帧和彩色帧的宽度和高度
                            depth_image_width = depth_frame.get_width()
                            depth_image_height = depth_frame.get_height()

                            # 计算并绘制目标框四个角点的三维坐标
                            corners = [(int(xyxy[0]), int(xyxy[1])),  # 左上角
                                       (int(xyxy[2]), int(xyxy[1])),  # 右上角
                                       (int(xyxy[0]), int(xyxy[3])),  # 左下角
                                       (int(xyxy[2]), int(xyxy[3]))]  # 右下角

                            # corner_coordinates = [None, None, None, None]  # 确保四个角点都有位置，即使没有深度信息

                            corner_coordinates = [(0, 0), (0, 0), (0, 0), (0, 0)]

                            for i, (x, y) in enumerate(corners):

                                # 确保 x 和 y 在图像范围内
                                x = min(max(0, x), depth_image_width - 1)
                                y = min(max(0, y), depth_image_height - 1)

                                dis_corner = depth_frame.get_distance(x, y)
                                if dis_corner == 0:  # 检查深度值是否为零
                                    print(f"Corner {i + 1} has no depth information.")
                                    continue  # 跳过无效的深度值

                                camera_xyz_corner = rs.rs2_deproject_pixel_to_point(depth_intri, (x, y), dis_corner)
                                camera_xyz_corner = np.round(np.array(camera_xyz_corner), 3)  # 转成3位小数
                                camera_xyz_corner = camera_xyz_corner * 1000  # 单位转换为毫米
                                camera_xyz_corner = list(camera_xyz_corner)

                                # 打印角点的三维坐标
                                print(f"Corner {i + 1} 3D coordinates: {camera_xyz_corner}")
                                corner_coordinates[i] = camera_xyz_corner  # 存储角点三维坐标

                                # 在图像上绘制角点坐标
                                cv2.circle(im0, (x, y), 4, (0, 255, 0), -1)  # 标出角点
                                cv2.putText(im0, f"C{i + 1}:{camera_xyz_corner}", (x + 5, y + 10), 0, 0.5,
                                            [0, 255, 0], thickness=1, lineType=cv2.LINE_AA)

                            # 将检测结果存储到列表中，准备写入 CSV 文件
                            results_data.append({
                                # 'Class': cls,
                                'Class': class_name,  # 将类别编号转换为类别名称
                                'Confidence': confidence,
                                'Bounding Box': [x1, y1, x2, y2],
                                'Center 3D Coordinates': camera_xyz_center,
                                'Corner 1 Coordinates': corner_coordinates[0],
                                'Corner 2 Coordinates': corner_coordinates[1],
                                'Corner 3 Coordinates': corner_coordinates[2],
                                'Corner 4 Coordinates': corner_coordinates[3]
                            })

            # 显示检测结果
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', im0)
            key = cv2.waitKey(1)  # 等待用户输入

            # Press esc or 'q' to close the image window
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
