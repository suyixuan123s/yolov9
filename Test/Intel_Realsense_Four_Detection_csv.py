import time
import cv2
import numpy as np
import pyrealsense2 as rs
import YOLOv9_Detect_API  # 假设你已经定义了 YOLOv9 的 API
import csv  # 导入CSV模块

# 配置 RealSense 深度相机
pipeline = rs.pipeline()  # 定义流程 pipeline，创建一个管道
config = rs.config()  # 定义配置 config

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # 配置 depth 流
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # 配置 color 流

pipe_profile = pipeline.start(config)  # streaming 流开始
align = rs.align(rs.stream.color)  # 对齐深度流和彩色流


# 获取对齐的彩色和深度图像
def get_aligned_images():
    frames = pipeline.wait_for_frames()  # 等待获取图像帧
    aligned_frames = align.process(frames)  # 对齐深度和颜色帧
    aligned_depth_frame = aligned_frames.get_depth_frame()  # 获取对齐后的深度帧
    aligned_color_frame = aligned_frames.get_color_frame()  # 获取对齐后的彩色帧

    # 获取相机内参
    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics  # 获取深度参数
    color_intrin = aligned_color_frame.profile.as_video_stream_profile().intrinsics  # 获取彩色参数

    # 将 RealSense 图像转换为 NumPy 数组
    img_color = np.asanyarray(aligned_color_frame.get_data())  # RGB 图
    img_depth = np.asanyarray(aligned_depth_frame.get_data())  # 深度图（默认 16 位）

    return color_intrin, depth_intrin, img_color, img_depth, aligned_depth_frame


# 根据像素坐标获取 3D 相机坐标
def get_3d_camera_coordinate(depth_pixel, aligned_depth_frame, depth_intrin):
    x, y = depth_pixel
    height, width = aligned_depth_frame.get_height(), aligned_depth_frame.get_width()

    # 检查像素坐标是否在图像范围内
    if 0 <= x < width and 0 <= y < height:
        dis = aligned_depth_frame.get_distance(x, y)  # 获取该像素点对应的深度
        camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, dis)  # 将像素坐标转换为 3D 坐标
        return dis, camera_coordinate
    else:
        # 如果超出范围，返回一个默认值（例如 NaN）
        return None, [np.nan, np.nan, np.nan]


if __name__ == '__main__':
    # 加载 YOLOv9 模型
    model = YOLOv9_Detect_API.DetectAPI(weights='E:/ABB/AI/yolov9/runs/train/exp19/weights/best.pt')

    # 设置计时器
    start_time = time.time()
    interval = 0  # 检测间隔时间（秒）

    # 初始化存储检测结果的列表，稍后用于保存到 CSV 文件
    detection_data = []

    try:
        while True:
            # 获取对齐的彩色图像和深度图像
            color_intrin, depth_intrin, img_color, img_depth, aligned_depth_frame = get_aligned_images()

            if not img_color.any() or not img_depth.any():
                continue

            # 生成伪彩色深度图用于可视化
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(img_depth, alpha=0.03), cv2.COLORMAP_JET)
            images = np.hstack((img_color, depth_colormap))  # 将彩色图像和深度图像并排显示

            # 检查是否达到间隔时间
            if time.time() - start_time >= interval:
                start_time = time.time()  # 重置计时器

                # 调用 YOLOv9 检测函数
                im0, pred = model.detect([img_color])

                # 定义类别ID到类别名称的映射字典
                class_id_to_name = {
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

                # 处理 YOLOv9 检测结果
                if pred is not None and len(pred):
                    for det in pred:
                        if len(det):
                            for *xyxy, conf, cls in det:
                                # 获取检测框的四个角点的像素坐标
                                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

                                # 计算四个角点的 3D 坐标
                                corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
                                corner_coords = []
                                for corner in corners:
                                    result = get_3d_camera_coordinate(corner, aligned_depth_frame, depth_intrin)
                                    if result is not None:
                                        dis, camera_xyz = result
                                        camera_xyz = np.round(np.array(camera_xyz), 3)  # 保留 3 位小数
                                        camera_xyz = camera_xyz * 1000  # 转换单位为毫米
                                        corner_coords.append(list(camera_xyz))
                                    else:
                                        corner_coords.append([np.nan, np.nan, np.nan])  # 如果角点超出范围，标记为 NaN

                                # 获取物体类别名称
                                class_name = class_id_to_name.get(int(cls), 'Unknown')  # 如果类别ID不在字典中，返回 'Unknown'

                                # 将检测结果保存到 CSV 数据列表中
                                detection_data.append([
                                    class_name,
                                    conf,
                                    corner_coords[0][0], corner_coords[0][1], corner_coords[0][2],
                                    corner_coords[1][0], corner_coords[1][1], corner_coords[1][2],
                                    corner_coords[2][0], corner_coords[2][1], corner_coords[2][2],
                                    corner_coords[3][0], corner_coords[3][1], corner_coords[3][2]
                                ])

                                # 显示检测结果
                                print(f"Detected object: {class_name}, Confidence: {conf:.2f}, 3D Coordinates for corners: {corner_coords}")

                # 显示检测结果
                cv2.namedWindow('detection', flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
                cv2.resizeWindow('detection', 640, 480)
                cv2.imshow('detection', im0)

            # 等待用户按键输入
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:  # 按 'q' 或 'ESC' 退出
                cv2.destroyAllWindows()
                pipeline.stop()
                break
    finally:
        # 停止 RealSense 流
        pipeline.stop()

        # 将检测结果保存到CSV文件
        with open('results/detection_results.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            # 写入表头
            writer.writerow([
                'Class', 'Confidence',
                'Corner1_X(mm)', 'Corner1_Y(mm)', 'Corner1_Z(mm)',
                'Corner2_X(mm)', 'Corner2_Y(mm)', 'Corner2_Z(mm)',
                'Corner3_X(mm)', 'Corner3_Y(mm)', 'Corner3_Z(mm)',
                'Corner4_X(mm)', 'Corner4_Y(mm)', 'Corner4_Z(mm)'
            ])
            # 写入检测数据
            writer.writerows(detection_data)

        print("Detection results have been saved to detection_results.csv")
