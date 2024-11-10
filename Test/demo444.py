import time
import cv2
import numpy as np

import pyrealsense2 as rs
import YOLOv9_Detect_API

# 配置Realsence 深度相机

pipeline = rs.pipeline()  # 定义流程 pipeline，创建一个管道
config = rs.config()  # 定义配置config

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # 配置 depth 流
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # 配置 color 流

pipe_profile = pipeline.start(config)  #streaming 流开始
align = rs.align(rs.stream.color) # 对齐深度流和彩色流

# 获取对齐的彩色和深度图像

def get_aligned_images():
    frames = pipeline.wait_for_frames()  # 等待获取图像帧
    aligned_frames = align.process(frames)  # 对齐深度和颜色帧
    aligned_depth_frame = aligned_frames.get_depth_frame()  # 读取对齐后的深度帧
    aligned_color_frame = aligned_frames.get_color_frame()  # 获取对齐后的彩色帧

    # 获取相机的内参

    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics  # 获取深度参数
    color_intrin = aligned_color_frame.profile.as_video_stream_profile().intrinsics  # 获取彩色参数

    # 将 Realsence 图像转换为 Numpy 数组

    img_color = np.asarray(aligned_color_frame.get_data())  # RGB 图
    img_depth = np.asarray(aligned_depth_frame.get_data())  # 深度图 （默认 16 位）

    # 将深度图映射为伪彩色图像以便更好地可视化
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(img_depth, alpha=0.008), cv2.COLORMAP_JET)

    return color_intrin, depth_intrin, img_color, img_depth, aligned_depth_frame


# 根据像素坐标获取 3D 相机的坐标
def get_3d_camera_coordinate(depth_pixel, alighed_depth_frame, depth_intrin):
    x, y = depth_pixel
    dis = alighed_depth_frame.get_distance(x, y)  # 获取该像素点对应的深度
    camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, dis)  # 将像素坐标转换为 3D 坐标
    return dis, camera_coordinate

if __name__== '__main__':
    # 加载 YOLOv9 模型
    model = YOLOv9_Detect_API.DetectAPI(weights = 'E:/ABB/AI/yolov9/runs/train/exp19/weights/best.pt')

    # 设置计时器
    start_time = time.time()
    interval = 0 # 检测间隔时间（秒）

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
                start_time = time.time() # 重置计时器

                # 调用 YOLOv9 检测函数
                im0, pred = model.detect([img_color])

                # 初始化存储检测结果的列表
                camera_xyz_list = []
                class_id_list = []
                xyxy_list = []
                conf_list = []

                # 处理 YOLOv9 检测结果
                if pred is not None and len(pred):
                    for det in pred:
                        if len(det):
                            for *xyxy, conf, cls in det:
                                xyxy_list.append(xyxy)
                                class_id_list.append(int(cls))
                                conf_list.append(float(conf))

                                # 获取检测框中心点的像素坐标
                                ux = int((xyxy[0] + xyxy[2]) / 2)  # 计算 x 中心
                                uy = int((xyxy[1] + xyxy[3]) / 2)  # 计算 y 中心

                                # 获取对应像素的深度uz并转换为 3D 坐标
                                dis = aligned_depth_frame.get_distance(ux, uy)
                                camera_xyz = rs.rs2_deproject_pixel_to_point(depth_intrin, (ux, uy ), dis)
                                camera_xyz = np.round(np.array(camera_xyz), 3)
                                camera_xyz = camera_xyz * 1000 # 转换单位为毫米
                                camera_xyz_list.append(list(camera_xyz))

                                # 在图像上绘制目标的中心点和 3D 坐标
                                cv2.circle(im0, (ux, uy), 4, (255, 255, 255), 5)
                                cv2.putText(im0, str(camera_xyz), (ux + 20 ,uy + 10 ), 0, 1,[255, 255,255], thickness=1, lineType=cv2.LINE_AA)

                # 显示检测结果
                cv2.namedWindow('detection', flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
                cv2.resizeWindow('detection', 640, 480)
                cv2.imshow('detection', im0)

            # 等待用户按键输入
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q') or key == 27: # 按‘q’ 或 ‘ESC’ 退出
                    cv2.destroyAllWindows()
                    pipeline.stop()
                    break
    finally:
         # 停止 Realsence 流
     pipeline.stop()








