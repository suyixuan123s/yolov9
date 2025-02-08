import cv2
import numpy as np
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

def get_aligned_images():
    frames = pipeline.wait_for_frames()  # 等待获取图像帧
    aligned_frames = align.process(frames)  # 获取对齐帧，将深度框与颜色框对齐
    depth_frame = aligned_frames.get_depth_frame()  # 获取深度帧
    color_frame = aligned_frames.get_color_frame()  # 获取对齐帧中的的color帧

    depth_image = np.asanyarray(depth_frame.get_data())  # 将深度帧转换为NumPy数组
    color_image = np.asanyarray(color_frame.get_data())  # 将彩色帧转化为numpy数组

    # 获取相机内参
    depth_intri = depth_frame.profile.as_video_stream_profile().intrinsics
    color_intri = color_frame.profile.as_video_stream_profile().intrinsics

    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.07), cv2.COLORMAP_JET)
    return depth_intri, depth_frame, color_image


if __name__ == '__main__':
    try:
        while True:
            depth_intri, depth_frame, color_image = get_aligned_images()  # 获取深度帧和彩色帧
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
                for det in pred:  # 处理每个检测结果
                    if len(det):
                        for *xyxy, conf, cls in det:
                            xyxy_list.append(xyxy)
                            class_id_list.append(int(cls))
                            conf_list.append(float(conf))

                            # 获取目标中心点的像素坐标并计算3D坐标
                            ux = int((xyxy[0] + xyxy[2]) / 2)  # 计算x中心
                            uy = int((xyxy[1] + xyxy[3]) / 2)  # 计算y中心
                            dis = depth_frame.get_distance(ux, uy)
                            camera_xyz = rs.rs2_deproject_pixel_to_point(depth_intri, (ux, uy), dis)
                            camera_xyz = np.round(np.array(camera_xyz), 3)  # 转成3位小数
                            camera_xyz = camera_xyz * 1000  # 将单位转换为毫米
                            camera_xyz = list(camera_xyz)

                            # 在图像上绘制中心点和3D坐标
                            cv2.circle(im0, (ux, uy), 4, (255, 255, 255), 5)  # 标出中心点
                            cv2.putText(im0, str(camera_xyz), (ux + 20, uy + 10), 0, 0.5,
                                        [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)  # 标出坐标

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
        # Stop streaming
        pipeline.stop()
