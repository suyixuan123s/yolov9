import json

import cv2
import pyrealsense2 as rs
import numpy as np

# 创建一个管道对象
pipeline = rs.pipeline()
config = rs.config()

# 启用深度流和颜色流
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# 启动管道
profile = pipeline.start(config)  # 流程开始

# 创建对齐对象，将深度图与彩色图像对齐
align_to = rs.stream.color  # 与color流对齐
align = rs.align(align_to)


def get_aligned_images():
    # 等待一帧数据以获得相机的内外参
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)  # 获取对齐帧
    aligned_depth_frame = aligned_frames.get_depth_frame()  # 获取对齐帧中的depth帧
    color_frame = aligned_frames.get_color_frame()  # 获取对齐帧中的color帧

    #
    # color_frame = frames.get_color_frame()
    # depth_frame = frames.get_depth_frame()

    # 获取相机的内参
    intrinsic = color_frame.profile.as_video_stream_profile().intrinsics
    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics  # 获取深度参数（像素坐标系转相机坐标系会用到）

    fx = intrinsic.fx  # 焦距x
    fy = intrinsic.fy  # 焦距y
    cx = intrinsic.ppx  # 主点x
    cy = intrinsic.ppy  # 主点y
    distortion = intrinsic.model  # 畸变模型
    coeffs = intrinsic.coeffs  # 畸变系数

    print("Camera Intrinsics:")
    print(f"  Focal Length (fx, fy): ({fx}, {fy})")
    print(f"  Principal Point (cx, cy): ({cx}, {cy})")
    print(f"  Distortion Coefficients: {coeffs}")
    #
    # # 保存内参到本地
    # with open('./intr7insics.json', 'w') as fp:
    #     json.dump(camera_parameters, fp)
    # #######################################################

    # # 获取外参
    # sensor = pipeline.get_active_profile().get_device().first_depth_sensor()
    # extrinsics = sensor.get_extrinsics_to(color_frame.profile)
    #
    # # 提取外参
    # rotation = np.array(extrinsics.rotation).reshape(3, 3)  # 旋转矩阵
    # translation = np.array(extrinsics.translation)  # 位移向量
    #
    # print("Camera Extrinsics:")
    # print(f"  Rotation Matrix:\n{rotation}")
    # print(f"  Translation Vector: {translation}")

    depth_image = np.asanyarray(aligned_depth_frame.get_data())  # 深度图（默认16位）
    depth_image_8bit = cv2.convertScaleAbs(depth_image, alpha=0.03)  # 深度图（8位）
    depth_image_3d = np.dstack((depth_image_8bit, depth_image_8bit, depth_image_8bit))  # 3通道深度图
    color_image = np.asanyarray(color_frame.get_data())  # RGB图

    return intrinsic, depth_intrin, color_image, depth_image, aligned_depth_frame


if __name__ == "__main__":
    while 1:
        intrinsic, depth_intrin, rgb, depth, aligned_depth_frame = get_aligned_images()  # 获取对齐的图像与相机内参
        # 定义需要得到真实三维信息的像素点（x, y)，本例程以中心点为例
        print("============")
        print(aligned_depth_frame)
        x = 320
        y = 240
        dis = aligned_depth_frame.get_distance(x, y)  # （x, y)点的真实深度值
        print("dis: ", dis)
        camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y],
                                                            dis)  # （x, y)点在相机坐标系下的真实值，为一个三维向量。其中camera_coordinate[2]仍为dis，camera_coordinate[0]和camera_coordinate[1]为相机坐标系下的xy真实距离。
        print(camera_coordinate)

        cv2.imshow('RGB image', rgb)  # 显示彩色图像

        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            pipeline.stop()
            break
    cv2.destroyAllWindows()


# import pyrealsense2 as rs
# import numpy as np
#
# # 创建一个管道对象
# pipeline = rs.pipeline()
# config = rs.config()
#
# # 启用深度流和颜色流
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
#
# # 启动管道
#
# profile = pipeline.start(config)  # 流程开始
# align_to = rs.stream.color  # 与color流对齐
# align = rs.align(align_to)
#
# try:
#     # 等待一帧数据以获得相机的内外参
#     frames = pipeline.wait_for_frames()
#     aligned_frames = align.process(frames)  # 获取对齐帧
#     aligned_depth_frame = aligned_frames.get_depth_frame()  # 获取对齐帧中的depth帧
#     color_frame = aligned_frames.get_color_frame()  # 获取对齐帧中的color帧
#
#     #
#     # color_frame = frames.get_color_frame()
#     # depth_frame = frames.get_depth_frame()
#
#     # 获取相机的内参
#
#     intrinsic = depth_frame.profile.as_video_stream_profile().intrinsics
#     fx = intrinsic.fx  # 焦距x
#     fy = intrinsic.fy  # 焦距y
#     cx = intrinsic.ppx  # 主点x
#     cy = intrinsic.ppy  # 主点y
#     distortion = intrinsic.model  # 畸变模型
#     coeffs = intrinsic.coeffs  # 畸变系数
#
#     print("Camera Intrinsics:")
#     print(f"  Focal Length (fx, fy): ({fx}, {fy})")
#     print(f"  Principal Point (cx, cy): ({cx}, {cy})")
#     print(f"  Distortion Coefficients: {coeffs}")
#
#     # 获取外参
#     sensor = pipeline.get_active_profile().get_device().first_depth_sensor()
#     extrinsics = sensor.get_extrinsics_to(color_frame.profile)
#
#     # 提取外参
#     rotation = np.array(extrinsics.rotation).reshape(3, 3)  # 旋转矩阵
#     translation = np.array(extrinsics.translation)  # 位移向量
#
#     print("Camera Extrinsics:")
#     print(f"  Rotation Matrix:\n{rotation}")
#     print(f"  Translation Vector: {translation}")
#
# finally:
#     # 停止管道
#     pipeline.stop()
