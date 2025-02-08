import os
import cv2
import numpy as np
import pandas as pd
import pyrealsense2 as rs
from YOLOv9_Detect_API import DetectAPI

# 加载YOLOv9模型
model = DetectAPI(weights='E:/ABB/AI/yolov9/runs/train/exp19/weights/best.pt')

# 深度相机配置
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

pipe_profile = pipeline.start(config)
align = rs.align(rs.stream.color)

# 创建保存目录
results_folder = 'E:/ABB/AI/yolov9/results'
os.makedirs(results_folder, exist_ok=True)

# CSV 文件路径
csv_file_path = os.path.join(results_folder, 'detection_results.csv')

# 初始化存储检测结果的列表
results_data = []

# 类别名称字典
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

# 手动调节的固定参数，用于相机坐标系到仿真坐标系的转换
alpha, beta, gamma = -148.0, -0.4, -178.0
tx, ty, tz = 0.525, 0.76, 1.25


def get_transformation_matrix(alpha_deg, beta_deg, gamma_deg, tx, ty, tz):
    alpha = np.radians(alpha_deg)
    beta = np.radians(beta_deg)
    gamma = np.radians(gamma_deg)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(alpha), -np.sin(alpha)],
        [0, np.sin(alpha), np.cos(alpha)]
    ])
    Ry = np.array([
        [np.cos(beta), 0, np.sin(beta)],
        [0, 1, 0],
        [-np.sin(beta), 0, np.cos(beta)]
    ])
    Rz = np.array([
        [np.cos(gamma), -np.sin(gamma), 0],
        [np.sin(gamma), np.cos(gamma), 0],
        [0, 0, 1]
    ])
    rotation_matrix = Rz @ Ry @ Rx
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = [tx, ty, tz]

    return transformation_matrix


transformation_matrix = get_transformation_matrix(alpha, beta, gamma, tx, ty, tz)


def transform_point_to_simulation(point):
    homogeneous_point = np.append(point, 1)
    transformed_point = transformation_matrix @ homogeneous_point
    return transformed_point[:3]


def get_aligned_images():
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    depth_intri = depth_frame.profile.as_video_stream_profile().intrinsics
    color_intri = color_frame.profile.as_video_stream_profile().intrinsics

    return depth_intri, color_intri, color_image, depth_image, depth_frame, color_frame


if __name__ == '__main__':
    try:
        while True:
            depth_intri, color_intri, color_image, depth_image, depth_frame, color_frame = get_aligned_images()
            source = [color_image]
            im0, pred = model.detect(source)

            if pred is not None and len(pred):
                for det in pred:
                    if len(det):
                        for *xyxy, conf, cls in det:
                            class_name = class_names.get(int(cls), "Unknown")
                            ux, uy = int((xyxy[0] + xyxy[2]) / 2), int((xyxy[1] + xyxy[3]) / 2)
                            dis_center = depth_frame.get_distance(ux, uy)

                            if dis_center == 0:
                                print("No depth information at center.")
                                continue

                            camera_xyz_center = rs.rs2_deproject_pixel_to_point(depth_intri, (ux, uy), dis_center)
                            sim_xyz_center = transform_point_to_simulation(camera_xyz_center)

                            print(f"Camera XYZ (mm): {camera_xyz_center}, Simulation XYZ (mm): {sim_xyz_center}")

                            corners = [
                                (int(xyxy[0]), int(xyxy[1])),
                                (int(xyxy[2]), int(xyxy[1])),
                                (int(xyxy[0]), int(xyxy[3])),
                                (int(xyxy[2]), int(xyxy[3]))
                            ]

                            corner_coordinates = []
                            for i, (x, y) in enumerate(corners):
                                x, y = min(max(0, x), depth_frame.get_width() - 1), min(max(0, y),
                                                                                        depth_frame.get_height() - 1)
                                dis_corner = depth_frame.get_distance(x, y)

                                if dis_corner == 0:
                                    print(f"No depth at corner {i + 1}")
                                    corner_coordinates.append(None)
                                    continue

                                camera_xyz_corner = rs.rs2_deproject_pixel_to_point(depth_intri, (x, y), dis_corner)
                                sim_xyz_corner = transform_point_to_simulation(camera_xyz_corner)
                                corner_coordinates.append(sim_xyz_corner)
                                print(
                                    f"Corner {i + 1} Camera XYZ (mm): {camera_xyz_corner}, Simulation XYZ (mm): {sim_xyz_corner}")

                            cv2.circle(im0, (ux, uy), 3, (255, 255, 255), -1)
                            cv2.putText(im0, f"{sim_xyz_center}", (ux + 20, uy + 10), 0, 0.5, [225, 255, 255],
                                        thickness=1, lineType=cv2.LINE_AA)

                            results_data.append({
                                'Class': class_name,
                                'Confidence': float(conf),
                                'Bounding Box': [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])],
                                'Center Simulation XYZ': sim_xyz_center,
                                'Corner 1 Simulation XYZ': corner_coordinates[0],
                                'Corner 2 Simulation XYZ': corner_coordinates[1],
                                'Corner 3 Simulation XYZ': corner_coordinates[2],
                                'Corner 4 Simulation XYZ': corner_coordinates[3]
                            })

            cv2.imshow('RealSense', im0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        if results_data:
            df = pd.DataFrame(results_data)
            df.to_csv(csv_file_path, index=False)
            print(f"Results saved to {csv_file_path}")

        pipeline.stop()
