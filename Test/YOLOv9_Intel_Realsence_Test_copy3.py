import os
import numpy as np
import cv2
import pandas as pd
import pyrealsense2 as rs

from YOLOv9_Detect_API import DetectAPI

model = DetectAPI(weights='')

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipe_profile = pipeline.start(config)
align = rs.align(rs.stream.color)

results_folder = 'E:/ABB/AI/yolov9/results'
os.makedirs(results_folder, exist_ok=True)

csv_file_path = os.path.join(results_folder, 'detection_results.csv')

results_data = []

class_name = {
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





def get_aligend_images():
    frames = pipeline.wait_for_frames()
    aligned_frames =align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    depth_intri = depth_frame.profile.as_video_stream_profile().intrinsics
    color_intri = color_frame.profile.as_video_strame_profile().intrinsics
    depth_colormap =cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.07), cv2.COLORMAP_JET)

if __name__ == '__main__':

    try:
        while True:
            depth_intri, depth_frame, color_image =get_aligend_images()
            source = [color_image]

            im0, pred =model.detect(source)

            camera_xyz_list = []
            class_id_list = []
            xyxy_list = []
            conf_list = []

            if pred is not None and len (pred):
                for det in pred:
                    if len(det):
                        for *xyxy, conf, cls in det:
                            xyxy_list.append(xyxy)
                            class_id_list.append(int(cls))
                            conf_list.append(float(conf))


                            class_name = class_name.get(int(cls),"Unknown")

                            # 将Confidence 从张量转换为浮点数
                            confidence = conf.item()

                            print(f"Detected class: {cls}, Confidence: {conf}")

                            # 转换边框坐标为浮点数

                            x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
                            print(f"Bounding box coordinates: [{x1}, {y1}, {x2}, {y2}]")

                            # 获取目标中心点像素坐标并计算3D坐标
                            ux = int((xyxy[0] + xyxy[1]) / 2)
                            uy = int((xyxy[1] + xyxy[3]) / 2)
                            dis_center = depth_frame.get_distance(ux, uy)
                            camera_xyz_center = rs.rs2_deproject_pixel_to_point(depth_intri , (ux, uy), dis_center)
                            camera_xyz_center = np.round(np.array(camera_xyz_center), 3)
                            camera_xyz_center = camera_xyz_center * 1000
                            camera_xyz_center = list(camera_xyz_center)
                            # 打印中心点的三维坐标
                            print(f"Center 3D coordintes: {camera_xyz_center}")

                            # 在图像上绘制中心点和3D坐标

                            cv2.circle(im0, (ux,uy),4, (255, 255, 255), 5) # 标出中心点
                            cv2.putText(im0, str(camera_xyz_center), (ux + 20, uy + 10), 0, 0.5,
                                        [255, 255, 255], thickness=1, lineType=cv2.LINE_AA) # 标出坐标


                            # 计算并绘制和彩色帧的宽度和高度
                            depth_image_width = depth_frame.get_width()
                            depth_image_height = depth_frame.get_height()

                            # 计算并绘制目标框四个角点的三维坐标
                            corners = [(int(xyxy[0]), int(xyxy[1])),
                                       (int(xyxy[2]), int(xyxy[1])),
                                       (int(xyxy[0]), int(xyxy[3])),
                                       (int(xyxy[2]), int(xyxy[3])),
                                       ]

                            corner_coordinates = [None, None, None, None]

                            for i, (x, y)  in enumerate(corners):
                                x = min(max(0, x), depth_image_width -1)
                                y = min(max(0, y), depth_image_height -1)

                                dis_corner = depth_frame.get_distance(x, y)


                                if dis_corner == 0:
                                    print(f" Corner {i + 1} has no depth information.")
                                    continue

                                camera_xyz_corner = rs.rs2_deproject_pixel_to_point(depth_intri, (x, y), dis_corner)

                                camera_xyz_corner = np.round(np.array(camera_xyz_corner), 3)* 1000
                                camera_xyz_corner = list(camera_xyz_corner)

                                # 打印角点的三维坐标
                                print(f"Corner {i+1} 3D coordinates: {camera_xyz_corner}")
                                corner_coordinates[i] = camera_xyz_corner

                                # 在图像上绘制角点坐标
                                cv2.circle(im0, (x, y), 4, (0, 255, 0), -1)
                                cv2.putText(im0, f"C{i+1}: {camera_xyz_corner}" (x+5, y+10), 0,  0.5, [0, 255 ,0], thickness=1,  lineType=cv2.LINE_AA )

                                # 将检测的结果存储到列表中，准备写入 CSV 文件
                                results_data.append({
                                    'Class': class_name,
                                    'Confidence': confidence,
                                    'Bounding Box': [x1, y1, x2, y2],
                                    'Center 3D Coordinates': camera_xyz_center,
                                    'Center 1 Coordinates': corner_coordinates[0],
                                    'Center 2 Coordinates': corner_coordinates[1],
                                    'Center 3 Coordinates': corner_coordinates[2],
                                    'Center 4 Coordinates': corner_coordinates[3]

                                })

                # 显示检测结果
                cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('Realsence', im0)
                key = cv2.waitKey(1)


                if key & 0xFF == ord('q') or key == 27:
                    cv2.destroyAllWindows()
                    pipeline.stop()
                    break

            cv2.namedWindow('Realsence', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Realsence',im0)
            key =cv2.waitKey(1)

            if key & 0xFF == ord('q') or key ==27:
                pipeline.stop()
                break

    finally:

        # 将数据写入 CSV 文件
        if results_data:
            df = pd.DataFrame(results_data)
            df.to_csv(csv_file_path, index=False)
            print(f" Results saved to {csv_file_path}")
        pipeline.stop()











