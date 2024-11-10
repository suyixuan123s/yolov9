import cv2
import numpy as np

# 相机内参
fx = 383.11187744140625  # 焦距
fy = 383.11187744140625  # 焦距
cx = 325.05340576171875  # 主点x
cy = 242.58470153808594  # 主点y

# 图像路径
rgb_image_path = r'E:\ABB\AI\yolov9\data\data_realsense\color_image_20241014-112404.jpg'
depth_image_path = r'E:\ABB\AI\yolov9\data\data_realsense\depth_image_20241014-112404.png'

# 加载RGB图像和深度图像
color_image = cv2.imread(rgb_image_path)
depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)  # 深度图像为16位图像，单位为毫米


def get_3d_coordinates(depth_image, center_x, center_y, fx, fy, cx, cy):
    # 获取中心点的深度值
    depth_value = depth_image[int(center_y), int(center_x)]  # 深度图中 (center_x, center_y) 的深度值（单位：毫米）

    if depth_value == 0:  # 深度值为0表示无效，需要处理或跳过
        return None

    # 计算3D坐标
    X = (center_x - cx) * depth_value / fx
    Y = (center_y - cy) * depth_value / fy
    Z = depth_value  # Z即为深度值
    return X, Y, Z


# 从YOLOv9检测结果中获取的中心点坐标
center_points = [(341.0, 183.5), (216.0, 355.0), (352.5, 240.0), (502.0, 178.5), (423.5, 229.5)]
object_names = ["blood_tube", "5ML_sorting_tube_rack", "blood_tube", "5ML_centrifuge_tube", "blood_tube"]

# 打上标签的图像副本
labeled_image = color_image.copy()

# 遍历所有检测到的目标
for idx, (center_x, center_y) in enumerate(center_points):
    # 计算3D坐标
    depth_point = get_3d_coordinates(depth_image, center_x, center_y, fx, fy, cx, cy)

    if depth_point is not None:
        label = f"{object_names[idx]}: X={depth_point[0]:.2f}, Y={depth_point[1]:.2f}, Z={depth_point[2]:.2f}"
        print(f"目标 {idx + 1} 的三维坐标: {label}")

        # 在RGB图像上标记三维坐标
        cv2.putText(labeled_image, label, (int(center_x), int(center_y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        cv2.circle(labeled_image, (int(center_x), int(center_y)), 5, (0, 255, 0), -1)
    else:
        print(f"目标 {idx + 1} 的深度值无效，无法计算三维坐标")

img_label = cv2.imread(r'E:\ABB\AI\yolov9\runs\detect\exp20\color_image_20241014-112404.jpg')




# 显示原始RGB图像和打上标签后的图像
cv2.imshow('Original RGB Image', color_image)
cv2.imshow('Labeled Image with 3D Coordinates', labeled_image)
cv2.imshow('img_label', img_label)


# 保存打上标签的图像
cv2.imwrite(r'E:\ABB\AI\yolov9\runs\detect\labeled_image_20241014-112404.jpg', labeled_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
