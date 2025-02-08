import cv2
import os

# Specify the path to your image
image_path = r'E:\ABB\segment-anything\dataset_path\images\00000.jpg'

# Read the image to get its dimensions
image = cv2.imread(image_path)
image_height, image_width, _ = image.shape


detections_path = r'E:\ABB\segment-anything\dataset_path\txt\labels\image9.txt'

# 检查文件是否存在
if not os.path.exists(detections_path):
    # 如果文件夹不存在，则先创建文件夹
    os.makedirs(os.path.dirname(detections_path), exist_ok=True)
    # 创建一个空文件
    with open(detections_path, 'w') as f:
        pass

print(f"File '{detections_path}' is ready.")

bboxes = []
class_ids = []
conf_scores = []

with open(detections_path, 'r') as file:
    for line in file:
        components = line.split()
        class_id = int(components[0])
        confidence = float(components[5])
        cx, cy, w, h = [float(x) for x in components[1:5]]

        # Convert from normalized [0, 1] to image scale
        cx *= image_width
        cy *= image_height
        w *= image_width
        h *= image_height

        # Convert the center x, y, width, and height to xmin, ymin, xmax, ymax
        xmin = cx - w / 2
        ymin = cy - h / 2
        xmax = cx + w / 2
        ymax = cy + h / 2

        class_ids.append(class_id)
        bboxes.append((xmin, ymin, xmax, ymax))
        conf_scores.append(confidence)

# Display the results
for class_id, bbox, conf in zip(class_ids, bboxes, conf_scores):
    print(f'Class ID: {class_id}, Confidence: {conf:.2f}, BBox coordinates: {bbox}')