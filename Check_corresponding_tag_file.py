import os

images_dir = r'E:\ABB\AI\yolov9\data\images'
labels_dir = r'E:\ABB\AI\yolov9\data\txt'

image_files = set(os.path.splitext(f)[0] for f in os.listdir(images_dir))
label_files = set(os.path.splitext(f)[0] for f in os.listdir(labels_dir))

missing_labels = image_files - label_files
missing_images = label_files - image_files

if missing_labels:
    print(f"以下图像没有对应的标签文件: {missing_labels}")
if missing_images:
    print(f"以下标签文件没有对应的图像: {missing_images}")

# import os
#
# labels_dir = r'E:\ABB\AI\yolov9\data\txt'
# for label_file in os.listdir(labels_dir):
#     label_path = os.path.join(labels_dir, label_file)
#     if os.path.getsize(label_path) == 0:
#         print(f"标签文件 {label_file} 是空的。")
