import os
import cv2
import numpy as np
import pandas as pd
import pyrealsense2 as rs
from YOLOv9_Detect_API import DetectAPI  # 假设你已经定义了YOLOv9的API

# 加载YOLOv9模型
model = DetectAPI(weights='E:/ABB/AI/yolov9/runs/train/exp19/weights/best.pt')

# 创建保存目录
results_folder = 'E:/ABB/AI/yolov9/results'
os.makedirs(results_folder, exist_ok=True)
csv_file_path = os.path.join(results_folder, 'detection_results.csv')
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


# 拍摄并保存图片的函数
def capture_image(pipeline):
    align = rs.align(rs.stream.color)
    pipeline.start()
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()

    color_image = np.array(color_frame.get_data())
    cv2.imwrite('captured_image.png', color_image)  # 保存拍摄的图片
    pipeline.stop()
    return 'captured_image.png'


# 检测图片中的目标
def detect_image(image_path):
    image = cv2.imread(image_path)
    source = [image]
    im0, pred = model.detect(source)

    # 读取深度信息
    depth_intri = None  # 可以根据需要加载深度信息

    if pred is not None and len(pred):
        for det in pred:
            if len(det):
                for *xyxy, conf, cls in det:
                    class_name = class_names.get(int(cls), "Unknown")
                    confidence = conf.item()
                    x1, y1, x2, y2 = [float(x) for x in xyxy]

                    # 打印和保存检测结果
                    print(f"Detected class: {class_name}, Confidence: {confidence}")

                    # 在图像上绘制目标
                    cv2.rectangle(im0, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(im0, f"{class_name} {confidence:.2f}", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    results_data.append({
                        'Class': class_name,
                        'Confidence': confidence,
                        'Bounding Box': [x1, y1, x2, y2]
                    })

    # 显示结果
    cv2.imshow('Detected Image', im0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # 深度相机配置
    pipeline = rs.pipeline()
    captured_image_path = capture_image(pipeline)  # 捕获并保存图片
    detect_image(captured_image_path)  # 对拍摄的图片进行检测

    # 将数据写入 CSV 文件
    if results_data:
        df = pd.DataFrame(results_data)
        df.to_csv(csv_file_path, index=False)
        print(f"Results saved to {csv_file_path}")
