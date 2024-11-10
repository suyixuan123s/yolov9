import cv2

# 加载图片
image_path = r'E:\ABB\AI\yolov9\runs\detect\exp19\color_image_20241014-112404.jpg'
image = cv2.imread(image_path)

# 转换为灰度图像并应用边缘检测
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150)

# 查找轮廓
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 遍历找到的轮廓
for contour in contours:
    # 获取轮廓的边界框
    x, y, w, h = cv2.boundingRect(contour)

    # 计算中心点
    center_x = x + w // 2
    center_y = y + h // 2

    # 在图片上绘制边界框和中心点
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.circle(image, (center_x, center_y), 5, (255, 0, 0), -1)

    # 输出中心点坐标
    print(f"目标框中心点: ({center_x}, {center_y})")

# 显示图片并保存结果
cv2.imshow("Detected Image", image)
cv2.imwrite('/mnt/data/detected_image_with_centers.jpg', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
