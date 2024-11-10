import cv2
import numpy as np

# 读取图像
image_path = "111.png"
image = cv2.imread(image_path)

# 将图像转换为灰度图像
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# circless = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100, param1=50, param2=30, minRadius=20, maxRadius=100)


# 应用高斯模糊来减少噪声
blurred = cv2.GaussianBlur(gray_image, (9, 9), 2)

# 使用霍夫圆检测找到圆形区域
circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100, param1=50, param2=30, minRadius=20,
                           maxRadius=100)

# 确保检测到了圆
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")  # 将检测到的圆取整

    for (x, y, r) in circles:
        # 绘制检测到的圆
        cv2.circle(image, (x, y), r, (0, 255, 0), 4)

        # 计算圆的面积
        circle_area = np.pi * (r ** 2)
        print(f"Detected Circle - Center: ({x}, {y}), Radius: {r}, Area: {circle_area:.2f} pixels^2")

        # 在图像上标记中心
        cv2.circle(image, (x, y), r, (0, 0, 255), 2)

    # 显示结果图像
    cv2.imshow("Detected Circle", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("No circles were detected.")

