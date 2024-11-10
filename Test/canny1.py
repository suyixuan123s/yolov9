import cv2
import numpy as np

image_path = "111.png"
image = cv2.imread(image_path)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_gauss = cv2.GaussianBlur(gray_image, (9, 9), 2)

# 使用霍夫圆检测找到圆形的区域 使用霍夫圆找到圆形的取区域
circles = cv2.HoughCircles(image_gauss, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100, param1=50, param2=30, minRadius=20,
                           maxRadius=100)

if circles is not None:
    circles = np.round(circles[0, :]).astype("int")  # 将检测到的圆取整

    for (x, y, r) in circles:
        # 绘制检测到的圆
        cv2.circle(image, (x, y), r, (255, 0, 0), 2)

        # 计算圆的面积
        circles_area = np.pi * (r ** 2)
        print(f"Detected Circle - Center: ({x}, {y}), Radius: {r}, Area: {circles_area:.2f} pixels^2")

    # 显示结果图像
    cv2.imshow("Detected Circle", image)
    cv2.waitKey()
    cv2.destroyAllWindows()


else:
    print("No circles were detected.")
