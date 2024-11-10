# import cv2
# import numpy as np
#
# img = cv2.imread("E:/ABB/AI/yolov9/data/Realsence_Data/00001.jpg")
# img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
# img_Gary = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# img_Gauss = cv2.GaussianBlur(img_Gary, (3, 3), 0 )
#
# img_Canny = cv2.Canny(img_Gauss, 50, 150)
#
# cv2.imshow("Canny", img_Canny)
#
# cv2.waitKey()
# cv2.destroyAllWindows()
#
#
#
#


# import cv2
# import numpy as np
#
# img = cv2.imread("E:/ABB/AI/yolov9/data/Realsence_Data/00001.jpg")
# img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
# img_Gary = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# img_Gaussian = cv2.GaussianBlur(img_Gary, (3, 3), 0)
# img_Canny = cv2.Canny(img_Gaussian, 50, 150)
#
#
# cv2.imshow("Canny", img_Canny)
# cv2.waitKey()
# cv2.destroyAllWindows()


import numpy as np
import cv2

img = cv2.imread("E:/ABB/AI/yolov9/data/Realsense_Data/00001.jpg")
# img2 = cv2.imread("E:/ABB/AI/yolov9/data/Realsence_Data/00002.jpg")
# cv2.imshow('demo2', img)
# cv2.imshow('demo1', img2)
img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_Gray = cv2.cvtColor(img_RGB, cv2.COLOR_BGR2GRAY)
img_GaussianBlur = cv2.GaussianBlur(img_Gray, (3, 3), 0)
img_conny = cv2.Canny(img_GaussianBlur, 150, 50)
# cv2.imshow("Conny", img_conny)
img_Stack = np.hstack((img_RGB, img_Gray))
cv2.imshow("demo", img_Stack)
cv2.waitKey()
cv2.destroyAllWindows()



