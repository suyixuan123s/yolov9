import cv2
import numpy as np

img = cv2.imread('E:/ABB/AI/yolov9/data/Realsence_Data/00001.jpg')
img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_Gray = cv2.cvtColor(img_RGB, cv2.COLOR_BGR2GRAY)

img_Guass = cv2.GaussianBlur(img_Gray, (3, 3), 0)
img_Conny = cv2.Canny(img_Gray, 150, 50)

cv2.imshow('demo1', img_Conny)
cv2.waitKey()
cv2.destroyAllWindows()
