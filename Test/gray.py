from PIL import Image

img = Image.open('data/Realsence_Data/00001.jpg')

gray_img = img.convert("L")

gray_img.save('data/Realsence_Data/example1.jpg')

gray_img.show()



