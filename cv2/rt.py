import cv2
import numpy as np

# 读取图像
image_path = '3_1.bmp'  # 替换为您的图像路径
original_image = cv2.imread(image_path)

# 获取图像尺寸
height, width = original_image.shape[:2]

# 定义旋转角度
angle = 15

# 计算旋转矩阵
rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)

# 进行旋转
rotated_image = cv2.warpAffine(original_image, rotation_matrix, (width, height))

# 保存旋转后的图像
output_path = '3_1_rt.bmp'  # 替换为您想要保存的路径
cv2.imwrite(output_path, rotated_image)

# 显示原始图像和旋转后的图像（可选）
cv2.imshow('Original Image', original_image)
cv2.imshow('Rotated Image', rotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
