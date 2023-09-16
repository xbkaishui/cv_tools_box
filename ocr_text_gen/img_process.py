import pytest
from loguru import logger
import cv2
import io
import os
from pathlib import Path
import numpy as np
import math 
from sahi.slicing import slice_image
from sahi.utils.file import load_json

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


def avg_filter_img():
    input_img_path = '/Users/xbkaishui/opensource/cv_hz/cv_tools_box/ocr_text_gen/output/Image_20230804140423649.jpg'
    input_img_path = '/Users/xbkaishui/opensource/cv_hz/cv_tools_box/ocr_text_gen/output/blurred_image.png'
    image = cv2.imread(input_img_path)
    # 定义滤波器尺寸（即周围像素点的范围）
    kernel_size = 5

    # 执行均值滤波
    blurred_image = cv2.blur(image, (kernel_size, kernel_size))

    # 保存滤波后的图像
    cv2.imwrite("blurred_image.png", blurred_image)


def resize_image_with_ratio(img, resize_h, resize_w):
    ori_h, ori_w = img.shape[:2]  # (h, w, c)
    resize_w = ori_w * resize_h / ori_h
    N = math.ceil(resize_w / 32)
    resize_w = N * 32
    ratio_h = float(resize_h) / ori_h
    ratio_w = float(resize_w) / ori_w
    img = cv2.resize(img, (int(resize_w), int(resize_h)), interpolation=cv2.INTER_AREA)
    return img, [ratio_h, ratio_w]
    

def resize_imgs():
    input_file_dir = '/Users/xbkaishui/opensource/cv_hz/cv_tools_box/ocr_text_gen/output'
    output_dir = "./resize_output"

    input_file_dir = '/Users/xbkaishui/opensource/cv_hz/cv_tools_box/ocr_text_gen/output'
    output_dir = "./backup_resize_output"
    
    os.makedirs(output_dir, exist_ok=True)
    
    img_files = [os.path.join(input_file_dir, f) for f in os.listdir(input_file_dir)]
    logger.info(f"img_files: {img_files}")
    
    for img_file in img_files:
        logger.info("handle file {}", img_file)
        img = cv2.imread(img_file)
        resize_img, _ = resize_image_with_ratio(img, 4096, 800)
        cv2.imwrite(f'{output_dir}/{Path(img_file).stem}.jpg', resize_img)
    

def crop_img():
    input_file_dir = '/Users/xbkaishui/Downloads/backup2'
    output_dir = "./output"
    
    os.makedirs(output_dir, exist_ok=True)
    
    img_files = [os.path.join(input_file_dir, f) for f in os.listdir(input_file_dir)]
    logger.info(f"img_files: {img_files}")
    
    for img_file in img_files:
        logger.info("handle file {}", img_file)
        # do crop img
        img = cv2.imread(img_file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)  # 根据面积大小进行排序
        # 调用 minAreaRect 函数
        rect = cv2.minAreaRect(contours[0])

        # 提取返回的值
        (center, size, angle) = rect
        logger.info(f"center: {center}, size: {size}, angle: {angle}")
        
        # 计算矩形的顶点
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        delta = 500

        # 计算矩形的边界框
        x, y, w, h = cv2.boundingRect(box)
        y = max(y, 0)
        logger.info(f"x: {x}, y: {y}, w: {w}, h: {h}")
        logger.info(f"img.shape: {img.shape}")
        # 从原图像中裁剪出矩形区域
        cropped_region = img[y:y+h, x-delta:x+w + delta]
        file_name = Path(img_file).stem + ".jpg"
        crop_file_path = f'{output_dir}/{file_name}'
        logger.info(f"cropped_region shape: {cropped_region.shape}")
        cv2.imwrite(crop_file_path, cropped_region)
    
    ...
    
def slice_img():
    input_img_path = '/Users/xbkaishui/opensource/cv_hz/cv_tools_box/ocr_text_gen/backup_resize_output/001.jpg'
    slice_image_result = slice_image(
            image=input_img_path,
            coco_annotation_list=None,
            output_file_name="slice",
            output_dir="/tmp/slices",
            slice_height=1024,
            slice_width=800,
            overlap_height_ratio=0,
            overlap_width_ratio=0,
            min_area_ratio=0.1,
            out_ext=".jpg",
            verbose=True,
        )
    logger.info(f"slice_image_result: {slice_image_result}")
    ...
    
def centor_crop():
        from PIL import Image
        import PIL.Image
        PIL.Image.MAX_IMAGE_PIXELS = 933120000
        img_file = '/Users/xbkaishui/Downloads/backup2/001.bmp'
        original_image = Image.open(img_file)

        # 定义目标裁剪尺寸
        target_width = 4200  # 示例宽度
        target_height = original_image.height  # 示例高度

        # 计算裁剪的起始位置（左上角坐标）
        left = (original_image.width - target_width) // 2
        top = (original_image.height - target_height) // 2

        # 计算裁剪的结束位置（右下角坐标）
        right = left + target_width
        bottom = top + target_height

        # 进行居中裁剪操作
        center_cropped_image = original_image.crop((left, top, right, bottom))

        # 保存裁剪后的图像
        center_cropped_image.save('output_center_cropped_image.bmp')
        ...
    
if __name__ == '__main__':
    # crop_img()
    # avg_filter_img()
    # resize_imgs()
    # centor_crop()
    slice_img()