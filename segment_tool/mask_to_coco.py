from loguru import logger
import cv2
import numpy as np
import os
import json
from skimage import measure
from pathlib import Path


base_path = '/opt/datasets/bottle'
grund_truth_path = 'ground_truth'
test_path = 'test'


def get_shapes(image_path):
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # _, thresh = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
    # contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = measure.find_contours(gray_image, 0.5)
    shapes = []
    # 遍历每个轮廓并进行多边形近似
    for contour in contours:
        # epsilon = 0.001 * cv2.arcLength(contour, True)  # 调整epsilon以获得适当的多边形精度
        # approx = cv2.approxPolyDP(contour, epsilon, True)
        # approx_points = approx.reshape((-1, 2))
        polygon = measure.approximate_polygon(contour, tolerance=2.0)
        approx_points = polygon[:, [1, 0]]
        if len(approx_points) > 4:
            shapes.append(approx_points)
    return shapes, height, width


def gen_coco(class_name, image_dir, grund_truth_path):
    logger.info(f'gen coco for {image_dir}, grund_truth_path {grund_truth_path}')
    # 创建COCO数据结构
    coco_data = {
        "info": {"description": "Your COCO Dataset Description", "version": "1.0", "year": 2023, "contributor": "Your Name", "date_created": "2023-10-18"},
        "images": [],
        "annotations": [],
        "categories": []
    }

    # 添加类别信息
    category = {"id": 1, "name": class_name, "supercategory": ""}
    coco_data["categories"].append(category)


    # 遍历目录中的文件
    image_idx = 1
    anno_idx = 1
    for filename in os.listdir(image_dir):
        if not 'png' in filename:
            continue
        file_path = os.path.join(image_dir, filename)
        mask_path = f'{grund_truth_path}/{Path(filename).stem}_mask.png'
        # read shapes
        shapes, height, width = get_shapes(mask_path)
        # 添加图像信息
        image = {
            "id": image_idx,
            "file_name": filename,
            "width": width,
            "height": height
        }
        coco_data["images"].append(image)

        # 添加标注信息
        for idx, shape in enumerate(shapes):
            seg_points = []
            for point in shape.tolist():
                seg_points.extend(point)
            annotation = {
                "id": anno_idx,
                "image_id": image_idx,  # 对应图像的ID
                "category_id": 1,  # 类别ID
                "iscrowd": 0,
                "area": 0.0,  # 根据数据计算面积
                "bbox": [],  # 根据数据计算边界框
                "segmentation": [seg_points]
            }
            anno_idx = anno_idx + 1
            coco_data["annotations"].append(annotation)
        
        image_idx = image_idx + 1


    # 保存为COCO格式的JSON文件
    with open(f'{image_dir}/_annotations.coco.json', 'w') as output_file:
        json.dump(coco_data, output_file)

    

if __name__ == '__main__':
    test_dir = f'{base_path}/{test_path}'
    dirs = os.listdir(test_dir)
    for dir in dirs:
        if os.path.isfile(dir):
            continue
        if dir.startswith('.'):
            continue
        if dir == 'good':
            continue
        gen_coco(dir, f'{test_dir}/{dir}', f'{base_path}/{grund_truth_path}/{dir}')
        # break