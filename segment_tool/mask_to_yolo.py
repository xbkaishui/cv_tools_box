from loguru import logger
import cv2
import numpy as np
import os
import json
from skimage import measure
from pathlib import Path


base_path = '/opt/datasets/bottle/test_datas'


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


def gen_yolo(class_name, image_dir, grund_truth_path):
    logger.info(f'gen yolo for {image_dir}, grund_truth_path {grund_truth_path}')
    label_dir = Path(image_dir).parent / 'labels'
    os.makedirs(label_dir, exist_ok=True)
    # 遍历目录中的文件
    image_idx = 1
    anno_idx = 1
    for filename in os.listdir(image_dir):
        if not 'png' in filename:
            continue
        file_path = os.path.join(image_dir, filename)
        mask_path = os.path.join(grund_truth_path, filename)
        # read shapes
        shapes, height, width = get_shapes(mask_path)

        # 添加标注信息
        labels = []
        for idx, shape in enumerate(shapes):
            seg_points = []
            for point in shape.tolist():
                scale_x = point[0] / width
                scale_y = point[1] / height
                seg_points.extend([str(scale_x), str(scale_y)])
            # 
            print(len(seg_points))
            labels.append('0 ' + ' '.join(seg_points))
        label_file_name = str(label_dir)+'/' + Path(filename).stem + '.txt'
        # write labels to label_file 
        with open(label_file_name, 'w') as f:
            f.write('\n'.join(labels))
        image_idx = image_idx + 1


if __name__ == '__main__':
    test_dir = f'{base_path}'
    dirs = os.listdir(test_dir)
    for dir in dirs:
        if os.path.isfile(dir):
            continue
        if dir.startswith('.'):
            continue
        if dir == 'good':
            continue
        gen_yolo(dir, f'{test_dir}/{dir}/images', f'{test_dir}/{dir}/masks')
        # break