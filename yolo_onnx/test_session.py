from onnx_infer import YOLOv8
import time
import os
import psutil
import subprocess

import cv2
from loguru import logger
from PIL import Image
import numpy as np
import gc

def print_gpu_mem():
    command = "nvidia-smi"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    logger.info("gpu memory info {}", result.stdout)

def print_cpu_mem(pid):
     # 获取当前进程的内存信息
    process = psutil.Process(pid)
    memory_info = process.memory_info()
    # 打印内存信息
    print(f"当前进程占用的内存：{memory_info.rss} bytes")

def test_multi_sessions():
    count = 10
    detectors = []
    pid = psutil.Process().pid
    print_cpu_mem(pid)
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = f"{cur_dir}/best.onnx"
    image = Image.open(f"{cur_dir}/test.jpg")
     # RGB -> BGR
    image = cv2.cvtColor(np.array(image, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    for i in range(count):
        detector = YOLOv8(model_path, conf_thres=0.3, iou_thres=0.7)
        detectors.append(detector)
    
    print_gpu_mem()
        
    for i in range(count):
        for detecor in detectors:
            start = time.time()
            detector(image)
            end = time.time()
            print(f"Time: {(end - start)*1000:.2f} ms")
            break
    time.sleep(10)
    print_gpu_mem()
    
    for detector in detectors:
        detector.close()
    del detectors
    logger.info("del detectors")
    print_gpu_mem()
    print_cpu_mem(pid)
    gc.collect()
    logger.info("after gc detectors")
    time.sleep(10)
    print_gpu_mem()
    print_cpu_mem(pid)
    
            


def test_single_session():
    pid = psutil.Process().pid
    logger.info("cur pid is {}".format(pid))

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = f"{cur_dir}/best.onnx"
    detector = YOLOv8(model_path, conf_thres=0.3, iou_thres=0.7)
    image = cv2.imread(f"{cur_dir}/test.jpg")
    print_cpu_mem(pid)
    print_gpu_mem()

    for i in range(10):
        start = time.time()
        detector(image)
        end = time.time()
        logger.info(f"Total Time: {(end - start)*1000:.2f} ms")
        
    # Draw detections
    boxes, scores, class_ids, class_names = detector(image)
    combined_img = detector.draw_detections(image, boxes, scores, class_ids, class_names)
    cv2.imwrite("detection_res.jpg", combined_img)

if __name__ == '__main__':
    test_single_session()
    # test_multi_sessions()
    