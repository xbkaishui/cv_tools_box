from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPrompt
import cv2
import time
from loguru import logger

# Define image path and inference device
IMAGE_PATH = 'ultralytics/assets/bus.jpg'
DEVICE = 'cuda'

# Create a FastSAM model
# model = FastSAM('FastSAM-x.pt')  # or FastSAM-x.pt
model = FastSAM('FastSAM-x.engine')

IMAGE_PATH = "/opt/product/DINO/weights/hl.jpg"
# Run inference on an image
everything_results = model(IMAGE_PATH,
                           device=DEVICE,
                           retina_masks=True,
                           imgsz=1024,
                           conf=0.01,
                           iou=0.7)

prompt_process = FastSAMPrompt(IMAGE_PATH, everything_results, device=DEVICE)

# Everything prompt
# ann = prompt_process.everything_prompt()

# Bbox default shape [0,0,0,0] -> [x1,y1,x2,y2]
ann = prompt_process.box_prompt(bbox=[318, 25, 414, 117])

# # Text prompt
# ann = prompt_process.text_prompt(text='a photo of a dog')

cv2.imwrite("test.jpg", ann.squeeze() * 255)
# Point prompt
# points default [[0,0]] [[x1,y1],[x2,y2]]
# point_label default [0] [1,0] 0:background, 1:foreground
# ann = prompt_process.point_prompt(points=[[200, 200]], pointlabel=[1])
prompt_process.plot(annotations=ann, output='./', withContours=False, retina=True, bbox=[318, 25, 414, 117], mask_random_color=False, better_quality=False)


count = 100
start = time.time()
for i in range(count):
    everything_results = model(IMAGE_PATH,
                           device=DEVICE,
                           retina_masks=True,
                           imgsz=1024,
                           conf=0.1,
                           iou=0.7)
    prompt_process = FastSAMPrompt(IMAGE_PATH, everything_results, device=DEVICE)
    ann = prompt_process.box_prompt(bbox=[318, 25, 414, 117])
end = time.time()
logger.info("time: {} ms", end - start)
