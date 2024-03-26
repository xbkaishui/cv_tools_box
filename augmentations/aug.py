import albumentations as albu
from PIL import Image
import numpy as np
from loguru import logger
import warnings
import os
import cv2
import pytest

warnings.filterwarnings("ignore", category=UserWarning)

def test_center_crop():
    transform =albu.CenterCrop(1500,1500,p=1) #set height, width, and probability
    image = np.array(Image.open('/Users/xbkaishui/Desktop/73_2.bmp'))
    image = transform(image=image)['image']
    logger.info("img shape {}, img type {}", image.shape, type(image))
    img = Image.fromarray(image)
    img.save('pics/center-crop.jpg')
    
    
def test_horizontal_flip():
    transform =albu.HorizontalFlip(p=1)
    image = np.array(Image.open('/opt/product/cv_tools_box/augmentations/pics/center-crop-orig.jpg'))
    augmented_image = transform(image=image[None])['image']
    logger.info("img shape {}, img type {}", image.shape, type(image))
    img = Image.fromarray(augmented_image)
    img.save('pics/horizontal-flip.jpg')


def test_vertical_flip():
    transform =albu.VerticalFlip(p=1)
    image = np.array(Image.open('pics/test.bmp'))
    augmented_image = transform(image=image)['image']
    logger.info("img shape {}, img type {}", image.shape, type(image))
    img = Image.fromarray(augmented_image)
    img.save('pics/vertical-flip.jpg')


def calc_array_hash(arr):
    import hashlib
    arr_bytes = arr.tobytes()
    md5_hash = hashlib.md5(arr_bytes).hexdigest()
    return md5_hash

def test_exposure():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    import albumentations as A
    import cv2
    transform = A.Compose([
        # A.CenterCrop(width=256, height=256),
        A.RandomGamma(p=.3, gamma_limit=(50, 120))
    ])
    image = cv2.imread(f'{dir_path}/pics/center-crop-orig.jpg')
    for i in range(10):
        transformed_image = transform(image=image)["image"]
        ts_image_hash = calc_array_hash(transformed_image)
        print(ts_image_hash)
        cv2.imwrite(f'{dir_path}/pics/test-crop-{i}.jpg', transformed_image)
    ...
    
def test_random_grip():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    import albumentations as A
    import cv2
    transform = A.Compose([
        # A.CenterCrop(width=256, height=256),
        A.RandomTile(p=1, grid=(2, 2))
    ])
    image = cv2.imread(f'{dir_path}/pics/center-crop-orig.jpg')
    for i in range(10):
        transformed_image = transform(image=image)["image"]
        ts_image_hash = calc_array_hash(transformed_image)
        print(ts_image_hash)
        cv2.imwrite(f'{dir_path}/pics/test-crop-{i}.jpg', transformed_image)
    ...
    
def test_compose_aug():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    import albumentations as A
    import cv2
    transform = A.Compose([
        A.CenterCrop(width=256, height=256),
        A.HorizontalFlip(p=.5),
        # 对比度
        A.Equalize(p=.5),
        A.RandomBrightnessContrast(p=1),

    ])
    image = cv2.imread(f'{dir_path}/pics/center-crop-orig.jpg')
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for i in range(10):
        transformed_image = transform(image=image)["image"]
        ts_image_hash = calc_array_hash(transformed_image)
        print(ts_image_hash)
        cv2.imwrite(f'{dir_path}/pics/test-crop-{i}.jpg', transformed_image)

def test_random_rotation():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    import albumentations as A
    transform = A.Compose([
        A.Rotate(limit=180, p=1, border_mode=cv2.BORDER_CONSTANT)  # 限制旋转角度在 -30 到 +30 之间
    ])
    image = cv2.imread(f'{dir_path}/pics/center-crop-orig.jpg')
    for i in range(20):
        transformed_image = transform(image=image)["image"]
        cv2.imwrite(f'{dir_path}/pics/test-rotation-{i}.jpg', transformed_image)
        
    
    
if __name__ == '__main__':
    # test_center_crop()
    # test_horizontal_flip()
    # test_compose_aug()
    # test_exposure()
    # test_random_grip()
    test_random_rotation()