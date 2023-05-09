# !pip install git+https://github.com/Lednik7/CLIP-ONNX.git

import os
# set to single omp thread
# os.environ["OMP_NUM_THREADS"] = "1"

from numpy import linalg as LA
import onnxruntime
import clip
from PIL import Image
import numpy as np
from loguru import logger
from clip_onnx import clip_onnx, attention
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

logger.info(onnxruntime.get_device()) # priority device

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def softmax(x):
    x = x.reshape(-1)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def tranformer(n_px = 224):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    
def preprocess_image(image, n_px = 224):
    # resize so that the shorter side is 256, maintaining aspect ratio
    def image_resize(image, min_len):
        image = Image.fromarray(image)
        ratio = float(min_len) / min(image.size[0], image.size[1])
        if image.size[0] > image.size[1]:
            new_size = (int(round(ratio * image.size[0])), min_len)
        else:
            new_size = (min_len, int(round(ratio * image.size[1])))
        image = image.resize(new_size, Image.BILINEAR)
        return np.array(image)
    image = image_resize(image, n_px)

    # Crop centered window 224x224
    def crop_center(image, crop_w, crop_h):
        h, w, c = image.shape
        start_x = w//2 - crop_w//2
        start_y = h//2 - crop_h//2    
        return image[start_y:start_y+crop_h, start_x:start_x+crop_w, :]
    image = crop_center(image, n_px, n_px)
    
    # transpose
    image = image.transpose(2, 0, 1)

    # convert the input data into the float32 input
    img_data = image.astype('float32')    
    
    # normalize
    mean_vec = np.array([0.48145466, 0.4578275, 0.40821073])
    stddev_vec = np.array([0.26862954, 0.26130258, 0.27577711])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    print(f'image data shape {img_data.shape}')
    for i in range(img_data.shape[0]):
        norm_img_data[i,:,:] = (img_data[i,:,:]/255 - mean_vec[i]) / stddev_vec[i]
    
    # add batch channel
    norm_img_data = norm_img_data.reshape(1, 3, 224, 224).astype('float32')
    return norm_img_data

def convert(model = "ViT-B/32", visual_path = "clip_visual.onnx", textual_path = "clip_textual.onnx"):
    # onnx cannot export with cuda
    model, preprocess = clip.load(model, device="cpu", jit=False)

    # preprocess first
    image = preprocess(Image.open("dog.jpeg")).unsqueeze(0).cpu() # [1, 3, 224, 224]
    # tokenize first
    text = clip.tokenize(["a diagram", "a dog", "a cat"]).cpu() # [3, 77]

    onnx_model = clip_onnx(model, visual_path=visual_path, textual_path=textual_path)
    onnx_model.convert2onnx(image, text, verbose=True)

def load_model(visual_path = "clip_visual.onnx", textual_path = "clip_textual.onnx") -> clip_onnx:
    # ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
    onnx_model = clip_onnx(None, visual_path=visual_path, textual_path=textual_path)
    onnx_model.load_onnx(visual_path = "clip_visual.onnx", textual_path = "clip_textual.onnx",logit_scale=100.00000762939453)
    # init session with providers
    onnx_model.start_sessions(providers=["CPUExecutionProvider"]) # cpu mode
    return onnx_model

def inference_with_torch_onnx(onnx_model: clip_onnx, debug = False):
    # inference with torch and onnx, todo replace torch to onnx runtime
    preprocess = tranformer()
    # only support cpu inference preprocess handler
    image_preprocessed = preprocess(Image.open("CLIP.png")) # [3, 224, 224]
    if debug:
        logger.info("image preprocessed shape {}", image_preprocessed.shape)
    image = image_preprocessed.unsqueeze(0).cpu() # [1, 3, 224, 224]
    image_onnx = image.detach().cpu().numpy().astype(np.float32)
    if debug:
        logger.info("preprocess image shape {}", image.shape)

    # batch first
    text = clip.tokenize(["a diagram", "a dog", "a cat"]).cpu() # [3, 77]
    text_onnx = text.detach().cpu().numpy().astype(np.int32)
    if debug:
        logger.info("tokenize text shape {}", text.shape)

    # inference
    logits_per_image, logits_per_text = onnx_model(image_onnx, text_onnx)
    probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
    if debug:
        logger.info("Label probs: {}", probs)  # prints: [[0.9927937  0.00421067 0.00299571]]
        
def inference_with_full_onnx(onnx_model: clip_onnx, debug = False):
    # inference with torch and onnx, todo replace torch to onnx runtime
    image = Image.open("CLIP.png").convert("RGB")
    # only support cpu inference preprocess handler
    image_preprocessed = preprocess_image(np.array(image))# [3, 224, 224]
    if debug:
        logger.info("image preprocessed shape {}", image_preprocessed.shape)

    # batch first
    text = clip.tokenize(["a diagram", "a dog", "a cat"]).cpu() # [3, 77]
    text_onnx = text.detach().cpu().numpy().astype(np.int32)
    if debug:
        logger.info("tokenize text shape {}", text.shape)

    # inference
    image_features = onnx_model.visual_run(image_preprocessed)
    text_features = onnx_model.textual_run(text_onnx)
    logger.info("image features shape {}", image_features.shape)
    logger.info("text features shape {}", text_features.shape)
     # normalized features
    image_features = image_features / LA.norm(image_features, axis=1, keepdims=True)
    text_features = text_features / LA.norm(text_features, axis=1, keepdims=True)
    logger.info("after norm image features shape {}", image_features.shape)
    logger.info("after norm text features shape {}", text_features.shape)

    # cosine similarity as logits
    logits_per_image = np.dot(onnx_model.logit_scale * image_features,  text_features.T)
    probs = softmax(logits_per_image)
    if debug:
        logger.info("Label probs: {}", probs)  # prints: [[0.9927937  0.00421067 0.00299571]]

def test_inference():
    onnx_model = load_model()
    logger.info("start infer")
    for i in range(1):
        inference_with_full_onnx(onnx_model, debug=True)
        inference_with_torch_onnx(onnx_model, debug=True)
    logger.info("end infer")
    
if __name__ == "__main__":
    # convert()
    test_inference()
    