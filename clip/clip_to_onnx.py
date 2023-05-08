# !pip install git+https://github.com/Lednik7/CLIP-ONNX.git

import os
# set to single omp thread
# os.environ["OMP_NUM_THREADS"] = "1"

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

def tranformer(n_px = 224):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

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

def inference(onnx_model: clip_onnx, debug = False):
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

def test_inference():
    onnx_model = load_model()
    logger.info("start infer")
    for i in range(100):
        inference(onnx_model)
    logger.info("end infer")
    
if __name__ == "__main__":
    # convert()
    test_inference()
    