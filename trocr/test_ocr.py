from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

model_name = "microsoft/trocr-base-handwritten"
model_name = "microsoft/trocr-base-printed"
model_name = "microsoft/trocr-large-printed"
model_path = "/tmp/ocrs/checkpoint-10/"
processor = TrOCRProcessor.from_pretrained(model_name)
# processor = TrOCRProcessor.from_pretrained(pretrained_model_name_or_path=model_path)
# model = VisionEncoderDecoderModel.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(pretrained_model_name_or_path=model_path)

# 将模型移动到 GPU 上
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
model.to(device)

# load image from the IAM dataset
# image = Image.open("/opt/LMFlow/a01-122-02.jpg").convert("RGB")
image = Image.open("/opt/LMFlow/test_ocr_2.png").convert("RGB")
# image = Image.open("/opt/LMFlow/ocr_2.jpg").convert("RGB")
# image = Image.open("/tmp/ocr_2.jpg").convert("RGB")
image = Image.open("/tmp/test_ocr_2.png").convert("RGB")
image = Image.open("/tmp/results/a2.png").convert("RGB")
# 26461 ok
image = Image.open("/tmp/ocrs/6.png").convert("RGB")
# AT3206 wrong 4T3206
image = Image.open("/tmp/ocrs/5_gan.png").convert("RGB")
# 413205 wrong 4T3205
image = Image.open("/tmp/ocrs/4.png").convert("RGB")
# 26161 wrong 26461
image = Image.open("/tmp/ocrs/3_gan.png").convert("RGB")
# 313205 wrong 3T3205
# image = Image.open("/tmp/ocrs/2_gan.png").convert("RGB")
# 26161 wrong 26461
# image = Image.open("/tmp/ocrs/1.png").convert("RGB")


pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
generated_ids = model.generate(pixel_values).to(device)

generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(f'result {generated_text}')