import kornia
import torch
from kornia.augmentation import RandomCrop
import time

_ = torch.manual_seed(0)
inputs = torch.randn(32, 3, 1024, 1024).to('cuda')
start = time.time()
aug = RandomCrop((640, 640), p=1., cropping_mode="resample")
for i in range(1000):
    out = aug(inputs)
end = time.time()
print(f'kornia random_crop: {(end - start) * 1000}')