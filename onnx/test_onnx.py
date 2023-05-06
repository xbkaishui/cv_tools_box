# 导入相关包
import io
import numpy as np
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx
import torch.nn as nn
import torch.nn.init as init
from loguru import logger
from PIL import Image
import torchvision.transforms as transforms
import onnxruntime


# 定义超分辨网络
class SuperResolutionNet(nn.Module):
    def __init__(self, upscale_factor, inplace=False):
        super(SuperResolutionNet, self).__init__()

        self.relu = nn.ReLU(inplace=inplace)
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x
    
	# 模型初始化
    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)

def export_model():
    # 实例化模型
    torch_model = SuperResolutionNet(upscale_factor=3)
    model_url = '/Users/xbkaishui/opensource/cv_hz/cv_tools_box/onnx/superres_epoch100-44c6958e.pth'
    batch_size = 1    # just a random number
    # 加载预训练得到权重
    map_location = lambda storage, loc: storage
    if torch.cuda.is_available():
        map_location = None
    # torch_model.load_state_dict(torch.load(model_url, map_location=map_location))
    torch_model.load_state_dict(torch.load(model_url, map_location="cpu"))

    # 将模型设置为推理模式
    torch_model.eval()
    # Input to the model
    x = torch.randn(batch_size, 1, 224, 224, requires_grad=True)
    output = torch_model(x)
    logger.info("output shape {}", output.shape)
    # 导出模型
    torch.onnx.export(torch_model,               # model being run
                    x,             # model input (or a tuple for multiple inputs)
                    "super_resolution.onnx",   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    verbose = True,
                    opset_version=15,   # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input names
                    output_names = ['output'], # the model's output names
                    # variable length axes
                    dynamic_axes={'input' : {0 : 'batch_size'},    
                                    'output' : {0 : 'batch_size'}})

# 将张量转化为ndarray格式
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    

def inference():
    ort_session = onnxruntime.InferenceSession("super_resolution.onnx")
    # 读取图片
    img = Image.open("/Users/xbkaishui/opensource/cv_hz/cv_tools_box/onnx/test_sr.png")
    # 对图片进行resize操作
    resize = transforms.Resize([224, 224])
    img = resize(img)

    img_ycbcr = img.convert('YCbCr')
    img_y, img_cb, img_cr = img_ycbcr.split()
    logger.info("img_y shape {}", img_y.size)
    to_tensor = transforms.ToTensor()
    img_y = to_tensor(img_y)
    img_y.unsqueeze_(0)
    logger.info("img_y shape {}", img_y.shape)
    # 构建输入的字典并将value转换位array格式
    input_name = ort_session.get_inputs()[0].name
    logger.info("input name {}", input_name)
    ort_inputs = {input_name: to_numpy(img_y)}
    ort_outs = ort_session.run(None, ort_inputs)
    img_out_y = ort_outs[0]
    img_out_y = Image.fromarray(np.uint8((img_out_y[0] * 255.0).clip(0, 255)[0]), mode='L')

    # 保存最后得到的图片
    final_img = Image.merge(
        "YCbCr", [
            img_out_y,
            img_cb.resize(img_out_y.size, Image.BICUBIC),
            img_cr.resize(img_out_y.size, Image.BICUBIC),
        ]).convert("RGB")

    final_img.save("cat_superres_with_ort.jpg")

if __name__ == '__main__':
    # export_model()
    inference()