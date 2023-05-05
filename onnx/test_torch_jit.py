import torch 
from torchvision.models import resnet18 
 
# 使用PyTorch model zoo中的resnet18作为例子 
model = resnet18() 
model.eval() 
 
# 通过trace的方法生成IR需要一个输入样例 
dummy_input = torch.rand(1, 3, 224, 224) 
 
# IR生成 
with torch.no_grad(): 
    jit_model = torch.jit.trace(model, dummy_input) 