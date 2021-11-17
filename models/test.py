from models.model import ViT
import torch
import torch.nn as nn
from models.vit import MyViT
from utils import *

model_name = 'B_16_imagenet1k'
model1 = ViT(model_name, pretrained=True, num_classes=50, image_size=128)

"""from pytorchcv.model_provider import get_model
model2 = get_model("mobilenet_w1", pretrained=True)"""

model3 = MyViT(pretrained=True)
freeze_up_to(model3, "test", only_conv=False)

print(model1)
#print(model2)
print(model3)