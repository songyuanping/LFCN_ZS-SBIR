import torch
import torch.nn as nn
from senet import cse_resnet50
full_model=cse_resnet50(num_classes=1000)
x=torch.rand((128,3,224,224))

print(full_model.features(x,torch.ones((128, 1))).shape) #torch.Size([128, 2048, 7, 7])
print(full_model.logits(full_model.features(x,torch.ones((128, 1)))).shape) #torch.Size([128, 1000])
