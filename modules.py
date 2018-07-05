import numpy as np
import torch
from torch.utils.serialization import load_lua
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential()
        self.model.add_module('conv1', nn.Conv2d(8, 32, kernel_size=7, stride=1, padding=3))
        self.model.add_module('relu1', nn.ReLU())
        self.model.add_module('conv2', nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=3))
        self.model.add_module('relu2', nn.ReLU())
        self.model.add_module('conv3', nn.Conv2d(64, 32, kernel_size=7, stride=1, padding=3))
        self.model.add_module('relu3', nn.ReLU())
        self.model.add_module('conv4', nn.Conv2d(32, 16, kernel_size=7, stride=1, padding=3))
        self.model.add_module('relu4', nn.ReLU())
        self.model.add_module('conv5', nn.Conv2d(16, 2, kernel_size=7, stride=1, padding=3))
        
        
    def forward(self, input):
        output = self.model(input)
        return output



def loadTorchModel(torch_model_path):
    net_pytorch = Net()
    model_pytorch = net_pytorch.model
    model_torch = load_lua(torch_model_path)

    assert len(model_torch) == len(model_pytorch), 'Error: Torch Module and PyTorch Module have different Lengths!'

    for i in range(len(model_torch)):
        if i % 2 == 0:
            model_pytorch[i].weight = torch.nn.Parameter(model_torch.modules[i].weight)
            model_pytorch[i].bias = torch.nn.Parameter(model_torch.modules[i].bias)

    return net_pytorch