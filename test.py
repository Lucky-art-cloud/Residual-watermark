import torch
from wideresnet import WideResNet
import numpy as np
from main import Y

torch.set_printoptions(threshold=np.inf)
x = torch.rand(5)
y = torch.rand(5)
print(x.unsqueeze(1).size())