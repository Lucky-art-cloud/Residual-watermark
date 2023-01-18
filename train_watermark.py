import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable

from wideresnet import WideResNet

# used for logging to TensorBoard
from tensorboard_logger import configure, log_value

batch_size = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Data loading code
normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                 std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                      (4, 4, 4, 4), mode='reflect').squeeze()),
    transforms.ToPILImage(),
    transforms.RandomCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])
# transform_train = transforms.Compose([
#         transforms.ToTensor(),
#         normalize,
#     ])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize
])

data_train = datasets.CIFAR10(root='../data', transform=transform_train, train=True, download=True)
data_test = datasets.CIFAR10(root='../data', transform=transform_test, train=False)
train_loader = torch.utils.data.DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(dataset=data_test, batch_size=batch_size, shuffle=True, num_workers=0)

# create model
model = WideResNet(28, 10, 4).cuda(device)

# define loss function (criterion) and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), 0.001,
                            momentum=0.9,
                            weight_decay=5e-4)

n_epochs = 8
for epoch in range(n_epochs):
    running_loss = 0.0
    running_correct = 0
    water_loss = 0.0
    print("Epoch {}/{}".format(epoch, n_epochs))
    print("-" * 10)
    for data in train_loader:
        X_train, y_train = data
        X_train, y_train = Variable(X_train), Variable(y_train)
        outputs = model(X_train)
        _, pred = torch.max(outputs.data, 1)
        optimizer.zero_grad()
        # name, parameter = model.conv1[0].named_parameters()
        # P = name[1].data[0:64].mean(3).view(192, -1)
        # Y = X.mm(P)
        # Y_emb = sig(Y)
        # Y_emb = Y_emb.squeeze()
        # loss2 = cost2(Y_emb, tensor_li)
        loss1 = criterion(outputs, y_train)
        # loss = loss1 + 0.3 * loss2
        loss = loss1
        loss.backward()
        optimizer.step()
        running_loss += loss.data
        # water_loss += loss2.data
        running_correct += torch.sum(pred == y_train.data)
    testing_correct = 0
    # '''       extract watermark         '''
    # if epoch == 7:
    #     name1, parameter1 = model.conv1[0].named_parameters()
    #     P1 = name1[1].data[0:64].mean(3).view(-1, 1)
    #     res = ""
    #     ext = X.mm(P1).permute(1, 0)
    #     print(ext)
    # for x in ext:
    #     if x >= 0:
    #         res += '1'
    #     else:
    #         res += '0'
    # print(hex(int(res, 2)))
    for data in test_loader:
        X_test, y_test = data
        X_test, y_test = Variable(X_test), Variable(y_test)
        outputs = model(X_test)
        _, pred = torch.max(outputs.data, 1)
        testing_correct += torch.sum(pred == y_test.data)
    print("Loss is:{:.4f}, Water_loss is:{:.4f}, Train Accuracy is:{:.4f}%, Test Accuracy is:{:.4f}".format(
        running_loss / len(data_train),
        water_loss / len(data_train),
        100 * running_correct / len(
            data_train),
        100 * testing_correct / len(
            data_test)))
torch.save(model.state_dict(), "model_parameter_watermark.pkl")
