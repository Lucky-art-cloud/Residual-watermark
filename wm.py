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
from numpy import compat
from torch.autograd import Variable
from main import Y
from wideresnet import WideResNet


batch_size = 64
public_key = '585028aa0f794af812ee3be8804eb14a585028aa0f794af812ee3be8804eb14a'
wm = bin(compat.long(public_key, 16))[2:]
while len(wm) < 256:
    wm = '0' + wm
tmp_li = []
for x in wm:
    tmp_li.append(int(x))
tensor_li = torch.Tensor(tmp_li)
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
model = WideResNet(28, 10, 4).cuda()

# define loss function (criterion) and optimizer
sig = torch.nn.Sigmoid().cuda()
criterion = nn.CrossEntropyLoss().cuda()
criterion1 = nn.BCELoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), 0.1,
                            momentum=0.9,
                            weight_decay=5e-4)
schedule = torch.optim.lr_scheduler.StepLR(optimizer, 60, gamma=0.2, last_epoch=-1)

n_epochs = 200
for epoch in range(n_epochs):
    model.train()
    running_loss = 0.0
    running_correct = 0
    water_loss = 0.0
    print("Epoch {}/{}".format(epoch, n_epochs))
    print("-" * 10)
    for X_train, y_train in train_loader:
        X_train = X_train.cuda()
        y_train = y_train.cuda()
        X_train, y_train = Variable(X_train), Variable(y_train)
        outputs = model(X_train)
        _, pred = torch.max(outputs.data, 1)
        optimizer.zero_grad()
        parameter = model.conv1.weight.data
        P = parameter.mean(3).view(144, -1)
        Z = Y.mm(P)
        Z_emb = sig(Y).squeeze()
        loss2 = criterion1(Z_emb, tensor_li)
        loss1 = criterion(outputs, y_train)
        loss = loss1 + 0.01 * loss2
        # loss = loss1
        loss.backward()
        optimizer.step()
        running_loss += loss.data
        water_loss += loss2.data
        running_correct += torch.sum(pred == y_train.data)
    testing_correct = 0
    schedule.step()
    model.eval()
    for X_test, y_test in test_loader:
        X_test = X_test.cuda()
        y_test = y_test.cuda()
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
# '''       extract watermark         '''
res1 = model.conv1.weight.data
P1 = res1.mean(3).view(-1, 1)
res = ""
ext = Y.mm(P1).permute(1, 0)
print(ext)
for x in ext:
    if x >= 0:
        res += '1'
    else:
        res += '0'
print(hex(int(res, 2)))
torch.save(model.state_dict(), "model_parameter_watermark.pkl")
