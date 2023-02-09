import torch
from numpy import compat
from wideresnet import WideResNet

# 1.training set


# x_data = [1.0, 2.0, 3.0, 4.0]
model = WideResNet(28, 10, 4)
x_data = torch.randn(256, 144)
sig = torch.nn.Sigmoid()
criterion1 = torch.nn.BCELoss()
# y_data = [2.0, 4.0, 6.0, 8.0]
public_key = '585028aa0f794af812ee3be8804eb14a585028aa0f794af812ee3be8804eb14a'
wm = bin(compat.long(public_key, 16))[2:]
while len(wm) < 256:
    wm = '0' + wm
tmp_li = []
for x in wm:
    tmp_li.append(int(x))
y_data = torch.Tensor(tmp_li)

# # 2.initial_w
# w = torch.Tensor([1.0])
# # 需要计算梯度
# w.requires_grad = True
parameter = model.conv1.weight
P = parameter.mean(3).view(144, -1)
# P.requires_grad = True
optimizer = torch.optim.SGD(model.conv1.parameters(), 0.1)


# 3.定义函数

def forward(x):
    global parameter, P
    P = parameter.mean(3).view(144, -1)
    # P.requires_grad = True
    Z = x.mm(P)
    return sig(Z).squeeze()


def loss(x, y):
    y_pred = forward(x)
    loss = criterion1(y_pred, y)
    return loss


def convert(y):
    res = ''
    for x in y:
        if x >= 0.5:
            res += '1'
        else:
            res += '0'
    return hex(int(res, 2))


resOri = forward(x_data)
print("Predict (Before Training):", convert(resOri))

# w是tensor，所以相乘得到的forward(x)也是tensor，取其值时需要用item()将其转换成标量


# 5.迭代
for epoch in range(200):
    for x, y in zip(x_data, y_data):  # 遍历每一组数据
        x = x.unsqueeze(0)
        l = loss(x, y)  # 先把从w到loss的流程走一遍，相当于前向传播
        l.backward()  # 反向传播，程序会自动求出所需要的梯度
        # P.data = P.data - 0.1 * P.grad.data
        # P.grad.data.zero_()
        optimizer.step()
        optimizer.zero_grad()

    print("progress:", epoch, l.item())  # 输出最后一组数据的损失值
resFin = forward(x_data)
print("Predict (After Training)", convert(resFin))
