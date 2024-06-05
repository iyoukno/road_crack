'''
@Project ：road_crack 
@File    ：train.py
@Author  ：yuk
@Date    ：2024/5/28 10:15 
description：
'''
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import UNet2
import MyDataset

path = r'/data0/zjt/datasets/road_seg/CrackForest-dataset-master'
module = r'./weight/module.pth'
img_save_path = r'IMG'
batch = 1
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

net = UNet.MainNet().to(device)
optimizer = torch.optim.Adam(net.parameters())
loss_func = nn.BCELoss()

dataloader = DataLoader(MyDataset.MKDataset(path), batch_size=10, shuffle=True)

if os.path.exists(module):
    net.load_state_dict(torch.load(module))
    print('module is loaded !')
if not os.path.exists(img_save_path):
    os.mkdir(img_save_path)

for epoch in range(200):
    for i, (xs, ys) in enumerate(dataloader):
        xs = xs.to(device)
        ys = ys.to(device)

        xs_ = net(xs)

        loss = loss_func(xs_, ys)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 5 == 0:
            print('batch: {}  count: {}  loss: {}'.format(batch, (i + 1) * 4, loss))

    torch.save(net.state_dict(), module)
    print('module is saved !')

    x = xs[0]#原图
    x_ = xs_[0]#网络输出
    y = ys[0]#标签
    z = torch.cat((x, x_, y), 2)
    img_save = transforms.ToPILImage()(z.cpu())
    img_save.save(os.path.join(img_save_path, '{}.png'.format(batch)))

    batch += 1