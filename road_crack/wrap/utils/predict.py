'''
@Project ：road_crack 
@File    ：predict.py
@Author  ：yuk
@Date    ：2024/6/5 11:51 
description：
'''
import time

import cv2
import os
import numpy as np
import torchvision
from . import UNet
import torch
from plot_mask import *


transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
class DetectCrack():
    def __init__(self,model_path,device):
        self.device = device
        self.net = UNet.MainNet().to(device)
        if os.path.exists(model_path):
            self.net.load_state_dict(torch.load(model_path, map_location=device))
            print('module is loaded !')
            self.net.eval()
        else:
            raise 'no model loaded'

    def run(self, data):
        img_h = data.shape[0]
        img_w = data.shape[1]
        ratio = 256 / max(img_h, img_w)
        new_size = (int(img_w * ratio), int(img_h * ratio))
        img_r = cv2.resize(data, new_size)

        bkg = np.zeros((256, 256, 3), dtype=np.uint8)
        bkg[:int(img_h * ratio), :int(img_w * ratio)] = img_r

        img_t = transform(bkg)
        img_t = img_t.unsqueeze(dim=0).to(self.device)
        s_t = time.time()
        res = self.net(img_t)
        e_t = time.time()
        print(e_t - s_t)
        res_img = res.squeeze(dim=0).permute(2, 1, 0).detach().cpu().numpy()
        # res_img = res.squeeze(dim=0).permute(2, 1, 0).detach().numpy()
        res_img = cv2.rotate(res_img, cv2.ROTATE_90_CLOCKWISE)
        res_img = cv2.flip(res_img, 1)
        mask = res_img[:int(img_h * ratio), :int(img_w * ratio)]
        mask = cv2.resize(mask, (img_w, img_h))
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        # cv2.imshow('img', mask)
        # cv2.waitKey(0)
        return mask_gray

# if __name__ == '__main__':
#     module = r'../module.pth'
#     device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
#     test_img = cv2.imread(r'../001.jpg')
#
#     d = DetectCrack(module,device)
#     mask = d.run(test_img)
#     mask[mask>0.05] = 255
#     mask[mask<0.05] = 0
#     # 在原图上画掩码图
#     mask_color = [
#         0, 255, 0
#     ]  # 输入颜色数组为RGB格式,这里只支持单类别的颜色设置
#     label, color_RGB = read_image_color(mask, mask_color)
#     masked_image = plot_mask(test_img, label, colors=color_RGB, alpha=0.3)
#
#     cv2.imshow('mask',masked_image)
#     cv2.waitKey(0)
