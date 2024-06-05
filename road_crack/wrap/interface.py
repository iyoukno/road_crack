'''
@Project ：road_crack 
@File    ：interface.py
@Author  ：yuk
@Date    ：2024/6/5 15:52 
description：
'''
import sys
sys.path.append(r'/data0/zjt/Unet/road_crack')

import time

import torch

from utils.predict import DetectCrack
from utils.plot_mask import *



if __name__ == '__main__':
    module = r'module.pth'
    device = torch.device('cpu')
    test_img = cv2.imread(r'001.jpg')

    d = DetectCrack(module, device)
    # s_t = time.time()
    mask = d.run(test_img)
    # e_t = time.time()
    # print(e_t-s_t)
    mask[mask > 0.05] = 255
    mask[mask < 0.05] = 0
    # 在原图上画掩码图
    mask_color = [0, 255, 0]  # 输入颜色数组为RGB格式,这里只支持单类别的颜色设置
    label, color_RGB = read_image_color(mask, mask_color)
    masked_image = plot_mask(test_img, label, colors=color_RGB, alpha=0.3)
    out_path = 'result.jpg'
    cv2.imwrite(out_path, masked_image)
    # cv2.imshow('mask', masked_image)
    # cv2.waitKey(0)

