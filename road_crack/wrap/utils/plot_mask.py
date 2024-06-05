'''
@Project ：road_crack 
@File    ：plot_mask.py
@Author  ：yuk
@Date    ：2024/6/5 15:31 
description：
'''
import numpy as np
import cv2


def plot_mask(img, masks, colors=None, alpha=0.5) -> np.ndarray:
   """Visualize segmentation mask.

    Parameters
    ----------
    img: numpy.ndarray
        Image with shape `(H, W, 3)`.
    masks: numpy.ndarray
        Binary images with shape `(N, H, W)`.
    colors: numpy.ndarray
        color for mask, shape `(N, 3)`.
        if None, generate random color for mask
    alpha: float, optional, default 0.5
        Transparency of plotted mask

    Returns
    -------
    numpy.ndarray
        The image plotted with segmentation masks, shape `(H, W, 3)`

    """
   if colors is None:
      colors = np.random.random((masks.shape[0], 3)) * 255
   else:
      if colors.shape[0] < masks.shape[0]:
         raise RuntimeError(
            f"colors count: {colors.shape[0]} is less than masks count: {masks.shape[0]}"
         )
   for mask, color in zip(masks, colors):
      mask = np.stack([mask, mask, mask], -1)
      img = np.where(mask, img * (1 - alpha) + color * alpha, img)

   return img.astype(np.uint8)


def read_image_color(mask, color_map):
   """
   :param image_path: 图像文件路径
   :param label_path: 标签文件路径
   :param color_map: 输入的颜色列表
   :returns:
   img: numpy.ndarray
        Image with shape `(H, W, 3)`.
    label_trans: numpy.ndarray
        Binary images with shape `(N, H, W)`.
    RGB_map: numpy.ndarray
        color for mask, shape `(N, 3)`.
   """
   # img = cv2.imread(image_path)
   # mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
   mask = mask[:, :, np.newaxis]
   # print(mask.shape)
   label_trans = mask.transpose(2, 0, 1)
   color_map[0], color_map[2] = color_map[2], color_map[0]
   RGB_map = np.array(color_map)
   RGB_map = RGB_map.T
   RGB_map = RGB_map[np.newaxis, :]

   return label_trans, RGB_map


# if __name__ == "__main__":
#    img_path = r'D:\datasets\road_det\CrackForest-dataset-master\image\001.jpg'
#    mask_path = r'D:\datasets\road_det\CrackForest-dataset-master\groundTruthPngImg\001.png'
#    mask_color = [
#       255, 0, 0
#    ]  # 输入颜色数组为RGB格式,这里只支持单类别的颜色设置，多类别的后续可能会改进出来
#    image, label, color_RGB = read_image_color(img_path, mask_path, mask_color)
#    masked_image = plot_mask(image, label, colors=color_RGB, alpha=0.3)
#    out_path = 'test.jpg'
#    cv2.imwrite(out_path, masked_image)