# Downsample Dataset

# imprt package
import numpy as np
import glob
import os
import torch
import torchvision.transforms as transforms

# make dataset
from PIL import Image
class DownsampleDataset(torch.utils.data.Dataset):
  def __init__(self, root, transform=None, highreso_size=128, lowreso_size=32):
    self.transform = transform

    self.highreso_resize = transforms.Resize(highreso_size)  # high resolution
    self.lowreso_resize = transforms.Resize(lowreso_size)  # low resolution

    self.image_paths = sorted(glob.glob(os.path.join(root + '/*/*.jpg')))  # get image path list
    self.images_n = len(self.image_paths)

  def __len__(self):
    return self.images_n  # count number of image

  def __getitem__(self, index):
    path = self.image_paths[index]  # get file path
    image = Image.open(path)  # load image

    # 画像のリサイズ
    highreso_image = self.highreso_resize(image)  # high resolution image
    lowreso_image = self.highreso_resize(self.lowreso_resize(image))  # make low resolution image

    # if argument transform
    if self.transform:
      highreso_image = self.transform(highreso_image)
      lowreso_image = self.transform(lowreso_image)

    return highreso_image, lowreso_image
