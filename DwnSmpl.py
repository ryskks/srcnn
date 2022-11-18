#

# パッケージのインポート
import numpy as np
import glob
import os
import torch
import torchvision.transforms as transforms

# データセットの作成
from PIL import Image
class DownsampleDataset(torch.utils.data.Dataset):
  def __init__(self, root, transform=None, highreso_size=128, lowreso_size=32):
    self.transform = transform

    self.highreso_resize = transforms.Resize(highreso_size)  # 高解像度
    self.lowreso_resize = transforms.Resize(lowreso_size)  # 低解像度

    self.image_paths = sorted(glob.glob(os.path.join(root + '/*/*jpg')))  # 画像パスのリスト取得
    self.images_n = len(self.image_paths)

  def __len__(self):
    return self.images_n  # 画像数のカウント

  def __getitem__(self, index):
    path = self.image_paths[index]  # indexをもとに画像のファイルパスを取得
    image = Image.open(path)  # 画像読み込み

    # 画像のリサイズ
    highreso_image = self.highreso_resize(image)  # 高解像度画像
    lowreso_image = self.highreso_resize(self.lowreso_resize(image))  # 低解像度画像。一度低解像度にしてから高解像度と同じ画像サイズに変換

    # transformが引数で与えられた場合
    if self.transform:
      highreso_image = self.transform(highreso_image)
      lowreso_image = self.transform(lowreso_image)

    return highreso_image, lowreso_image
