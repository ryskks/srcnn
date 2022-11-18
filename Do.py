# SRCNN
#

# パッケージのインポート
import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
import torch.nn.functional as F
from torch import optim
from skimage.metrics import peak_signal_noise_ratio, structural_similarity 


# データセット取得関数
import DwnSmpl

# 画像前処理の設定
transform=transforms.Compose([transforms.ToTensor(),  # Tensorに変換
                              transforms.Normalize(
                                  [0.5, 0.5, 0.5],  # RGBの平均
                                  [0.5, 0.5, 0.5],  # RGBの標準偏差
                                  )])

# データセットの作成を実行
dataset = DwnSmpl.DownsampleDataset('../../../Rdata/lfw-deepfunneled', transform=transform, highreso_size=128, lowreso_size=32)
print("dataset size: {}".format(len(dataset)))

# 訓練データとテストデータに分割
from sklearn.model_selection import train_test_split
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2)
print("train_dataset size: {}".format(len(train_dataset)))
print("test_dataset size: {}".format(len(test_dataset)))

# バッチサイズ
batch_size = 64

# データローダーを作成
train_batch = torch.utils.data.DataLoader(dataset=train_dataset,  # 対象となるデータセット
                                          batch_size=batch_size,  # バッチサイズ
                                          shuffle=True,  # 画像のシャッフル
                                          num_workers=0)  # 並列処理数，増やすとエラー出る可能性
test_batch = torch.utils.data.DataLoader(dataset=test_dataset,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=0)

# ミニバッチデータセットの確認
for highreso_images, lowreso_images in train_batch:
    print("batch highreso_images size: {}".format(highreso_images.size()))  # 高解像度画像バッチのサイズ
    print("highreso image size: {}".format(highreso_images[0].size()))  # 1枚の高解像度画像サイズ
    print("batch lowreso_images size: {}".format(lowreso_images.size()))  # 低解像度画像バッチのサイズ
    print("lowreso image size: {}".format(lowreso_images[1].size()))  # 1枚の低解像度画像サイズ
    break


# 画像の表示
def cat_imshow(x, y, images1, images2, images3=None):
  plt.figure(figsize=(9, 7))
  for i in range(x*y):  # X * Y枚の画像を表示
    if i <= 3:
      images = images1
      image = images[i] / 2 + 0.5  # 標準化を解除
    elif i > 3 and i <= 7:
      images = images2
      image = images[i-4] / 2 + 0.5
    elif images3 != None: 
      images = images3
      image = images[i-8] / 2 + 0.5
    
    image = image.numpy()  # Tensorからndarrayへ
    plt.subplot(x, y, i+1)  # X x Yとなるように格子状にプロット
    plt.imshow(np.transpose(image, (1, 2, 0)))  # matplotlibでは(縦, 横, チャネル)の順
    plt.axis('off')  # 目盛を消去
    plt.subplots_adjust(wspace=0, hspace=0)  # 画像間の余白の設定
  plt.show()  # 表示

# 画像の確認
for highreso_images, lowreso_images in train_batch:
  cat_imshow(2, 4, highreso_images, lowreso_images)  # 画像の表示
  break


# ニューラルネットワークの定義
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=5, padding=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x


# ネットワークのロード
# CPUとGPUどちらを使うかを指定
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'mps') # For M1 mac
# デバイスの確認
print("Device: {}".format(device))

net = SRCNN().to(device)
print(net)


# 損失関数の定義
criterion = nn.MSELoss()
# 最適化関数の定義
optimizer = optim.Adam(net.parameters())


# 画像評価指標の計算
def cal_psnr_ssim(img1, img2, data_range=1):
  dim = len(img1.size()) # 画像の次元数を確認
  img1 = img1.to('cpu').detach().numpy()  # Tensorからndarrayに変換
  img2 = img2.to('cpu').detach().numpy()

  # 画像が1枚だけの場合
  if dim == 3:
    # (チャネル, 縦, 横)から(縦, 横, チャネル)の順になるよう並び替え
    img1 = np.transpose(img1, (1, 2, 0))  
    img2 = np.transpose(img2, (1, 2, 0))
    psnr = peak_signal_noise_ratio(img1, img2, data_range=data_range)  # PSNR
    # ssim = structural_similarity(img1, img2, multichannel=True, data_range=data_range)  # SSIM with multichannel
    ssim = structural_similarity(img1, img2, data_range=data_range, channel_axis = -1)  # SSIM with channel_axis = -1
    return psnr, ssim
  
  # 画像がバッチで渡された場合
  else:
    img1 = np.transpose(img1, (0, 2, 3, 1))  
    img2 = np.transpose(img2, (0, 2, 3, 1))
    # 初期化
    all_psnr = 0
    all_ssim = 0
    n_batchs = img1.shape[0]
    for i in range(n_batchs):
      psnr = peak_signal_noise_ratio(img1[i], img2[i], data_range=data_range)  # PSNR
      # ssim = structural_similarity(img1[i], img2[i], data_range=data_range, multichannel=True)  # SSIM
      ssim = structural_similarity(img1[i], img2[i], data_range=data_range, channel_axis = -1)  # SSIM
      all_psnr += psnr
      all_ssim += ssim

    mean_psnr = all_psnr / n_batchs
    mean_ssim = all_ssim / n_batchs
    return mean_psnr, mean_ssim

for highreso_images, lowreso_images in train_batch:
  psnr, ssim = cal_psnr_ssim(highreso_images[0], lowreso_images[0], data_range=1)  # 画像1枚だけ入力
  batch_psnr, batch_ssim = cal_psnr_ssim(highreso_images, lowreso_images, data_range=1)  # 画像バッチで入力
  print("SINGLE PSNR: {:.4f}, SSIM: {:.4f}".format(psnr, ssim))
  print("BATCH  PSNR: {:.4f}, SSIM: {:.4f}".format(batch_psnr, batch_ssim))
  break

# 損失を保存するリストを作成
train_loss_list = []  # 学習損失（MSE）
test_loss_list = []  # 評価損失（MSE）
train_psnr_list = []  # 学習PSNR
test_psnr_list = []  # 評価PSNR
train_ssim_list = []  # 学習SSIM
test_ssim_list = []  # 評価SSIM


# 学習（エポック）の実行
epoch = 5  # 学習回数

for i in range(epoch):
    # エポックの進行状況を表示
    print('---------------------------------------------')
    print("Epoch: {}/{}".format(i+1, epoch))

    # 損失の初期化
    train_loss = 0  # 学習損失（MSE）
    test_loss = 0  # 評価損失（MSE）
    train_psnr = 0  # 学習PSNR
    test_psnr = 0  # 評価PSNR
    train_ssim = 0  # 学習SSIM
    test_ssim = 0  # 評価SSIM

    # ---------学習パート--------- #
    # ニューラルネットワークを学習モードに設定
    net.train()
    # ミニバッチごとにデータをロードし学習
    i = 0
    for highreso_images, lowreso_images in train_batch:
        # GPUにTensorを転送
        highreso_images = highreso_images.to(device)  # 高解像度画像
        lowreso_images = lowreso_images.to(device)  # 低解像度画像


        # 勾配を初期化
        optimizer.zero_grad()
        # データを入力して予測値を計算（順伝播）
        y_pred = net(lowreso_images)
        # 損失（誤差）を計算
        loss = criterion(y_pred, highreso_images)  # MSE
        psnr, ssim = cal_psnr_ssim(y_pred, highreso_images)
        # 勾配の計算（逆伝搬）
        loss.backward()
        # パラメータ（重み）の更新
        optimizer.step()
        # ミニバッチごとの損失を蓄積
        train_loss += loss.item()  # MSE
        train_psnr += psnr  # PSNR
        train_ssim += ssim  # SSIM
        i += 1


    # ミニバッチの平均の損失を計算
    batch_train_loss = train_loss / len(train_batch)  # 損失(MSE)
    batch_train_psnr = train_psnr / len(train_batch)  # PSNR
    batch_train_ssim = train_ssim / len(train_batch)  # SSIM
    # ---------学習パートはここまで--------- #

    # ---------評価パート--------- #
    # ニューラルネットワークを評価モードに設定
    net.eval()
    # 評価時の計算で自動微分機能をオフにする
    with torch.no_grad():
        for highreso_images, lowreso_images in test_batch:
            # GPUにTensorを転送
            highreso_images = highreso_images.to(device)
            lowreso_images = lowreso_images.to(device)
            # データを入力して予測値を計算（順伝播）
            y_pred = net(lowreso_images)
            # 損失（誤差）を計算
            loss = criterion(y_pred, highreso_images)  # MSE
            psnr, ssim = cal_psnr_ssim(y_pred, highreso_images)
            # ミニバッチごとの損失を蓄積
            test_loss += loss.item()  # MSE
            test_psnr += psnr  # PSNR
            test_ssim += ssim  # SSIM

    # ミニバッチの平均の損失を計算
    batch_test_loss = test_loss / len(test_batch)  # 損失(MSE)
    batch_test_psnr = test_psnr / len(test_batch)  # PSNR
    batch_test_ssim = test_ssim / len(test_batch)  # SSIM
    # ---------評価パートはここまで--------- #

    # エポックごとに損失を表示
    print("Train_Loss: {:.4f} Train_PSNR: {:.4f}  Train_SSIM: {:.4f}".format(
        batch_train_loss, batch_train_psnr, batch_train_ssim))
    print("Test_Loss: {:.4f} Test_PSNR: {:.4f}  Test_SSIM: {:.4f}".format(
        batch_test_loss, batch_test_psnr, batch_test_ssim))
    # 損失をリスト化して保存
    train_loss_list.append(batch_train_loss)  # 訓練損失リスト
    test_loss_list.append(batch_test_loss)  # テスト訓練リスト
    train_psnr_list.append(batch_train_psnr)  # 訓練PSNR
    test_psnr_list.append(batch_test_psnr)  # テストPSNR
    train_ssim_list.append(batch_train_ssim)  # 訓練SSIM
    test_ssim_list.append(batch_test_ssim)  # テストSSIM

# 損失（MSE）
plt.figure()
plt.title('Train and Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(range(1, epoch+1), train_loss_list, color='blue',
         linestyle='-', label='Train_Loss')
plt.plot(range(1, epoch+1), test_loss_list, color='red',
         linestyle='--', label='Test_Loss')
plt.legend()  # 凡例

# PSNR
plt.figure()
plt.title('Train and Test PSNR')
plt.xlabel('Epoch')
plt.ylabel('PSNR')
plt.plot(range(1, epoch+1), train_psnr_list, color='blue',
         linestyle='-', label='Train_PSNR')
plt.plot(range(1, epoch+1), test_psnr_list, color='red',
         linestyle='--', label='Test_PSNR')
plt.legend()  # 凡例

# SSIM
plt.figure()
plt.title('Train and Test SSIM')
plt.xlabel('Epoch')
plt.ylabel('SSIM')
plt.plot(range(1, epoch+1), train_ssim_list, color='blue',
         linestyle='-', label='Train_SSIM')
plt.plot(range(1, epoch+1), test_ssim_list, color='red',
         linestyle='--', label='Test_SSIM')
plt.legend()  # 凡例

# 表示
plt.show()

# 再構成した画像の確認
# ニューラルネットワークを評価モードに設定
net.eval()
# 推定時の計算で自動微分機能をオフにする
with torch.no_grad():
    for highreso_images, lowreso_images in test_batch:
        # GPUにTensorを転送
        highreso_images = highreso_images.to(device)
        lowreso_images = lowreso_images.to(device)
        # データを入力して予測値を計算（順伝播）
        y_pred = net(lowreso_images)
        # 画質評価
        psnr1, ssim1 = cal_psnr_ssim(lowreso_images, highreso_images)  # 低解像度 vs 高解像度
        psnr2, ssim2 = cal_psnr_ssim(y_pred, highreso_images)  # ディープラーニング再構成 vs 高解像度
        print("Lowreso vs Highreso, PSNR: {:.4f}, SSIM: {:.4f}".format(psnr1, ssim1))
        print("DL_recon vs Highreso, PSNR: {:.4f}, SSIM: {:.4f}".format(psnr2, ssim2))
        
        # 画像表示
        cat_imshow(3, 4, highreso_images.to('cpu'), lowreso_images.to('cpu'), y_pred.to('cpu'))
        plt.show()  # 表示
        break
