# SRCNN

# import package
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


# import dataset function
import DwnSmpl

# transform
transform=transforms.Compose([transforms.ToTensor(),  # to Tensor
                              transforms.Normalize(
                                  [0.5, 0.5, 0.5],  # average RGB
                                  [0.5, 0.5, 0.5],  # std RGB
                                  )])

# Do function
dataset = DwnSmpl.DownsampleDataset('../../../Rdata/data', transform=transform, highreso_size=128, lowreso_size=32)
print("dataset size: {}".format(len(dataset)))

# divide train test
from sklearn.model_selection import train_test_split
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2)
print("train_dataset size: {}".format(len(train_dataset)))
print("test_dataset size: {}".format(len(test_dataset)))

# batch size
batch_size = 4

# dataloader
train_batch = torch.utils.data.DataLoader(dataset=train_dataset,  # subject to dataset
                                          batch_size=batch_size,  # batch size
                                          shuffle=True,  # image shuffle
                                          num_workers=0)  # parallel computing
test_batch = torch.utils.data.DataLoader(dataset=test_dataset,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=0)

# comfirm mini batch dateset
for highreso_images, lowreso_images in train_batch:
    print("batch highreso_images size: {}".format(highreso_images.size()))  # size of high reso
    print("highreso image size: {}".format(highreso_images[0].size()))  # one of high reso
    print("batch lowreso_images size: {}".format(lowreso_images.size()))  # size of low reso
    print("lowreso image size: {}".format(lowreso_images[1].size()))  # size of low reso
    break


# show image
def cat_imshow(x, y, images1, images2, images3=None):
  plt.figure(figsize=(9, 7))
  for i in range(x*y):  # X * Y
    if i <= 3:
      images = images1
      image = images[i] / 2 + 0.5  # De-standardization
    elif i > 3 and i <= 7:
      images = images2
      image = images[i-4] / 2 + 0.5
    elif images3 != None: 
      images = images3
      image = images[i-8] / 2 + 0.5
    
    image = image.numpy()  # Tensor to ndarrayへ
    plt.subplot(x, y, i+1)  # plot X x Y
    plt.imshow(np.transpose(image, (1, 2, 0)))  # change for matplotlib
    plt.axis('off') 
    plt.subplots_adjust(wspace=0, hspace=0)
  plt.show() 

# confirm image
for highreso_images, lowreso_images in train_batch:
  cat_imshow(2, 4, highreso_images, lowreso_images) 
  break


# neural network
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


# load network
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'mps') # For M1 mac
# check device
print("Device: {}".format(device))

net = SRCNN().to(device)
print(net)


# loss
criterion = nn.MSELoss()
# optimizer
optimizer = optim.Adam(net.parameters())


# evaluation
def cal_psnr_ssim(img1, img2, data_range=1):
  dim = len(img1.size()) # check image demension
  img1 = img1.to('cpu').detach().numpy()  # Tensor to ndarray
  img2 = img2.to('cpu').detach().numpy()

  # if image only one
  if dim == 3:
    # (channel, row, col)から(row, col, channel)
    img1 = np.transpose(img1, (1, 2, 0))  
    img2 = np.transpose(img2, (1, 2, 0))
    psnr = peak_signal_noise_ratio(img1, img2, data_range=data_range)  # PSNR
    # ssim = structural_similarity(img1, img2, multichannel=True, data_range=data_range)  # SSIM with multichannel
    ssim = structural_similarity(img1, img2, data_range=data_range, channel_axis = -1)  # SSIM with channel_axis = -1
    return psnr, ssim
  
  # if image get batch
  else:
    img1 = np.transpose(img1, (0, 2, 3, 1))  
    img2 = np.transpose(img2, (0, 2, 3, 1))
    # init
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

# define list
train_loss_list = []  # train MSE
test_loss_list = []  # test MSE
train_psnr_list = []  # train PSNR
test_psnr_list = []  # test PSNR
train_ssim_list = []  # train SSIM
test_ssim_list = []  # test SSIM


# number of epoch
epoch = 300 

for i in range(epoch):
    # progress
    print('---------------------------------------------')
    print("Epoch: {}/{}".format(i+1, epoch))

    # init loss
    train_loss = 0
    test_loss = 0 
    train_psnr = 0
    test_psnr = 0  
    train_ssim = 0
    test_ssim = 0  

    # ---------train part--------- #
    net.train()
    # load every mini batch
    i = 0
    for highreso_images, lowreso_images in train_batch:
        # GPU Tensor
        highreso_images = highreso_images.to(device)  
        lowreso_images = lowreso_images.to(device) 


        # init grad
        optimizer.zero_grad()
        # predict
        y_pred = net(lowreso_images)
        # loss
        loss = criterion(y_pred, highreso_images)  # MSE
        psnr, ssim = cal_psnr_ssim(y_pred, highreso_images)
        # calc loss
        loss.backward()
        # updata param
        optimizer.step()
        # Accumulate loss for each mini-batch
        train_loss += loss.item()  # MSE
        train_psnr += psnr  # PSNR
        train_ssim += ssim  # SSIM
        i += 1


    # ミニバッチの平均の損失を計算
    batch_train_loss = train_loss / len(train_batch)  # 損失(MSE)
    batch_train_psnr = train_psnr / len(train_batch)  # PSNR
    batch_train_ssim = train_ssim / len(train_batch)  # SSIM
    # ---------finish train part--------- #

    # ---------eval part--------- #
    # 
    net.eval()
    # 
    with torch.no_grad():
        for highreso_images, lowreso_images in test_batch:
            # 
            highreso_images = highreso_images.to(device)
            lowreso_images = lowreso_images.to(device)
            # 
            y_pred = net(lowreso_images)
            # 
            loss = criterion(y_pred, highreso_images)  # MSE
            psnr, ssim = cal_psnr_ssim(y_pred, highreso_images)
            # 
            test_loss += loss.item()  # MSE
            test_psnr += psnr  # PSNR
            test_ssim += ssim  # SSIM

    # Average loss of mini batch
    batch_test_loss = test_loss / len(test_batch)  # MSE
    batch_test_psnr = test_psnr / len(test_batch)  # PSNR
    batch_test_ssim = test_ssim / len(test_batch)  # SSIM
    # ---------finish eval part--------- #

    # show loss every epoch
    print("Train_Loss: {:.4f} Train_PSNR: {:.4f}  Train_SSIM: {:.4f}".format(
        batch_train_loss, batch_train_psnr, batch_train_ssim))
    print("Test_Loss: {:.4f} Test_PSNR: {:.4f}  Test_SSIM: {:.4f}".format(
        batch_test_loss, batch_test_psnr, batch_test_ssim))
    # save loss list
    train_loss_list.append(batch_train_loss)
    test_loss_list.append(batch_test_loss) 
    train_psnr_list.append(batch_train_psnr)
    test_psnr_list.append(batch_test_psnr)  
    train_ssim_list.append(batch_train_ssim)
    test_ssim_list.append(batch_test_ssim) 

# MSE
plt.figure()
plt.title('Train and Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(range(1, epoch+1), train_loss_list, color='blue',
         linestyle='-', label='Train_Loss')
plt.plot(range(1, epoch+1), test_loss_list, color='red',
         linestyle='--', label='Test_Loss')
plt.legend()

# PSNR
plt.figure()
plt.title('Train and Test PSNR')
plt.xlabel('Epoch')
plt.ylabel('PSNR')
plt.plot(range(1, epoch+1), train_psnr_list, color='blue',
         linestyle='-', label='Train_PSNR')
plt.plot(range(1, epoch+1), test_psnr_list, color='red',
         linestyle='--', label='Test_PSNR')
plt.legend() 

# SSIM
plt.figure()
plt.title('Train and Test SSIM')
plt.xlabel('Epoch')
plt.ylabel('SSIM')
plt.plot(range(1, epoch+1), train_ssim_list, color='blue',
         linestyle='-', label='Train_SSIM')
plt.plot(range(1, epoch+1), test_ssim_list, color='red',
         linestyle='--', label='Test_SSIM')
plt.legend()
plt.show()

# confirm reconstruction image
# eval mode
net.eval()
# auto grade off
with torch.no_grad():
    for highreso_images, lowreso_images in test_batch:
        highreso_images = highreso_images.to(device)
        lowreso_images = lowreso_images.to(device)
        y_pred = net(lowreso_images)
        # evaluate
        psnr1, ssim1 = cal_psnr_ssim(lowreso_images, highreso_images)  # low vs high
        psnr2, ssim2 = cal_psnr_ssim(y_pred, highreso_images)  # DL recon vs high
        print("Lowreso vs Highreso, PSNR: {:.4f}, SSIM: {:.4f}".format(psnr1, ssim1))
        print("DL_recon vs Highreso, PSNR: {:.4f}, SSIM: {:.4f}".format(psnr2, ssim2))
        
       
        cat_imshow(3, 4, highreso_images.to('cpu'), lowreso_images.to('cpu'), y_pred.to('cpu'))
        plt.show()
        break
