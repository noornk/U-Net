import torch
from torch import nn
import torch.nn.functional as F
import os
import random
import pydicom
from PIL import Image
import numpy as np
import torch.utils.data as utils_data
from os import listdir
from os.path import isfile, join
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt
import cv2

def img_to_numpy(img_path):
    """Reads general image path and returns np.array normalized to 1
    """
    mode_to_bits = {'1':1, 'L':8, 'P':8, 'RGB':8, 'RGBA':32, 'CMYK':32, 'YCbCr':24, 'I':16, 'F':32}
    img = Image.open(open(img_path, 'rb'))
    bits = 16
    img = np.array(img, dtype=float)
    norm = 2**bits-1
    img /= norm
    return img

  
def dicom_to_numpy(img_path):
    """Reads DICOM image path and returns np.array normalized to 1
    """
    img = pydicom.dcmread(img_path)
    bits = 8
    img = img.pixel_array
    img = img.astype(float)
    norm = 2**bits-1
    img /= norm
    return img

def shaped(arr):
    h, w = arr.shape
    return h, w

def reverse(mask):
    verse = {0.0: 1.0, 1.0: 0.0}
    for k in verse:
        mask[mask == k] = verse[k]
    return mask
      
class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding,
                 stride):
        super(BaseConv, self).__init__()

        self.act = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding,
                               stride)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size,
                               padding, stride)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return x


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding,
                 stride):
        super(DownConv, self).__init__()

        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv_block = BaseConv(in_channels, out_channels, kernel_size,
                                   padding, stride)

    def forward(self, x):
        x = self.pool1(x)
        x = self.conv_block(x)
        return x


class UpConv(nn.Module):
    def __init__(self, in_channels, in_channels_skip, out_channels,
                 kernel_size, padding, stride):
        super(UpConv, self).__init__()

        self.conv_trans1 = nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size=2, padding=0, stride=2)
        self.conv_block = BaseConv(
            in_channels=in_channels + in_channels_skip,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride)

    def forward(self, x, x_skip):
        x = self.conv_trans1(x)
        x = torch.cat((x, x_skip), dim=1)
        x = self.conv_block(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, n_class, kernel_size,
                 padding, stride):
        super(UNet, self).__init__()

        self.init_conv = BaseConv(in_channels, out_channels, kernel_size,
                                  padding, stride)

        self.down1 = DownConv(out_channels, 2 * out_channels, kernel_size,
                              padding, stride)

        self.down2 = DownConv(2 * out_channels, 4 * out_channels, kernel_size,
                              padding, stride)

        self.down3 = DownConv(4 * out_channels, 8 * out_channels, kernel_size,
                              padding, stride)
        
        self.down4 = DownConv(8 * out_channels, 16 * out_channels, kernel_size,
                              padding, stride)

        self.up4 = UpConv(16 * out_channels, 8 * out_channels, 8 * out_channels,
                          kernel_size, padding, stride)
        
        self.up3 = UpConv(8 * out_channels, 4 * out_channels, 4 * out_channels,
                          kernel_size, padding, stride)

        self.up2 = UpConv(4 * out_channels, 2 * out_channels, 2 * out_channels,
                          kernel_size, padding, stride)

        self.up1 = UpConv(2 * out_channels, out_channels, out_channels,
                          kernel_size, padding, stride)

        self.out = nn.Conv2d(out_channels, n_class, kernel_size, padding, stride)

    def forward(self, x):
        # Encoder
        x = self.init_conv(x)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        # Decoder
        x_up = self.up4(x4, x3)
        x_up = self.up3(x3, x2)
        x_up = self.up2(x_up, x1)
        x_up = self.up1(x_up, x)
        x_out = self.out(x_up)
#         x_out = F.sigmoid(self.out(x_up))
#         x_out = F.relu(self.out(x_up))
        return x_out


class MyDataSet(utils_data.Dataset):

    def __init__(self, root_dir, image_dir, mask_dir, label, img_transform=None, mask_transform=None):
        self.dataset_path = root_dir
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        # self.mask_dir = os.path.join(mask_dir, label)
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        mask_full_path = os.path.join(self.dataset_path, self.mask_dir)
        self.mask_file_list = [f for f in listdir(mask_dir) if isfile(join(mask_dir, f))]
        random.shuffle(self.mask_file_list)
        
        self.mapping = {
          0.007782101167315175: 0,
          0.5019455252918288: 0,
          0.9961089494163424: 1

    def mask_to_class(self, mask):
        for k in self.mapping:
            mask[mask == k] = self.mapping[k]
        return mask

    def __getitem__(self, index):
        file_name = self.mask_file_list[index]
        img_name = os.path.join(self.dataset_path, self.image_dir, file_name.replace(".png", ".dcm"))
        mask_name = os.path.join(self.dataset_path, self.mask_dir, file_name)
        image = dicom_to_numpy(img_name)
        mask_p = img_to_numpy(mask_name)
        image = cv2.resize(image, (256, 256))
        mask_p = cv2.resize(mask_p, (256, 256))
        mask_p = self.mask_to_class(mask_p)
        image = blockshaped(image, 256, 256)
        mask = blockshaped(mask_p, 256, 256)
        labels = []
        img = []
        for j in range(len(mask)):
          mask[j] = torch.from_numpy(mask[j])
          labels.append(mask[j])
        for j in range(len(image)):
          image[j] = torch.from_numpy(image[j])
          img.append(image[j])
# for patches, instead of one single image and mask, an array of images, and an array of masks will be passed.
        return img, labels   
            
    def __len__(self):
        return len(self.mask_file_list)
  
def blockshaped(arr, nrows, ncols):
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols).swapaxes(1,2).reshape(-1, nrows, ncols))


image_dir = 'INbreast/test_mass/mass'
mask_dir = 'INbreast/test_mass/mask'
label = 'mass'
traindir = '/gdrive/My Drive/'
testdir = '/gdrive/My Drive/'

batch_size = 1
workers = 2

train_data = MyDataSet(traindir, image_dir, mask_dir, label, img_transform=None,
                       mask_transform=None)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=workers,
                                           pin_memory=True)
test_data = MyDataSet(testdir, image_dir, mask_dir, label, img_transform=None,
                      mask_transform=None)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=workers,
                                          pin_memory=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = UNet(in_channels=1,
             out_channels=64,
             n_class=2,
             kernel_size=3,
             padding=1,
             stride=1).to(device)

criterion = nn.BCELoss()
# optimizer = optim.SGD(model.parameters(), lr=1e-0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
dataloader = train_loader
test = test_loader
listt = []
epochs = 10

for epochs in range(5):
    print("EPOCH =================================================================================>", epochs)
    running_loss = 0.0
    i = 0
    for a, b in train_loader: # for X, y in dataloader:
      j = 0
      if i < 10:
        for j in range(len(a)):
          X = a[j].float()
          y = b[j].float()
          X = X.to(device)  # [N, 1, H, W]

          y = y.to(device, dtype=torch.int64)  # [N, H, W] with class indices (0, 1)
          X = X.unsqueeze(0)
          optimizer.zero_grad()
          print(X.shape)
          print(y.squeeze(0).shape, y.min(), y.max())
          prediction = model(X)  # [N, 2, H, W]
          print(prediction.shape, prediction.min(), prediction.max())
          print(prediction.shape, y.shape, "!")
          loss = F.cross_entropy(prediction, y)
          loss.backward()
          optimizer.step()
          unique, counts = np.unique(y.byte().cpu().numpy(), return_counts=True)
          dict(zip(unique, counts))
          print("target:")
          print(dict(zip(unique, counts)))
          unique, counts = np.unique(prediction[0, 0, :, :].byte().cpu().numpy(), return_counts=True)
          dict(zip(unique, counts))
          print("prediction:")
          print(dict(zip(unique, counts)))
          unique, counts = np.unique(prediction[0, 1, :, :].byte().cpu().numpy(), return_counts=True)
          print(dict(zip(unique, counts)))
          # print statistics
          running_loss += loss.item()
          listt.append(running_loss / 2000)
          scheduler.step(running_loss)
          running_loss = 0.0
      i = i + 1
    scheduler.step(running_loss)
plt.plot(listt)
plt.show()
print('Finished Training')
