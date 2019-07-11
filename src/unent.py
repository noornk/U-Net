# offline!!!
# Adapted from https://discuss.pytorch.org/t/unet-implementation/426

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
# import torch.optim.lr_scheduler

def img_to_numpy(img_path):
    """Reads general image path and returns np.array normalized to 1
    """
    mode_to_bits = {'1':1, 'L':8, 'P':8, 'RGB':8, 'RGBA':32, 'CMYK':32, 'YCbCr':24, 'I':16, 'F':32}
    img = Image.open(open(img_path, 'rb'))
#     bits = mode_to_bits[img.mode]
    bits = 16
    img = np.array(img, dtype=float)
    norm = 2**bits-1
    img /= norm
#     print(img.shape)
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
#     print(img.shape)
    return img
def shaped(arr):
    h, w = arr.shape
#         assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
#         assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)
    return h, w
def blockshaped(arr, nrows, ncols):
    h, w, z = arr.shape
#         assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
#         assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)
    return (arr.reshape(h//nrows, nrows, -1, ncols).swapaxes(1,2).reshape(-1, nrows, ncols))

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
#         print("relu")
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
#         self.mask_file_list = [f for f in listdir(mask_full_path) if isfile(join(mask_full_path, f))]
        self.mask_file_list = [f for f in listdir(mask_dir) if isfile(join(mask_dir, f))]
        random.shuffle(self.mask_file_list)
#         self.i = 0
#         self.fig, self.ax = plt.subplots(1,5, figsize=(10,4))  # 1 row, 2 columns
# {0.007782101167315175: 63126, 0.5019455252918288: 10, 0.9961089494163424: 2400}
        self.mapping = {
          0.007782101167315175: 0,
          0.5019455252918288: 0,
          0.9961089494163424: 1
        }
#         self.mapping = {
#           0.007782101167315175: 0,
#           1.0: 1
#         }

    def mask_to_class(self, mask):
        for k in self.mapping:
            mask[mask == k] = self.mapping[k]
        return mask
      
      
    def blockshaped(arr, nrows, ncols):
        h, w = arr.shape
#         assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
#         assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)
        return (arr.reshape(h//nrows, nrows, -1, ncols).swapaxes(1,2).reshape(-1, nrows, ncols))

    def __getitem__(self, index):
        file_name = self.mask_file_list[index]
        img_name = os.path.join(self.dataset_path, self.image_dir, file_name.replace(".png", ".dcm"))
        mask_name = os.path.join(self.dataset_path, self.mask_dir, file_name)
#         print(img_name, mask_name)
        image = dicom_to_numpy(img_name)
        mask_p = img_to_numpy(mask_name)
#         threshold = 6
#         image = (image > threshold).astype(float)
#         bits = 16
#         norm = 2**bits-1
#         mask = mask*norm
#         image = np.array(image)
#         image = np.rollaxis(image, 2, 0)
#         image = np.array(image, dtype=np.uint8)
#         if self.img_transform:
#             print("k")
        image = cv2.resize(image, (256, 256))
#             image = self.img_transform(image)
#         #         labels = np.array(mask).astype(np.uint8)
# #         mask = torch.from_numpy(np.array(mask, dtype=np.uint8))
#         mask = self.mask_to_class(mask)
#         if self.mask_transform:
        mask_p = cv2.resize(mask_p, (256, 256))
        mask_p = self.mask_to_class(mask_p)
#         a = np.ones(shape=(2, 256, 256))
#         a[0, :, :] = mask_p
#         a[1, :, :] = reverse(mask_p)
#         mask = a
#         unique, counts = np.unique(mask, return_counts=True)
#         dict(zip(unique, counts))
#         print(dict(zip(unique, counts)))
#         if self.mask_transform:
#             mask = self.mask_transform(mask)
#         image = image.to(device)  # [N, 1, H, W]
#         mask = mask.to(device, dtype=torch.int64)  # [N, H, W] with class indices (0, 1)
#         print(image.shape)
#         print(mask.shape)
#         image = image.squeeze(0)
#         mask = mask.squeeze(0)
#         print(image.shape)
#         print(mask.shape)
#         image = np.array(image)
#         mask = np.array(mask)
        image = blockshaped(image, 256, 256)
        mask = blockshaped(mask_p, 256, 256)
#         print(mask.shape)
#         print(mask[].shape)
#         print(mask.shape)
        labels = []
        img = []
        for j in range(len(mask)):
#           print(mask[j].shape, mask[j].min(), mask[j].max())
#           a = np.zeros(shape=(2, 256, 256))
#           a[0, :, :] = mask_p
#           a[1, :, :] = reverse(mask_p)
#           a = torch.from_numpy(a)
#           print(mask[j].shape, mask[j].min(), mask[j].max())
#           if self.img_transform:
#             image[j] = self.mask_transform(mask[j])
#           print(a.shape, a.min(), a.max())
          mask[j] = torch.from_numpy(mask[j])
          labels.append(mask[j])
        for j in range(len(image)):
          image[j] = torch.from_numpy(image[j])
#           if self.img_transform:
#             image[j] = self.img_transform(image[j])
#           print(image[j])
          img.append(image[j])
#           img.append(torch.from_numpy(image[j]))
#         fig, ax = plt.subplots()
#         imgplot = ax.imshow(img[0])
#         fig.savefig('foo' + str(index) + '.png')
#         print(mask.shape)
#         image = image.unsqueeze()
#         fig, ax = plt.subplots(1,5, figsize=(10,4))  # 1 row, 2 columns
#         if self.i == 0:
#         fig, ax = plt.subplots(1,5, figsize=(10,4))  # 1 row, 2 columns
#         img = image[0].byte().cpu().numpy()
#         img = Image.fromarray(img)
#         ax[self.i].imshow(img)
#         self.i = self.i + 1
# for patches, instead of one single image and mask, an array of images, and an array of masks will be passed.
        return img, labels
#         retutn image

    def __len__(self):
        return len(self.mask_file_list)


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
def dice_loss(input, target):
    smooth = 1.
    loss = 0.
    for c in range(2):
      iflat = input[:, c ].view(-1)
      tflat = target[:, c].view(-1)
      intersection = (iflat * tflat).sum()
      w = 1
      loss += w*(1 - ((2. * intersection + smooth) /(iflat.sum() + tflat.sum() + smooth)))
    return loss    
def blockshaped(arr, nrows, ncols):
    h, w = arr.shape
#         assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
#         assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)
    return (arr.reshape(h//nrows, nrows, -1, ncols).swapaxes(1,2).reshape(-1, nrows, ncols))


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

transform_pipeline = transforms.Compose([
#     transforms.RandomSizedCrop(224),
    transforms.ToPILImage(mode=None),
#     transforms.Scale((1024, 1024)),
    transforms.RandomHorizontalFlip(),
#     transforms.RandomGrayscale(),
    transforms.ToTensor(),
#     normalize
    transforms.Normalize((0.5,), (0.5,))
])
transform_pipeline_mask = transforms.Compose([
#     transforms.RandomSizedCrop(224),
    transforms.ToPILImage(mode=None),
#     transforms.Scale((1024, 1024)),
    transforms.RandomHorizontalFlip(),
#     transforms.RandomGrayscale(),
    transforms.ToTensor(),
    # normalize
    #             transforms.Normalize((0.5,), (0.5,))
])
image_dir = 'INbreast/test_mass/mass'
mask_dir = 'INbreast/test_mass/mask'
label = 'mass'
traindir = '/gdrive/My Drive/'
testdir = '/gdrive/My Drive/'
# image_dir = 'mass/'
# mask_dir = 'mask/'
# label = 'mass'
# traindir = '/gdrive/My Drive/INbreast/July3rd/'
# testdir = '/gdrive/My Drive/INbreast/test_mass/'
batch_size = 1
workers = 2

train_data = MyDataSet(traindir, image_dir, mask_dir, label, img_transform=None,
                       mask_transform=None)

# train_data = MyDataSet(traindir, image_dir, mask_dir, label, img_transform=transform_pipeline,
#                        mask_transform=transform_pipeline_mask)

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
# model = UNet(n_classes=2, padding=True, up_mode='upsample').to(device)
# model = UNet(in_channels=3, n_classes=1, padding=True, up_mode='upsample').to(device)
criterion = nn.BCELoss()
# optimizer = optim.SGD(model.parameters(), lr=1e-0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
# optimizer = optim.Adam(net.parameters(), lr=0.001)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optim, 'min')
dataloader = train_loader
test = test_loader
listt = []
epochs = 10


def mask_to_class(mask):
    mapping = {
        510: 0,
        65280: 1
    }
    for k in mapping:
        mask[mask == k] = mapping[k]
    #         print("1")
    return mask


for epochs in range(5):
    print("EPOCH =================================================================================>", epochs)
    running_loss = 0.0
#     fig, ax = plt.subplots(1,5, figsize=(10,4))
    i = 0
    for a, b in train_loader: # for X, y in dataloader:
#     a = blockshaped(img, 128, 128)
      j = 0
#       print(a)
      if i < 10:
        for j in range(len(a)):
          X = a[j].float()
          y = b[j].float()
          X = X.to(device)  # [N, 1, H, W]
#           y = y.to(device)
#           print(y.shape)
          y = y.to(device, dtype=torch.int64)  # [N, H, W] with class indices (0, 1)
  #         unique, counts = np.unique(y.byte().cpu().numpy(), return_counts=True)
  #         dict(zip(unique, counts))
  #         if len(dict(zip(unique, counts))) > 1:
  #         y = y.squeeze(0)
  #         print(y[0].shape)
  #         img = y.argmax(0).byte().cpu().numpy()
  #         img = Image.fromarray(img)
  #         ax[j].imshow(img)
  #         plt.figure()
  #         plt.imshow(y[0, :, :])
  #         i = i + 1
  #         y = y.squeeze(1)
  #         print(X.shape)
          X = X.unsqueeze(0)
  #         print(X.shape)
  #         fig, ax = plt.subplots()
  #         imgplot = ax.imshow(img[0])
  #         fig.savefig('foo' + str(index) + '.png')
  #         print(X.shape)
          optimizer.zero_grad()
          print(X.shape)
          print(y.squeeze(0).shape, y.min(), y.max())
          prediction = model(X)  # [N, 2, H, W]
          print(prediction.shape, prediction.min(), prediction.max())
  #         print(model.parameters)
  #         print(torch.max(y, 1))
  #         print(prediction.max())
  #         print(y.max())
  #         print(y.shape)
  #         print("y:", y.shape)
  #         print("prediction:", prediction.shape)
#           criterion = nn.NLLLoss()
#           loss = criterion(prediction, y)
          print(prediction.shape, y.shape, "!")
          loss = F.cross_entropy(prediction, y)
  #         loss = dice_loss(prediction, y.unsqueeze(0))
  #         optimizer.zero_grad()
          loss.backward()
          optimizer.step()
  #         prediction = prediction.squeeze(0)
          unique, counts = np.unique(y.byte().cpu().numpy(), return_counts=True)
          dict(zip(unique, counts))
          print("target:")
          print(dict(zip(unique, counts)))
          unique, counts = np.unique(prediction[0, 0, :, :].byte().cpu().numpy(), return_counts=True)
          dict(zip(unique, counts))
          print("prediction:")
          print(dict(zip(unique, counts)))
          unique, counts = np.unique(prediction[0, 1, :, :].byte().cpu().numpy(), return_counts=True)
  #         print("prediction:", prediction[1].shape)
          print(dict(zip(unique, counts)))
          # print statistics
          running_loss += loss.item()
          listt.append(running_loss / 2000)
  #         running_loss += loss.item()
          scheduler.step(running_loss)
          running_loss = 0.0
      i = i + 1
    scheduler.step(running_loss)
plt.plot(listt)
plt.show()
print('Finished Training')
