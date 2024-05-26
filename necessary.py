import matplotlib.pyplot as plt 
import numpy as np 
import torch
import segmentation_models_pytorch as smp
import albumentations as A
import cv2
from torch.utils.data import Dataset
from torch import nn
from segmentation_models_pytorch.losses import DiceLoss as DLoss


class helper():

  def __init__(self):
    pass

  #Test out .show_image(), written on 26 May 2024
  def show_image(self, image, mask = None, image_from_model = None):
    if mask == None:
        if image_from_model == None:

            f, ax1 = plt.subplots(1, 1, figsize=(10,5))

            ax1.set_title('IMAGE')
            ax1.imshow(image.permute(1,2,0).squeeze(),cmap = 'gray')

        elif image_from_model != None :

            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

            ax1.set_title('IMAGE')
            ax1.imshow(image.permute(1,2,0).squeeze(),cmap = 'gray')

            ax2.set_title('MODEL OUTPUT')
            ax2.imshow(image_from_model.permute(1,2,0).squeeze(),cmap = 'gray')

    else:
        if image_from_model == None:

            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

            ax1.set_title('IMAGE')
            ax1.imshow(image.permute(1,2,0).squeeze(),cmap = 'gray')

            ax2.set_title('GROUND TRUTH')
            ax2.imshow(mask.permute(1,2,0).squeeze(),cmap = 'gray')

        elif image_from_model != None :

            f, (ax1, ax2,ax3) = plt.subplots(1, 3, figsize=(15,5))

            ax1.set_title('IMAGE')
            ax1.imshow(image.permute(1,2,0).squeeze(),cmap = 'gray')

            ax2.set_title('GROUND TRUTH')
            ax2.imshow(mask.permute(1,2,0).squeeze(),cmap = 'gray')

            ax3.set_title('MODEL OUTPUT')
            ax3.imshow(image_from_model.permute(1,2,0).squeeze(),cmap = 'gray')
        #return 0


class albumentation_addon():

  def __init__(self):
    #self.IMAGE_SIZE = IMAGE_SIZE
    pass

  def get_valid_augs(self, IMAGE_SIZE = 320):
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),]
        , is_check_shapes=False)

class SegmentationModel(nn.Module):

  def __init__(self, ENCODER, WEIGHTS):
    super(SegmentationModel, self).__init__()

    self.arc = smp.Unet(
        encoder_name = ENCODER,
        encoder_weights = WEIGHTS,
        in_channels = 3,
        classes = 1,
        activation = None)

  def forward(self, images, masks = None):
    logits = self.arc(images)

    if masks != None:
      loss1 = DLoss(mode = 'binary')(logits, masks)
      loss2 = nn.BCEWithLogitsLoss()(logits, masks)
      return logits, loss1 + loss2

    return logits


class SegmentationDataset(Dataset):

  def __init__(self, df, augumentations):
    self.df = df
    self.augumentations = augumentations

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    row = self.df.iloc[idx]

    image_path = row.images
    mask_path = row.masks

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = np.expand_dims(mask, axis = -1)

    if self.augumentations:
      data = self.augumentations(image = image, mask = mask)
      image = data['image']
      mask = data['mask']

    image = np.transpose(image, (2,0,1)).astype(np.float32)
    mask = np.transpose(mask, (2,0,1)).astype(np.float32)

    image = torch.Tensor(image)/255.0
    mask = torch.round(torch.Tensor(mask)/255.0)

    return image, mask

class testing():
  
  def __init__(self):
    pass

  def hello():
    print("hello from necessary")

