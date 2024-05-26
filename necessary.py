import matplotlib.pyplot as plt 
import numpy as np 
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from torch.utils.data import Dataset
from torch import nn
from segmentation_models_pytorch.losses import DiceLoss as DLoss

class helper():

  def __init__(self):
    return 0

  def show_image(image,mask,pred_image = None):
  return 0

class albumentation_addon():

  def __init__(self):
    return 0

  def get_valid_augs():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),]
        , is_check_shapes=False)

class SegmentationModel(nn.Module):

  def __init__(self):
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

