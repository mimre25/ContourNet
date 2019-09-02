import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


def weights_init_normal(m):
  """
  Initializes the weights for the model
  :param m: the model
  """
  classname = m.__class__.__name__
  if classname.find("Conv") != -1:
    init.normal_(m.weight.data, 0.0, 0.01)
  elif classname.find("Linear") != -1:
    init.normal_(m.weight.data, 0.0, 0.01)
  elif classname.find("BatchNorm") != -1:
    init.normal_(m.weight.data, 1.0, 0.02)
    init.constant_(m.bias.data, 0.0)


class ContourNet(nn.Module):
  """
  ContourNet model, PyTorch implementation
  """
  def __init__(self):
    super(ContourNet, self).__init__()
    self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
    self.conv2 = nn.Conv2d(64, 256, 3, padding=1)
    self.conv3 = nn.Conv2d(256, 512, 3, padding=1)
    self.conv4 = nn.Conv2d(512, 1024, 3, padding=1)
    self.conv5 = nn.Conv2d(1024, 1024, 3, padding=1)
    self.batch1 = nn.BatchNorm2d(64)
    self.batch2 = nn.BatchNorm2d(256)
    self.batch3 = nn.BatchNorm2d(512)
    self.batch4 = nn.BatchNorm2d(1024)
    self.deconv1 = nn.ConvTranspose2d(320, 1, 3, padding=1)
    self.deconv2 = nn.ConvTranspose2d(768, 64, 3, padding=1)
    self.deconv3 = nn.ConvTranspose2d(1536, 256, 3, padding=1)
    self.deconv4 = nn.ConvTranspose2d(2048, 512, 3, padding=1)
    self.deconv5 = nn.ConvTranspose2d(1024, 1024, 3, padding=1)
    self.ac = nn.LeakyReLU(0.2)

  def forward(self, x):
    x1 = self.ac(self.batch1(self.conv1(x)))
    x2 = self.ac(self.batch2(self.conv2(x1)))
    x3 = self.ac(self.batch3(self.conv3(x2)))
    x4 = self.ac(self.batch4(self.conv4(x3)))
    x5 = self.ac(self.batch4(self.conv5(x4)))
    dx5 = F.relu(self.batch4(self.deconv5(x5)))

    x = torch.cat([x5, dx5], 1)

    dx4 = F.relu(self.batch3(self.deconv4(x)))
    x = torch.cat([x4, dx4], 1)
    dx3 = F.relu(self.batch2(self.deconv3(x)))
    x = torch.cat([x3, dx3], 1)
    dx2 = F.relu(self.batch1(self.deconv2(x)))
    x = torch.cat([x2, dx2], 1)
    x = torch.sigmoid(self.deconv1(x))
    return x


