import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from ContourNetModel import ContourNet, weights_init_normal
from DataHandling import getdata, generatetrainingdata, discretizeData
from Inference import test
from globals import PATH, EPS


class DiceLoss(nn.Module):
  """
  Dice loss implementation for PyTorch
  """
  def __init__(self, smooth=1, p=2, reduction='mean'):
    super(DiceLoss, self).__init__()
    self.smooth = smooth
    self.p = p
    self.reduction = reduction

  def forward(self, predict, target):
    assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)

    num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
    den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth
    loss = 1 - num / den

    if self.reduction == 'mean':
      return loss.mean()
    elif self.reduction == 'sum':
      return loss.sum()
    elif self.reduction == 'none':
      return loss
    else:
      raise Exception('Unexpected reduction {}'.format(self.reduction))


class CrossEntropyLoss(nn.Module):
  """
  CrossEntropy loss definition for PyTorch using a penalty for one of the classes to distinguish better
  """
  def __init__(self, reduction='mean', penalty=1.0):
    super(CrossEntropyLoss, self).__init__()
    self.reduction = reduction
    self.penalty = penalty

  def forward(self, predict, target):
    assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    ones = torch.mul(target, torch.log(predict + EPS))
    zeros = torch.mul(1 - target, torch.log(1 - predict + EPS))
    loss = -1 * (self.penalty * ones + zeros)

    if self.reduction == 'mean':
      return loss.mean()
    elif self.reduction == 'sum':
      return loss.sum()
    elif self.reduction == 'none':
      return loss
    else:
      raise Exception('Unexpected reduction {}'.format(self.reduction))


def train(model, epochs, loss_type, data, mask, valiDationData, valiMask, name, originalData, timesteps, slices,
          args):
  """
  Model training loop
  :param model: the model to use
  :param epochs: the number of epochs to run
  :param loss_type: a string describing the loss to use {dice, cross-entropy, mse}
  :param data: the data to use for training
  :param mask: the masks for training
  :param valiDationData: the data for validation
  :param valiMask: the masks for validation
  :param name: the name the model is stored as
  :param originalData: the original data (for testing)
  :param timesteps: the number of time steps the data contains
  :param slices: the slices to use
  :param args: the args container
  """
  optimizer = optim.Adam(model.parameters(), lr=args.lr)
  if loss_type == 'dice':
    criterion = DiceLoss()
  elif loss_type == 'cross-entropy':
    criterion = CrossEntropyLoss(penalty=args.penalty)
  elif loss_type == 'mse':
    criterion = nn.MSELoss()
  else:
    print('Loss type unknown!')
    return 0

  for epoch in range(1, epochs + 1):
    train_loader = generatetrainingdata(data, mask, timesteps, slices, 4, args)
    vali_loader = generatetrainingdata(valiDationData, valiMask, timesteps, slices, 4, args)
    print("========================")
    print(epoch)
    x = time.time()
    loss = 0
    validationLoss = 0

    #### training
    for batch_idx, (contour, gd) in enumerate(train_loader):
      if args.cuda:
        gd = gd.cuda()
        contour = contour.cuda()

      pred = model(contour)
      optimizer.zero_grad()
      l = criterion(pred, gd)
      l.backward()
      loss += l.mean().item()
      optimizer.step()

    #### validation
    for batch_idx, (valData, valGT) in enumerate(vali_loader):
      if args.cuda:
        valData = valData.cuda()
        valGT = valGT.cuda()
      val = model(valData)
      v = criterion(val, valGT)
      v.backward()
      validationLoss += v.mean().item()

    y = time.time()
    print("Loss = " + str(loss))
    print("Validation Loss = " + str(validationLoss))
    print("Time = " + str(y - x))
    if epoch % 10 == 0:
      torch.save(model, PATH + '/' + name + '_' + str(epoch) + '.pth')
    if epoch % 100 == 0:
      test(model, data, mask, originalData, args)
  torch.save(model, PATH + '/' + name + '.pth')


def main():
  parser = argparse.ArgumentParser(
    description='PyTorch Implementation of the paper: "A Deep Learning Approach for Blob detection in Plasma Fusion')
  parser.add_argument('--lr', type=float, default=1e-6, metavar='LR', help='learning rate')
  parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
  parser.add_argument('--batch_size', type=int, default=1, metavar='N', help='input batch size for training')
  parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 100)')
  parser.add_argument('--outputName', type=str, default='contournet', help='output name for this run')
  parser.add_argument('--penalty', type=int, default='1',
                      help='penalty for cross-entropy, will be encoded into the name')

  args = parser.parse_args()
  print("Using cuda:", not args.no_cuda)
  print("Cuda available", torch.cuda.is_available())
  args.cuda = not args.no_cuda and torch.cuda.is_available()

  timesteps = [i for i in range(55, 66)]
  totalslices = [j for j in range(1, 16)]

  data, mask, _, _ = getdata(timesteps, totalslices)

  originalData = data.copy()
  data = discretizeData(data)
  trainingSize = int(len(timesteps) * 0.9)
  trainingData = data[:trainingSize]
  trainingMask = mask[:trainingSize]
  originalData = originalData[:trainingSize]

  testData = data[trainingSize:]
  testMask = mask[trainingSize:]

  model = ContourNet()
  if args.cuda:
    model.cuda()
  model.apply(weights_init_normal)
  epochs = args.epochs

  args.outputName = args.outputName + str(args.penalty)
  train(model, epochs, 'cross-entropy', trainingData, trainingMask, testData, testMask, args.outputName, originalData,
        timesteps, totalslices, args)


if __name__ == "__main__":
  main()
