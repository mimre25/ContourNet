import argparse
from timeit import default_timer as timer

import numpy as np
import torch
from skimage.measure import compare_psnr, compare_ssim, compare_mse, compare_nrmse, label
import matplotlib.pyplot as plt

from DataHandling import getDataOnly, getdata, discretizeData
from globals import PATH
from Vis import showDataWithNegAndPos
from Floodfill import computeContour, computeMask
from Metrics import computeOverlap, hausdorff, compareMasks


def computeMetric(blob1, blob2):
  """
  computes a metric for the overlap
  :param blob1: ground truth blob
  :param blob2: the candidate
  :return: the score for the overlap
  """
  return computeOverlap(blob1, blob2)[0]


def computeBestMatch(masks, blob, seeds):
  """
  computes the best match out of the array of masks with the given blob
  :param masks: an array of masks each [400,400] with 1 being the candidate and 0 the background
  :param blob: the blob in shape [400,400] with 1 being the blob and 0 the background
  :param seeds: the seeds corresponding to the blobs as 2d coordinates
  :return: the best match mask [400,400] in shape, and the corresponding seed
  """
  top = -1
  idx = -1
  for i, mask in enumerate(masks):
    ### compute metric to find minimum
    val = computeMetric(blob, mask)
    if val > top:
      top = val
      idx = i

  return masks[idx], seeds[idx]


def computeContours(data, labels):
  """
  computes the isocontours/superlevelsets for the detected blobs in the labels array
  :param data: the original data
  :param labels: the connected component of the blobs labeled
  :return: a single mask containing all the blobs, and a mask containing all the contours
  """
  data[data != data] = -1000
  mask = np.zeros([400, 400])
  contours = np.zeros([400, 400])
  for i in range(1, np.max(labels) + 1):
    ### find points on the contour of blob
    pts = tuple(zip(*np.where(labels == i)))
    blob = np.copy(labels)
    blob[blob != i] = 0
    blob[blob == i] = 1

    ###compute mask
    masks = []

    for seed in pts:
      masks.append(computeMask(data, seed)[0].reshape([400, 400]))

    ###compute best match
    if len(masks) > 0:
      m, s = computeBestMatch(masks, blob, pts)
      mask += m
      contours += computeContour(data, s)[0].reshape([400, 400])

  return mask, contours


def generatePrediction(model, data):
  '''
  Uses the model to generate the blob predictions
  :param model: the model to use
  :param data: the input data point (discretized)
  :return: the prediction
  '''
  gd = torch.FloatTensor(np.reshape(data, [1, 1, 400, 400]))
  pred = np.zeros([400, 400])
  for i in range(0, 3):
    si = i * 128
    ei = 400 if i == 3 else (i + 1) * 128
    for j in range(1, 3):
      sj = j * 128
      ej = 400 if j == 3 else (j + 1) * 128
      c = torch.FloatTensor(gd[:, :, si:ei, sj:ej]).cuda()
      pred_ = model(c)
      pred[si:ei, sj:ej] = pred_[0, 0].cpu().detach().numpy()
      print(i, j, "min", pred[si:ei, sj:ej].min(), "max", pred[si:ei, sj:ej].max(), "mean",
            pred[si:ei, sj:ej].mean())
  return pred


def createResults(model, data, originalData, timesteps, slices, args):
  '''
  creates the results by feeding the data through the model and saving the output as image
  :param model: the model to use
  :param data: the discretized data (for the model)
  :param originalData: the original data (for displaying)
  :param timesteps: the timesteps used
  :param slices: the slices used
  :param args: the args container
  '''

  minSize = 15
  numSlices = len(slices)
  for k, ts in enumerate(timesteps):
    for s in slices:
      testId = k * numSlices + s - 1
      # testId = 0
      gd = torch.FloatTensor(np.reshape(data[testId], [1, 1, 400, 400]))
      start = timer()
      pred = generatePrediction(model, gd)
      end = timer()
      print("LOG: prediction", end - start, "seconds")

      filter = 0.5
      pred[pred < filter] = 0
      pred[pred > 0] = 1
      labels = filterPrediction(pred, minSize)
      originalData[originalData != originalData] = -1000
      start = timer()
      pred2, conts = computeContours(originalData[testId].copy(), labels)
      end = timer()
      print("LOG: contours", end - start, "seconds")

      # storing images
      title = str(ts) + "-" + str(s)
      showDataWithNegAndPos(originalData[testId].copy(), np.zeros([400, 400]), conts, title=title,
                            fname=PATH + "/" + title + "-contour.png", save=True)
      showDataWithNegAndPos(originalData[testId].copy(), np.zeros([400, 400]), pred2, title=title,
                            fname=PATH + "/" + title + ".png", save=True)

  print("LOG: total", len(slices) * len(timesteps))


def filterPrediction(prediction, minSize):
  '''
  filters a prediction by removing blobs below a given minimum Size
  :param prediction: the blob mask
  :param minSize: the minimum size
  :return: the filtered prediction
  '''
  labels, num = label(prediction, return_num=True)
  for n in range(num + 1):
    if len(labels[labels == n]) < minSize:
      labels[labels == n] = 0
  return labels


def test(model, data, mask, originalData, meta, contours, args):
  dices = []
  minSize = 15
  startF = 0
  endF = 1
  totalDice = [0 for i in range(startF, endF)]

  for testId in range(len(data)):
    if args.cuda:
      gd = torch.FloatTensor(np.reshape(data[testId], [1, 1, 400, 400]))
      contour = np.reshape(mask[testId], [1, 1, 400, 400])
      print("mask", mask.min(), mask.max())
      pred_ = generatePrediction(model, gd)

      for i in range(startF, endF):
        pred = pred_.copy()
        filter = 0.65  # + 0.010 * i
        pred[pred < filter] = 0
        pred[pred > 0] = 1

        labels = filterPrediction(pred, minSize)
        start = timer()
        pred2, conts = computeContours(originalData[testId], labels)
        end = timer()
        print("finding isocontours too ", end - start, "seconds")
        print("fitler", filter, len(pred[pred > 0]))

        dice, jaccard = computeOverlap(mask[testId], pred2)
        hausdorffs = hausdorff(mask[testId], pred2)
        f1, acc = compareMasks(mask[testId], pred2)
        print("AVG hausdorff: {}".format(sum(hausdorffs) / len(hausdorffs)))
        print("Acc: {}".format(acc))
        print("F1: {}".format(f1))
        print("Dice: {}".format(dice))
        print("Jaccard: {}".format(jaccard))
        print("PSNR: {}".format(compare_psnr(mask[testId], pred2, data_range=1)))
        print("SSIM: {}".format(compare_ssim(mask[testId], pred2, data_range=1)))
        print("MSE: {}".format(compare_mse(mask[testId], pred2)))
        print("NRMSE: {}".format(compare_nrmse(mask[testId], pred2)))
        dices.append(dice)
        totalDice[i - startF] += dice

  dices = np.array(dices)
  print(np.average(dices), np.min(dices), np.max(dices))
  print(totalDice)
  for i, d in enumerate(totalDice):
    print("filter", 0.6 + (i + startF) / 100, "dice", d / len(data))

  print("end")


def inference(model, args):
  '''
  runs the inference
  :param model: the model to use
  :param args: the args container
  '''
  timesteps = [i * 5 for i in range(1, 185 // 5)]
  slices = [j for j in range(1, 16)]
  data = getDataOnly(timesteps, slices)
  originalData = data.copy()
  for i in range(len(timesteps) * len(slices)):
    data[i] = discretizeData(data[i])
  createResults(model, data, originalData, timesteps, slices, args)


def runTest(model, args):
  timesteps = [i for i in range(55, 66)]
  slices = [j for j in range(1, 16)]

  data, mask, meta, contours = getdata(timesteps, slices)
  print(data.shape, mask.shape)

  originalData = data.copy()
  data = discretizeData(data)
  trainingSize = int(len(timesteps) * 0.9)
  testData = data[trainingSize:]
  testMask = mask[trainingSize:]
  originalData = originalData[trainingSize:]
  metaData = meta[trainingSize:]
  test(model, testData, testMask, originalData, metaData, contours, args)


def main():
  parser = argparse.ArgumentParser(
    description='PyTorch Implementation of the paper: "A Deep Learning Approach for Blob detection in Plasma Fusion')
  parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
  parser.add_argument('modelname', type=str, default='contournet', help='name of the model to test')
  parser.add_argument('--runInference', action='store_true', default=False, help='Run inference instead of test')

  args = parser.parse_args()
  print("Using cuda:", not args.no_cuda)
  print("Cuda available", torch.cuda.is_available())
  args.cuda = not args.no_cuda and torch.cuda.is_available()

  model = torch.load(PATH + '/' + args.modelname + '.pth')

  if args.runInference:
    inference(model, args)
  else:
    runTest(model, args)

if __name__ == "__main__":
  main()
