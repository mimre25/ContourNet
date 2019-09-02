import functools
import math

import numpy as np
import json

import torch
from torch.utils.data import DataLoader

from globals import SAMPLE_PATH, DATA_PATH, RANDOM_SEED, MASK_PATH
from Vis import showDataWithNegAndPos
from Floodfill import computeContour, computeMask


def loadSingleJsonFile(filename, data, contour=False):
  """
  Loads a single json file and creates a mask with contours
  :param filename: the json file
  :param data: the underlying data
  :param contour: a flag to set to only load the contour (not filled) or the mask(filled)
  :return: either the 0/1 mask for the filled mask or the contour
  """

  totalMask = np.zeros([400, 400])
  with open(filename) as f:
    blobs = json.load(f)
    for b in blobs:
      mask, sz = computeContour(data, [400 - b[1] - 1, b[0]]) if contour else computeMask(data, [400 - b[1] - 1, b[0]])
      mask[mask > 0] = 1
      if 0 < sz < 100:
        mask = mask.reshape([400, 400])
        totalMask += mask

  totalMask[totalMask > 0] = 1
  return totalMask


def loadSingleData(filename):
  """
  Loads a single data file (norm/dpot) from disc and reshapes it properly
  :param filename: the file to load
  :return: the file as numpy array in shape of [400,400]
  """
  data = np.fromfile(filename).reshape([400, 400])
  data = np.flipud(data)
  return data


def loadDataFromJson(path, timestep, slices=range(1, 16), multiply=False):
  """
  Reads the data from a json file an creates the blobs/contours on the fly
  :param path: the path to the files
  :param timestep: the time step to load data for
  :param slices: the slices to use
  :param multiply: whether to use the multiply version, i.e. multiply the mask with the data instead of using it as separate channel
  :return: a tuple (positiveData, negativeData) containing data with corresponding labels
  """

  positives = []
  negatives = []
  for s in slices:
    try:
      posMask = np.zeros([400, 400])
      negMask = np.zeros([400, 400])
      fileBasename = 'norm-' + str(timestep) + '-' + str(s)
      filename = DATA_PATH + fileBasename + '.dat'
      data = loadSingleData(filename)

      #### positives
      with open(path + fileBasename + '-pos.json') as f:
        blobs = json.load(f)
        for b in blobs:
          mask, size = computeMask(data, [400 - b[1] - 1, b[0]])
          mask[mask > 0] = 1
          if 0 < size < 100:
            mask = mask.reshape([400, 400])
            posMask += mask
            if multiply:
              d = data * mask
              d = d.reshape([*d.shape, 1])
            else:
              d = np.stack([data, mask], axis=2)
            positives.append(d)

      #### negatives
      with open(path + fileBasename + '-neg.json') as f:
        blobs = json.load(f)
        for b in blobs:
          mask, size = computeMask(data, [400 - b[1] - 1, b[0]])
          mask[mask > 0] = 1
          if 0 < size < 100:
            mask = mask.reshape([400, 400])
            negMask += mask
            if multiply:
              d = data * mask
              d = d.reshape([*d.shape, 1])
            else:
              d = np.stack([data, mask], axis=2)
            negatives.append(d)
      # negatives = [negatives[-1]]
      posMask[posMask >= 1] = 2
      negMask[negMask >= 1] = 2
    except Exception as error:
      print("ERROR WITH:", fileBasename, ":")
      print(error)
      pass
      # showDataWithNegAndPos(data, posMask, negMask)
  positiveData = np.stack(positives)
  negativeData = np.stack(negatives)

  return positiveData, negativeData


def loadSingleFile(filename, nx =400, ny=400, nc=1, cropping=False):
  """
  Loads the file pointed to by filename and prepares it
  :param filename: the file to load
  :param nx: x extend
  :param ny: y extend
  :param nc: number of channels
  :param cropping: whether cropping is on
  :return: the prepared data
  """
  a = np.fromfile(filename, dtype='d')
  a = prepareDataArray(a)

  cropSize = 380 if cropping else 400
  tmp = np.reshape(a, [nx, ny])
  data = np.reshape(np.flipud(tmp), [1, cropSize, cropSize, nc])
  return data


def prepareDataArray(a):
  """
  Removes dummy values from the array that are used to flag NaN and normalizes it
  :param a: the array to prepare
  :return: the normalized array without dummy values
  """
  a = np.fromiter(map(lambda x: -1 if math.isnan(x) else x, a), 'd')
  min_ = functools.reduce(lambda s, x: x if x != -100000 and x < s else s, a, 10000)
  max_ = np.amax(a)
  data = np.fromiter(map(lambda x: x if x != -100000 else min_ - 1, a), 'd')
  normalized = (data - np.min(data)) / np.ptp(data)
  return normalized


def saveJsonFile(filename, jsonData):
  """
  saves json data file. Converts the format accordingly so that it works with JS version
  :param filename: the file name to store to
  :param jsonData: the data array
  """
  with open(filename, 'w') as f:
    tmp = []
    for s in jsonData:
      tmp.append([s[1], 400 - s[0] - 1])
    json.dump(tmp, f)


def getDataOnly(timesteps, slices):
  """
  Loads the data without labels
  :param timesteps: the timesteps to use
  :param slices: the slices
  :return: returns the data as np array in shape of (|timesteps|*|slices|,400,400)
  """
  ### read the .dat files into data arrays
  numOfSteps = len(timesteps)
  data = np.zeros((numOfSteps * len(slices), 400, 400))
  mask = np.zeros((numOfSteps * len(slices), 400, 400))
  d_ = []
  k = 0
  for i in range(numOfSteps):
    for j in slices:
      d = loadSingleData(DATA_PATH + "norm-" + str(timesteps[i]) + '-' + str(j) + '.dat')
      data[k] = d
      k += 1
  return data


def getdata(timesteps, totalslices, shuffle=True):
  """
  Loads the data for training and testing with labels
  :param timesteps: the time steps to use
  :param totalslices: the slices to use
  :param shuffle: whether to shuffle the data
  :return: a 4-tuple (data, mask, meta, contours), where data is the original data,
  mask is the blob mask, meta is the meta data of ts/slice and contour is the contour mask
  """
  ### read the .dat and .json files into data and mask arrays
  numOfSteps = int(len(timesteps) * 1)
  data = np.zeros((numOfSteps * len(totalslices), 400, 400))
  mask = np.zeros((numOfSteps * len(totalslices), 400, 400))
  contours = np.zeros((numOfSteps * len(totalslices), 400, 400))
  meta = [() for i in range(numOfSteps * len(totalslices))]
  d_ = []
  k = 0
  for i in range(numOfSteps):
    # d_.append(loadUNetData(PATH + '/ground-truth/', timesteps[i]))
    for j in totalslices:
      d = loadSingleData(DATA_PATH + "norm-" + str(timesteps[i]) + '-' + str(j) + '.dat')
      m = loadSingleJsonFile(MASK_PATH + "norm-" + str(timesteps[i]) + '-' + str(j) + '-pos.json', d)
      c = loadSingleJsonFile(MASK_PATH + "norm-" + str(timesteps[i]) + '-' + str(j) + '-pos.json', d, contour=True)
      data[k] = d
      mask[k] = m
      contours[k] = c
      meta[k] = (timesteps[i], j)
      k += 1
  if shuffle:
    np.random.seed(1234)
    np.random.shuffle(data)
    np.random.seed(1234)
    np.random.shuffle(mask)
    np.random.seed(1234)
    np.random.shuffle(meta)
    np.random.seed(1234)
    np.random.shuffle(contours)
  return data, mask, meta, contours


def crop(data, mask, numCroppings):
  """
  crops a given data array
  :param data: the data to crop
  :param mask: the mask for the data
  :param numCroppings: the number of croppings
  :return: the cropped data as list of arrays
  """
  count = 0
  d = []
  m = []
  # print(data.shape)
  while count < numCroppings:
    x = np.random.randint(0, 400 - 128)
    y = np.random.randint(150, 400 - 128)
    if np.sum(mask[x:x + 128, y:y + 128]) > 0:
      d.append(data[x:x + 128, y:y + 128])
      m.append(mask[x:x + 128, y:y + 128])
      count += 1
  return d, m


def generatetrainingdata(data, mask, timesteps, slices, numCroppings, args):
  """
  Generates the training DataLoader
  :param numCroppings: the number of croppings
  :param data: the data
  :param mask: the ground truth masks
  :param timesteps: the timesteps
  :param slices: the slices used
  :param args: the args container
  :return: a DataLoader object
  """
  kwargs = {'num_workers': 30, 'pin_memory': True} if args.cuda else {}

  ### number of training samples ###
  NumOfSteps = len(timesteps)
  data_training = np.zeros((numCroppings * NumOfSteps * len(slices), 1, 128, 128))
  mask_training = np.zeros((numCroppings * NumOfSteps * len(slices), 1, 128, 128))
  count = 0
  for i in range(0, len(data)):
    d, m = crop(data[i], mask[i], numCroppings)
    for k in range(0, len(d)):
      data_training[count][0] = d[k]
      mask_training[count][0] = m[k]
      count += 1
  data_training = torch.FloatTensor(data_training)
  mask_training = torch.FloatTensor(mask_training)
  dataset = torch.utils.data.TensorDataset(data_training, mask_training)
  train_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
  return train_loader


def discretizeData(data):
  """
  discrtizes the data into the scale 0,255
  :param data: input data
  :return: discretized data
  """
  idx = data != data
  data[idx] = -3

  data *= 255.0 / data.max()
  return np.digitize(data, [i for i in range(256)])