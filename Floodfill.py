import numpy as np
import math
from functools import reduce

from skimage.measure import find_contours

def dist(p1, p2):
  """
  squared 2d distance for 2 points
  :param p1: first point
  :param p2: second point
  :return: squqred eucledian distance
  """
  return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2


def fill(data, seed, filled=True):
  """
  first finds the contour for a given seed and then fills it if filled is turned on
  :param data: data 2d array
  :param seed: seed point (x,y)
  :param filled: whether to fill the contours or not
  :returns a mask of the contour/superlevelset in shape of [1, xlim, ylim, 1] with 2 as contour value
  """
  xlim, ylim = data.shape
  mask = np.zeros([1, data.shape[0], data.shape[1], 1])
  stack = [seed]
  initValue = data[seed[0], seed[1]]
  sx, sy = seed
  size = 0

  contours = find_contours(data, initValue)
  if len(contours) == 0:
    return mask, 0
  minDist = 200000000
  idx = -1
  for i, c in enumerate(contours):
    for pt in c:
      if dist(pt, seed) < minDist:
        minDist = dist(pt, seed)
        idx = i
  
  #print(idx, len(contours), seed)
  for x, y in contours[idx]:
    mask[0, int(x), int(y), 0] = 2

  if filled:
    cont = sorted(contours[idx], key=lambda tup: tup[1] * 1000 + tup[0])
    numPts = len(cont)
    for i in range(numPts - 1):
      p = cont[i]
      q = cont[i + 1]
      y = int(p[1])
      if int(q[1]) == y:
        ## same y position, fill
        for x in range(int(p[0]), int(q[0])):
          mask[0, x, y, 0] = 2

  return mask, len(mask[mask == 2])


def computeContour(data, seed):
  """
  Computes a contour for the given seed in the the data
  :param data: the data to use
  :param seed: the seed position
  :return: the contour and its size
  """
  return fill(data, seed, filled=False)


def computeMask(data, seed):
  """
  Computes a mask for the given seed in the the data
  :param data: the data to use
  :param seed: the seed position
  :return: the mask and its size
  """
  return fill(data, seed, filled=True)