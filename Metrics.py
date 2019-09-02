import sklearn.metrics as skm
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from skimage import measure


def compareMasks(gt, masks):
  """
  compares ground truth with a mask and compues f1 and accuracy
  :param gt: ground truth mask
  :param masks: computed mask
  :returns a tuple containing the f1 score and the accuracy
  """
  masks[masks>0] = 1
  tp = 0
  fp = 0
  tn = 0
  fn = 0
  for i in range(gt.shape[0]):
    for j in range(gt.shape[1]):
      l = gt[i][j]
      p = masks[i][j]
      if l == 1:
        if p == l:
          tp = tp + 1
        else:
          fn = fn + 1
      else:
        if p == l:
          tn = tn + 1
        else:
          fp = fp + 1
  prec = tp / (tp + fp + 1e-30)
  rec = tp / (tp + fn + 1e-30)
  acc = (tp + tn) / (tp + tn + fp + fn + 1e-30)
  f1 = 2 * (prec * rec) / (prec + rec + 1e-30)
  print("LOG: TP, TN, FP, FN, PREC, REC, ACC, F1".format(tp, tn, fp, fn, prec,rec,acc,f1))
  print("LOG: {:.3f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}".format(tp, tn, fp, fn, prec,rec,acc,f1))
  return (f1, acc)


def hausdorff(s1, s2):
  """
  computes hausdorff distance of the contours
  :param s1: grounds truth image
  :param s2: predicted image
  :returns the hausdorff distance
  """
  cs1 = measure.find_contours(s1, 0.9)
  cs2 = measure.find_contours(s2, 0.9)
  dis = [100000 for i in cs1]
  for i in range(len(cs1)):
    c1 = cs1[i]
    for c2 in cs2:
      dh = max(directed_hausdorff(c1, c2)[0], directed_hausdorff(c2, c1)[0])
      if dh < dis[i]:
        dis[i] = dh
  return dis


def jaccard(s1, s2, overlap):
  """
  computes the jaccard similarity between the two sets
  :param s1: ground truth 1s
  :param s2: predicted 1s
  :param overlap: number of overlapping pixels
  :returns the jaccard similarity between s1 and s2
  """
  return overlap / (s1 + s2 - overlap)


def diceCoeff_(s1, s2, overlap):
  """
  computes the dice coefficient between s1 and s2 that have an overlap
  :param s1: first set
  :param s2: second set
  :param overlap: number of overlaps
  :return: the dice coefficient
  """
  return (2 * overlap) / (s1 + s2)


def computeOverlap(c1, c2):
  """
  computes the overlap of two contours/masks
  :param c1: ground truth
  :param c2: predicted binary mask
  :returns a tuple containing the dice coefficient and the jaccard similarity of the two contours
  """
  d = c1 - c2

  p1s = np.where(c1 == 1)
  p2s = np.where(c2 > 0)
  x1 = p1s[0]
  y1 = p1s[1]
  x2 = p2s[0]
  y2 = p2s[1]
  overlaps = 0
  for i in range(len(x1)):
    x = x1[i]
    y = y1[i]
    for j in range(len(x2)):
      x_ = x2[j]
      y_ = y2[j]
      if x == x_ and y == y_:
        overlaps = overlaps + 1
  if len(x1) == 0 and len(x2) == 0:
        return 0
  # print("overlap", overlaps)
  return diceCoeff_(len(x1), len(x2), overlaps), jaccard(len(x1), len(x2), overlaps)
