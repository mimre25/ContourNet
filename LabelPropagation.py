import argparse

import numpy as np
import time

from tqdm import tqdm

import DataHandling
from DataHandling import loadDataFromJson
from Floodfill import computeMask
from globals import *

from Metrics import computeOverlap

def generateCandidates(data, psiFilter, step, size=400):
  """
  Computes blob candidates using a step x step grid
  :param data: the data to compute candidates in
  :param psiFilter: the psiFilter. None if it shouldn't be used
  :param step: the steps size for the grid
  :param size: the size of the data
  :return: a list of blob candidates
  """
  candidates = []
  for seed in tqdm([(x, y) for x in range(0, size, step) for y in range(0, size, step)]):
    if psiFilter is None or psiFilter[seed[0], seed[1]] == 1:
      candidate = computeMask(data, seed)
      candidates.append((seed, candidate[0]))
  return candidates


def findBestMatchCandidates(masks, candidates, seeds):
  """
  finds the best seeds for overlapping contours
  :param masks: the ground truth masks
  :param candidates: the candidates
  :param seeds: the seeds of the candidates
  :return: a list of seeds for the best match candidates
  """
  numMasks = len(masks)
  overlaps = [0 for i in range(numMasks)]
  overlapSeeds = [[] for i in range(numMasks)]
  for j in range(len(masks)):
    print("computing overlap for mask " + str(j) + "/" + str(numMasks))
    mask = masks[j, :, :]
    for k in range(len(candidates)):
      candidate = candidates[k][0, :, :, 0]
      candidate[candidate > 0] = 1
      ov, _ = computeOverlap(mask, candidate)
      if ov > overlaps[j]:
        overlaps[j] = ov
        overlapSeeds[j] = seeds[k]

  return list(filter(lambda x: x != [], overlapSeeds))



def propagateLabels(hostTs, s, timesteps):
  """
  Propagates the label from the host time step at a given slice to neighboring time steps
  :param hostTs: the starting time step
  :param s: the slice to use
  :param timesteps: list of timesteps to propagate to
  """
  nx = 400
  ny = 400
  print("Propagating from", hostTs, "to", timesteps)
  path = PATH + 'plasma-fusion-data/'
  filebaseName = "norm-" + str(hostTs) + "-" + str(s)
  filename = path + filebaseName + ".dat"
  data = DataHandling.loadSingleFile(filename)

  pos, neg = loadDataFromJson(PATH + '/ground-truth/', hostTs, [s])

  for i in timesteps:
    start__ = time.time()
    filebaseName = "norm-" + str(i) + "-" + str(s)
    filename = path + filebaseName + ".dat"
    d = DataHandling.loadSingleFile(filename)
    d = np.reshape(d, [nx, ny])

    ### generate candiadates
    psi = np.reshape(np.nan_to_num(np.fromfile('psi.dat')), [400, 400])
    psiFilter = np.copy(psi)
    psiFilter[psi > 0.285] = 0
    psiFilter[psi < 0.23] = 0
    psiFilter[psiFilter > 0] = 1

    step = 10
    s_ = time.time()
    blobs = generateCandidates(d, psiFilter, step)
    print("done, took " + str(time.time() - s_) + " seconds, got " + str(len(blobs)) + " candidates")
    (seeds, candidates) = zip(*blobs)

    ### compute overlaps

    ## positives
    masks = pos[:, :, :, 1]
    overlapSeeds = findBestMatchCandidates(masks, candidates, seeds)
    DataHandling.saveJsonFile(filebaseName + '-propagated-pos.json', overlapSeeds)

    ## negatives
    masks = neg[:, :, :, 1]
    overlapSeeds = findBestMatchCandidates(masks, candidates, seeds)
    DataHandling.saveJsonFile(filebaseName + '-propagated-neg.json', overlapSeeds)

def main():
  parser = argparse.ArgumentParser(description='Script to propagate labels from one time step to another')

  parser.add_argument('hostTS', type=int, help="The host time step to use")
  parser.add_argument('slice', type=int, help="The slice to use")
  parser.add_argument('timesteps', type=int, nargs='+', help="A list of timesteps to use")

  args = parser.parse_args()

  propagateLabels(args.hostTS, args.slice, args.timesteps)


if __name__ == "__main__":
  main()
