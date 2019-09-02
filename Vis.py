import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt


def showDataWithNegAndPos(a, mask, mask2, save=False, fname=None, title=None):
  """
  Show a dataset with both positive and negative labeled blobs
  :param a: the data array
  :param mask: the positive mask
  :param mask2: the negative mask
  :param save: a flag to decide whether to save or just show the files
  :param fname: the file name to save the figure to, only needed if save is true
  :param title: a title for the plot
  """
  print(a.shape, mask.shape, mask2.shape)
  print("a", a.min(), a.max())
  if save:
    fig = plt.figure(figsize=(12, 11))
  min_ = a[a > -1000].min()
  max_ = a[a == a].max()
  print(min, max)
  a[a != a] = min_
  a[(a > -1000) & (a < -1)] = -0.99
  a[a > 1] = 0.99
  a[mask2 > 0] = max_ + 100
  a[mask > 0] = min_ - 100
  bad = np.ma.masked_where(a <= -1000, a)

  plt.set_cmap('Blues')
  current_cmap = mpl.cm.get_cmap()
  current_cmap.set_under(color='yellow')
  current_cmap.set_bad(color='0.9')
  current_cmap.set_over(color='red')
  print(min_, max_)
  print(a.min(), a.max())
  title = None
  if title is not None:
    plt.title(title)
  plt.axis('off')
  plt.imshow(bad, cmap=current_cmap, vmin=-1, vmax=1)
  plt.colorbar()
  if save:
    plt.savefig(fname)
  else:
    plt.show()