import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_alignment(alignment, path, info=None):
  fig, ax = plt.subplots()
  im = ax.imshow(
    alignment,
    aspect='auto',
    origin='lower',
    interpolation='none')
  fig.colorbar(im, ax=ax)
  xlabel = 'Decoder timestep'
  if info is not None:
    xlabel += '\n\n' + info
  plt.xlabel(xlabel)
  plt.ylabel('Encoder timestep')
  plt.tight_layout()
  plt.savefig(path, format='png')
  plt.close()


def plot_weight(weight, path):
  fig, ax = plt.subplots()
  im = ax.imshow(
    weight,
    aspect='auto',
    origin='lower',
    interpolation='none')
  fig.colorbar(im, ax=ax)
  for i, head in enumerate(weight):
    for j, item in enumerate(head):
      plt.text(j-0.4, i-0.1, "%.2f" % item, fontsize=12, color = "r")
  plt.tight_layout()
  plt.savefig(path, format='png')
  plt.close()
