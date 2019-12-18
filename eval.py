from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

np.set_printoptions(precision=2)


def plot_hist(outfp, kendall_list):
  plt.hist(kendall_list, bins=5)
  plt.xlim(0, 1)
  plt.savefig(outfp)
  plt.close()


def eval_testing(test_labels, test_prediction, test_attentions, epoch=0):
  num_bags = len(test_labels)

  print("# Evaluate the Attention Value")

  positive_bags, corrected_bags = 0, 0
  corrected_instance = 0
  instance_TP = []

  for i in range(num_bags):
    labels, prediction = test_labels[i][0].astype(float), test_prediction[i].squeeze().astype(float)
    attentions = test_attentions[i][0]
    if prediction:
      positive_bags += 1

      if max(labels) == prediction:
        corrected_bags += 1
        tau, _ = kendalltau(labels, attentions)

        if is_max_accurate(labels, attentions):
          corrected_instance += 1

        n_positive = np.sum(labels).astype(int)
        indices = np.argsort(attentions)[::-1][:n_positive]
        tp = np.mean(labels[indices])
        instance_TP.append(tp)

  print("Number of Positive Bags: %d (%.2f)" % (positive_bags, positive_bags * 100.0 / num_bags))
  print("Number of Corrected Bags: %d (%.2f)" % (corrected_bags, corrected_bags * 100.0 / positive_bags))
  print("Number of Max Corrected: %d (%.2f)" % (corrected_instance, corrected_instance * 100.0 / corrected_bags))
  print("Mean of True Positive: %.2f" % (sum(instance_TP) / corrected_bags))

  # draw
  outfp = "./figs/epoch_%d.png" % epoch
  plot_hist(outfp, instance_TP)


def is_max_accurate(labels, attention):
  index = np.argmax(attention)
  if labels[index]:
    return True
  else:
    return False
