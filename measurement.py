from __future__ import division
import os
import sys
import numpy as np
import pdb
# import matplotlib.pyplot as plt
from skimage.measure import compare_ssim as ssim
# from sklearn import metrics
e = 1e-9

def sepdice(gtimg, predimg):
    overlap = np.sum(np.multiply(gtimg, predimg))
    gtall = np.sum(gtimg)
    predall = np.sum(predimg)

    d_gt = overlap/(gtall + e)
    d_pred = overlap/(predall + e)

    return d_gt, d_pred

def softdicescore(confimap, gtimg):

    tp = np.sum(np.multiply(confimap, gtimg))
    tpfp = np.sum(gtimg)
    tpfn = np.sum(confimap)

    softdice = (2 * tp) / (tpfp + tpfn + e)

    return softdice


def probabilitydistance(confimap, gtimg):

    probdistribution = np.sum(np.absolute(gtimg - confimap))
    jointdistribution = 2 * (np.sum(np.multiply(gtimg, confimap)) + e)

    pbd = 1 - probdistribution/jointdistribution

    return pbd

def mse(x, y):

    return np.linalg.norm(x - y)

def ssimf(img, img2):
    ssim_v = ssim(img, img2, data_range=img2.max() - img2.min() + e)

    return ssim_v


def corrcoef(img, img2):

    return np.corrcoef(img, img2)


def mutual_information(hgram):
    """ Mutual information for joint histogram
    """
    # Convert bins counts to probability values
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

def mutual_info(img, img2):
    hist_2d, x_edges, y_edges = np.histogram2d(img.ravel(), img2.ravel(), bins=20)

    mi = mutual_information(hist_2d)

    return mi

# get the same result with mutual_info() function if using the same bins
# different bins will result in very tiny different results
def sklearnmi(img, img2):
    c_xy = np.histogram2d(img.ravel(), img2.ravel(), bins=10)[0]
    mi = metrics.mutual_info_score(None, None, contingency=c_xy)

    return mi


