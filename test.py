from PIL import Image
import cv2
import numpy as np
import torch
from tabulate import tabulate
import torch.nn as nn

eps = 0.000000001

def compute_l1_loss(output, target):
    criterion = nn.L1Loss()
    l1_loss = criterion(output+eps, target+eps)
    return l1_loss.item()


def compute_l1_loss_log(output, target):

    criterion = nn.L1Loss()
    l1_loss = criterion(torch.log10(output+eps), torch.log10(target+eps))
    return l1_loss.item()

cmos = cv2.imread("./data/CMOS/1/0_cmos.png", -1)
gt = cv2.imread("./data/ground_truth/1/0.hdr", -1)
spad = cv2.imread("./data/SPAD_HDR_SR/1/0_spad_bilinear.hdr", -1)

num_pixels = cmos.size
print("size = ", num_pixels)


cmos_torch = torch.from_numpy(cmos.astype("float32"))
gt_torch = torch.from_numpy(gt.astype("float32"))
spad_torch = torch.from_numpy(spad.astype("float32"))
# np.set_printoptions(precision=3, suppress=False)

cmos_p1 = cmos + 1
cmos_p1_torch = cmos_torch + 1
resi = cmos_p1 - cmos

l1_plus1 = np.sum(np.abs(cmos - cmos_p1)) / num_pixels
l1_plus1_torch = compute_l1_loss(cmos_torch, cmos_p1_torch)
print("l1 via numpy: ", l1_plus1)
print("l1 via torch: ", l1_plus1_torch)



# l1_cmos = np.sum(np.abs(cmos - gt))
# l1_spad = np.sum(np.abs(spad - gt))
# print("numpy calculations:")
# print("l1 loss, cmos = ", l1_cmos)
# print("l1 loss, spad = ", l1_spad)
#
# l1_cmos_torch = compute_l1_loss_log(cmos_torch, gt_torch)
# l1_spad_torch = compute_l1_loss_log(spad_torch, gt_torch)
# print("pytorch calculations:")
# print("l1 loss, cmos = ", l1_cmos_torch)
# print("l1 loss, spad = ", l1_spad_torch)



