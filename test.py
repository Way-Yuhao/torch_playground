from PIL import Image
import cv2
import numpy as np
import torch
from tabulate import tabulate

cmos = cv2.imread("./data/CMOS/1/0_cmos.png", -1)
gt = cv2.imread("./data/ground_truth/1/0.hdr", -1)
spad = cv2.imread("./data/SPAD_HDR_SR/1/0_spad_bilinear.hdr", -1)

print(1)