import os
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import customDataFolder
import torch.nn as nn
import torch.nn.functional as F
from vgg_perceptual_loss import VGGPerceptualLoss
from tqdm import tqdm
from tabulate import tabulate

a = np.array([[3.345324], [4.51]])
#print(tabulate(a, floatfmt=".4f"))
table = tabulate(a, floatfmt=".2f", tablefmt="pipe")
print(table)
b = 12.943413
print("{:.3f}".format(b))