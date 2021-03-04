import numpy as np
import torch
import torchvision.transforms.functional as TF
import cv2

a = np.array([1, 2, 3])
np.savetxt("test.csv", a, delimiter=",")