from PIL import Image
import cv2
import numpy as np
import torch
from tabulate import tabulate

a = np.array([[1, 2, 3], [4, 5, 6]])
data_list = np.array(["cmos data", "spad data"])
a = np.hstack((data_list.reshape(-1, 1), a))
headers = ["l1 loss", "content loss", "vgg loss"]
table = tabulate(a, headers, tablefmt="pretty")
print(table)