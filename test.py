from PIL import Image
import cv2
import numpy as np
import torch


print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())