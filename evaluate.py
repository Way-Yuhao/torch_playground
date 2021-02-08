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

cmos_path = "./data/CMOS"
gt_path = "./data/ground_truth"
ds_size = 54


def set_device():
    if torch.cuda.is_available():
        dev = "cuda:0"
        print("using gpu")
    else:
        dev = "cpu"
        print("using cpu")


def compute_l1_loss(output, target):
    criterion = nn.L1Loss()
    l1_loss = criterion(output, target)
    return l1_loss


def compute_content_loss(output, target):
    return F.mse_loss(output, target)


def main():
    transform = transforms.Compose([transforms.ToTensor()])
    cmos_data_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(cmos_path, transform=transform),
        batch_size=ds_size, num_workers=0, shuffle=False)

    gt_data_loader = torch.utils.data.DataLoader(
        customDataFolder.ImageFolder(gt_path, transform=transform),
        batch_size=ds_size, num_workers=0, shuffle=False)

    cmos_it = iter(cmos_data_loader)
    gt_it = iter(gt_data_loader)
    cmos_data, _ = cmos_it.next()
    gt_data, _ = gt_it.next()

    cmos_data = cmos_data.to(torch.device("cuda:0"))
    gt_data = gt_data.to(torch.device("cuda:0"))

    l1_loss = compute_l1_loss(cmos_data, gt_data)
    content_loss = compute_content_loss(cmos_data, gt_data)
    print("content loss = ", content_loss)

if __name__ == "__main__":
    main()

