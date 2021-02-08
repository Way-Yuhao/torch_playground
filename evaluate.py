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

cmos_path = "./data/CMOS"
gt_path = "./data/ground_truth"
_mini_batch_size = 16


def set_device():
    if torch.cuda.is_available():
        dev = "cuda:0"
        print("using gpu")
    else:
        dev = "cpu"
        print("using cpu")
    return dev


def set_transform():
    return transforms.Compose([transforms.ToTensor()])


def load_data(ftype, path, transform):
    if ftype == "png":
        data_loader = torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder(path, transform=transform),
            batch_size=_mini_batch_size, num_workers=0, shuffle=False)
    elif ftype == "hdr":
        data_loader = torch.utils.data.DataLoader(
            customDataFolder.ImageFolder(path, transform=transform),
            batch_size=_mini_batch_size, num_workers=0, shuffle=False)
    else:
        raise TypeError("undefined behavior. Expected input images to be .hdr or .png")
    return data_loader


def compute_l1_loss(output, target):
    criterion = nn.L1Loss()
    l1_loss = criterion(output, target)
    return l1_loss


def compute_content_loss(output, target):
    return F.mse_loss(output, target)


def compute_vgg_loss(output, target):
    net = VGGPerceptualLoss()
    net.to(torch.device("cuda:0"))
    loss = net(output, target)
    return loss


def main():
    set_device()
    torch.cuda.empty_cache()
    transform = set_transform()

    ds_names = np.array(["cmos ldr"])
    loss_func_names = ["l1 loss", "content loss", "vgg16 loss"]

    # initialize data loaders
    cmos_data_loader = load_data("png", cmos_path, transform)
    gt_data_loader = load_data("hdr", gt_path, transform)

    # check if data sets match in size
    assert(len(cmos_data_loader.dataset) == len(gt_data_loader.dataset))
    num_mini_batches = len(cmos_data_loader)
    ds_size = len(cmos_data_loader.dataset)
    print("size of dataset = {}, mini-batch size = {}".format(ds_size, _mini_batch_size))

    cmos_it = iter(cmos_data_loader)
    gt_it = iter(gt_data_loader)

    # computing loss
    l1_loss, content_loss, vgg_loss = 0, 0, 0
    metrics = None
    for i in tqdm(range(num_mini_batches)):
        cmos_data, _ = cmos_it.next()
        gt_data, _ = gt_it.next()
        cmos_data = cmos_data.to(torch.device("cuda:0"))
        gt_data = gt_data.to(torch.device("cuda:0"))
        cur_mini_batch_size = len(cmos_data)

        l1_loss += compute_l1_loss(cmos_data, gt_data) * cur_mini_batch_size
        content_loss += compute_content_loss(cmos_data, gt_data) * cur_mini_batch_size
        vgg_loss += compute_vgg_loss(cmos_data, gt_data) * cur_mini_batch_size
    l1_loss /= ds_size
    content_loss /= ds_size
    vgg_loss /= ds_size
    cur_metrics = np.array([l1_loss, content_loss, vgg_loss])

    metrics = np.array([cur_metrics]) if metrics is None else np.vstack((metrics, cur_metrics))
    print(metrics)
    print(metrics.shape)

    # setting up table to print
    table_content = np.hstack((ds_names.reshape(-1, 1), metrics))
    table = tabulate(table_content, loss_func_names, tablefmt="pretty")
    print(table)

    # print("l1 loss = ", l1_loss.item())
    # print("content loss = ", content_loss.item())
    # print("vgg loss = ", vgg_loss.item())

    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()

