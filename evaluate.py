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

"""GLOBAL PARAMETERS"""
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


def init_networks():
    vgg_net = VGGPerceptualLoss()
    vgg_net.to(torch.device("cuda:0"))
    return vgg_net


def compute_l1_loss(output, target):
    criterion = nn.L1Loss()
    l1_loss = criterion(output, target)
    return l1_loss


def compute_content_loss(output, target):
    return F.mse_loss(output, target)


def compute_vgg_loss(vgg_net, output, target):
    with torch.no_grad():
        loss = vgg_net(output, target)
        return loss


def main():
    cmos_path = "./data/CMOS"
    gt_path = "./data/ground_truth"
    fusion_path = "./data/SPAD_HDR_SR"
    ds_names = np.array(["cmos ldr", "spad bilinear"])
    loss_func_names = ["l1 loss", "content loss", "vgg16 loss"]

    set_device()
    torch.cuda.empty_cache()
    transform = set_transform()
    vgg_net = init_networks()

    # initialize data loaders
    cmos_data_loader = load_data("png", cmos_path, transform)
    gt_data_loader = load_data("hdr", gt_path, transform)
    spad_bi_data_loader = load_data("hdr", fusion_path, transform)
    data_loaders = [gt_data_loader, cmos_data_loader, spad_bi_data_loader]

    # check if data sets match in size
    for i in range(1, len(data_loaders)):
        assert (len(data_loaders[i].dataset) == len(data_loaders[0].dataset))

    num_mini_batches = len(data_loaders[0])
    ds_size = len(data_loaders[0].dataset)
    print("metrics: ", loss_func_names)
    print("avaliable data sets: ", ds_names)
    print("size of dataset = {}, mini-batch size = {}".format(ds_size, _mini_batch_size))

    iterables = [iter(data_loaders[i]) for i in range(1, len(data_loaders))]

    metrics = None
    for i in range(len(iterables)):
        torch.cuda.empty_cache()
        print("evaluating data set ", ds_names[i])
        gt_it = iter(gt_data_loader)
        # computing loss
        cur_metrics = np.zeros(len(loss_func_names))
        for j in tqdm(range(num_mini_batches)):
            output_data, _ = iterables[i].next()
            gt_data, _ = gt_it.next()
            output_data = output_data.to(torch.device("cuda:0"))
            gt_data = gt_data.to(torch.device("cuda:0"))
            cur_mini_batch_size = len(output_data)

            l1_loss = compute_l1_loss(output_data, gt_data)
            content_loss = compute_content_loss(output_data, gt_data)
            vgg_loss = compute_vgg_loss(vgg_net, output_data, gt_data)
            cur_metrics = np.add(cur_metrics, cur_mini_batch_size * np.array([l1_loss, content_loss, vgg_loss]))

        cur_metrics = cur_metrics / ds_size
        metrics = np.array([cur_metrics]) if metrics is None else np.vstack((metrics, cur_metrics))

    table_content = np.hstack((ds_names.reshape(-1, 1), metrics))
    table = tabulate(table_content, loss_func_names, tablefmt="pretty")
    print(table)



    # # computing loss
    # l1_loss, content_loss, vgg_loss = 0, 0, 0
    # metrics = None
    # for i in tqdm(range(num_mini_batches)):
    #     cmos_data, _ = cmos_it.next()
    #     gt_data, _ = gt_it.next()
    #     cmos_data = cmos_data.to(torch.device("cuda:0"))
    #     gt_data = gt_data.to(torch.device("cuda:0"))
    #     cur_mini_batch_size = len(cmos_data)
    #
    #     l1_loss += compute_l1_loss(cmos_data, gt_data) * cur_mini_batch_size
    #     content_loss += compute_content_loss(cmos_data, gt_data) * cur_mini_batch_size
    #     vgg_loss += compute_vgg_loss(cmos_data, gt_data) * cur_mini_batch_size
    # l1_loss /= ds_size
    # content_loss /= ds_size
    # vgg_loss /= ds_size
    # cur_metrics = np.array([l1_loss, content_loss, vgg_loss])
    #
    # metrics = np.array([cur_metrics]) if metrics is None else np.vstack((metrics, cur_metrics))
    # print(metrics)
    # print(metrics.shape)

    # setting up table to print
    # table_content = np.hstack((ds_names.reshape(-1, 1), metrics))
    # table = tabulate(table_content, loss_func_names, tablefmt="pretty")
    # print(table)

    # print("l1 loss = ", l1_loss.item())
    # print("content loss = ", content_loss.item())
    # print("vgg loss = ", vgg_loss.item())

    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()

