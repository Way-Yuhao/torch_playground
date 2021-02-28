from matplotlib import pyplot as plt
import cv2
import numpy as np


def color_hist(hdr, title_name):
    fig = plt.gcf()
    fig.set_size_inches(18.5, 6.5)
    color = ('b', 'g', 'r')

    # ldr
    # plt.subplot(1, 2, 1)
    # for channel, col in enumerate(color):
    #     histr = cv2.calcHist([ldr], [channel], None, [2**16], [0, 2**16])
    #     plt.plot(histr, color=col)
    #     # plt.xlim([0, 2**14])
    #     plt.ylim([0, 30000])
    # plt.title("Histogram of simulated CMOS data")
    # plt.xlabel("pixel values")
    # plt.ylabel("pixel counts")
    # # plt.show()

    # hdr
    # plt.subplot(1, 2, 2)
    for channel, col in enumerate(color):
        histr = cv2.calcHist([hdr], [channel], None, [2**16], [0, 2**18])
        plt.plot(histr, color=col)
        # plt.xlim([0, 2**16])
        plt.ylim([0, 30000])
    plt.title(title_name)
    plt.xlabel("pixel values")
    plt.ylabel("pixel counts")
    plt.show()
    return

cmos_path = "../data/CMOS"
cmos_short_path = "../data/CMOS_short"
gt_path = "../data/ground_truth"
spad_path = "../data/SPAD_HDR_SR"
exp_brkt_path = "../data/exp_brkt"
fusion_path = "../data/fusion"

img = cv2.imread(gt_path + "/1/0.hdr", -1)
print(img.max())
print(img.mean())
print(np.median(img))
# color_hist(img, "scaled ground truth (x100000)")