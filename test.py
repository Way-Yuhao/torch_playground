from PIL import Image
import cv2
import numpy as np


cmos_path = "./data/CMOS/1/0_cmos.png"
gt_path = "./data/ground_truth/1/0.hdr"

with open(cmos_path, 'rb') as f:
    img1 = Image.open(f)
    # img1.show()
    im_np = np.asarray(img1)
print(im_np.shape)



img = cv2.imread(gt_path, -1)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# im_pil = Image.fromarray(img)

