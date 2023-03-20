import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import data
from skimage.filters import unsharp_mask
from skimage.exposure import exposure, equalize_hist, equalize_adapthist
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torch
import kornia
import torchvision.transforms as transforms
from torchvision.utils import save_image
import PIL
import torchvision.transforms.functional as TF

image = cv2.imread(r"C:\Users\bella\Downloads\JBG040 Data Challenge 1\pneumothorax.png")

# unsharp masking
unsharp_mask_1 = unsharp_mask(image, radius=1, amount=1)
unsharp_mask_2 = unsharp_mask(image, radius=1, amount=2)
unsharp_mask_3 = unsharp_mask(image, radius=5, amount=1)
unsharp_mask_4 = unsharp_mask(image, radius=5, amount=2)
unsharp_mask_5 = unsharp_mask(image, radius=20, amount=1)
unsharp_mask_6 = unsharp_mask(image, radius=20, amount=2)


fig, axes = plt.subplots(nrows=7, ncols=1,
                         sharex=True, sharey=True, figsize=(20, 30))
ax = axes.ravel()

ax[0].imshow(image, cmap=plt.cm.gray)
ax[0].set_title('Original image')
ax[1].imshow(unsharp_mask_1, cmap=plt.cm.gray)
ax[1].set_title('Enhanced image, radius=1, amount=1.0')
ax[2].imshow(unsharp_mask_2, cmap=plt.cm.gray)
ax[2].set_title('Enhanced image, radius=1, amount=2.0')
ax[3].imshow(unsharp_mask_3, cmap=plt.cm.gray)
ax[3].set_title('Enhanced image, radius=5, amount=1.0')
ax[4].imshow(unsharp_mask_4, cmap=plt.cm.gray)
ax[4].set_title('Enhanced image, radius=5, amount=2.0')
ax[5].imshow(unsharp_mask_5, cmap=plt.cm.gray)
ax[5].set_title('Enhanced image, radius=20, amount=1.0')
ax[6].imshow(unsharp_mask_6, cmap=plt.cm.gray)
ax[6].set_title('Enhanced image, radius=20, amount=2.0')


for a in ax:
    a.axis('off')
fig.tight_layout()
plt.show()

print("the result findings suggest when parameter is set to radius=5 and amount=2.0 gives the best result")


# unsharp masking + histogram equalization
unsharp_mask_1 = unsharp_mask(image, radius=1, amount=1)
equalized_image_1 = exposure.equalize_hist(unsharp_mask_1)
unsharp_mask_2 = unsharp_mask(image, radius=1, amount=2)
equalized_image_2 = exposure.equalize_hist(unsharp_mask_2)
unsharp_mask_3 = unsharp_mask(image, radius=5, amount=1)
equalized_image_3 = exposure.equalize_hist(unsharp_mask_3)
unsharp_mask_4 = unsharp_mask(image, radius=5, amount=2)
equalized_image_4 = exposure.equalize_hist(unsharp_mask_4)
unsharp_mask_5 = unsharp_mask(image, radius=20, amount=1)
equalized_image_5 = exposure.equalize_hist(unsharp_mask_5)
unsharp_mask_6 = unsharp_mask(image, radius=20, amount=2)
equalized_image_6 = exposure.equalize_hist(unsharp_mask_6)

fig, axes = plt.subplots(nrows=7, ncols=1,
                         sharex=True, sharey=True, figsize=(50, 50))
ax = axes.ravel()

ax[0].imshow(image, cmap=plt.cm.gray)
ax[0].set_title('Original image')
ax[1].imshow(equalized_image_1, cmap=plt.cm.gray)
ax[1].set_title('Enhanced image, radius=1, amount=1.0')
ax[2].imshow(equalized_image_2, cmap=plt.cm.gray)
ax[2].set_title('Enhanced image, radius=1, amount=2.0')
ax[3].imshow(equalized_image_3, cmap=plt.cm.gray)
ax[3].set_title('Enhanced image, radius=5, amount=1.0')
ax[4].imshow(equalized_image_4, cmap=plt.cm.gray)
ax[4].set_title('Enhanced image, radius=5, amount=2.0')
ax[5].imshow(equalized_image_5, cmap=plt.cm.gray)
ax[5].set_title('Enhanced image, radius=20, amount=1.0')
ax[6].imshow(equalized_image_6, cmap=plt.cm.gray)
ax[6].set_title('Enhanced image, radius=20, amount=2.0')


for a in ax:
    a.axis('off')
fig.tight_layout()
plt.show()

print("the result findings suggest when parameter is set to radius=5 and amount=2.0 gives the best result, which is in line with the previous findings")

# CLAHE

CLAHE_1 = equalize_adapthist(image, kernel_size=None, clip_limit=0.01, nbins=256)
CLAHE_2 = equalize_adapthist(image, kernel_size=None, clip_limit=0.02, nbins=256)
CLAHE_3 = equalize_adapthist(image, kernel_size=None, clip_limit=0.03, nbins=256)
CLAHE_4 = equalize_adapthist(image, kernel_size=None, clip_limit=0.04, nbins=256)
CLAHE_5 = equalize_adapthist(image, kernel_size=None, clip_limit=0.5, nbins=256)

fig, axes = plt.subplots(nrows=6, ncols=1,
                         sharex=True, sharey=True, figsize=(20, 20))
ax = axes.ravel()

ax[0].imshow(image, cmap=plt.cm.gray)
ax[0].set_title('Original image')
ax[1].imshow(CLAHE_1, cmap=plt.cm.gray)
ax[1].set_title('Enhanced image, clip_limit=0.01, nbins=256')
ax[2].imshow(CLAHE_2, cmap=plt.cm.gray)
ax[2].set_title('Enhanced image, clip_limit=0.02, nbins=256')
ax[3].imshow(CLAHE_3, cmap=plt.cm.gray)
ax[3].set_title('Enhanced image, clip_limit=0.03, nbins=256')
ax[4].imshow(CLAHE_4, cmap=plt.cm.gray)
ax[4].set_title('Enhanced image, clip_limit=0.04, nbins=256')
ax[5].imshow(CLAHE_5, cmap=plt.cm.gray)
ax[5].set_title('Enhanced image, clip_limit=0.5, nbins=256')

print("the results get clearer as the clip limit increases. possible drawback of CLAHE is its running time.")

# side-by-side comparison of unsharp masking + histogram equalization with CLAHE

fig, axes = plt.subplots(nrows=1, ncols=3,
                         sharex=True, sharey=True, figsize=(20, 20))
ax = axes.ravel()

ax[0].imshow(image, cmap=plt.cm.gray)
ax[0].set_title('Original image with pneumothorax')
ax[1].imshow(equalized_image_4, cmap=plt.cm.gray)
ax[1].set_title('Unsharp masking and histogram equalization')
ax[2].imshow(CLAHE_5, cmap=plt.cm.gray)
ax[2].set_title('CLAHE')

# implementing unsharp masking + histogram equalization on pytorch

data: torch.tensor = kornia.utils.image_to_tensor(image, keepdim=False)
data = data.float() / 255

sharpen = kornia.filters.UnsharpMask((25,25), (15,15))
sharpened_tensor = sharpen(data)
image_eq = kornia.enhance.equalize(image_sharp)
sharpened_image = kornia.utils.tensor_to_image(sharpened_tensor) 

fig, axs = plt.subplots(1, 2, figsize=(16, 10))
axs = axs.ravel()

axs[0].set_title('image source')
axs[0].imshow(image)


axs[1].set_title('unsharp masking and histogram equalization')
axs[1].imshow(sharpened_image)

# experimenting with pytorch library
transform = transforms.Compose([transforms.PILToTensor()])
img_tensor = transform(image)
img_tensor_org = torchvision.transforms.functional.to_pil_image(img_tensor, mode=None)


img_sharp = TF.adjust_sharpness(img_tensor, sharpness_factor=5)
img_bright = torchvision.transforms.functional.to_pil_image(img_sharp, mode=None)

fig, axes = plt.subplots(nrows=1, ncols=2,
                         sharex=True, sharey=True, figsize=(20, 20))
ax = axes.ravel()

ax[0].imshow(img_tensor_org, cmap=plt.cm.gray)
ax[0].set_title('Original image with pneumothorax')
ax[1].imshow(img_bright, cmap=plt.cm.gray)
ax[1].set_title('Sharpness')