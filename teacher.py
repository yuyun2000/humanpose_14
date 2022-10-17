import torchvision.models as m
import torchvision.io as tio
import torchvision.transforms
import torch

img = tio.read_image('./data/images/im00002.jpg')
transforms1 = torchvision.transforms.ToPILImage()
transforms2 = torchvision.transforms.ToTensor()
person_float = transforms1(img)
person_float = transforms2(person_float)

model = m.detection.keypointrcnn_resnet50_fpn(True,progress=False).eval()

outputs = model([person_float])

kpts = outputs[0]['keypoints']
scores = outputs[0]['scores']

print(kpts)
print(scores)

detect_threshold = 0.75
idx = torch.where(scores > detect_threshold)
keypoints = kpts[idx]

print(keypoints)

import numpy as np
import matplotlib.pyplot as plt

import torchvision.transforms.functional as F

plt.rcParams["savefig.bbox"] = 'tight'


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


from torchvision.utils import draw_keypoints

res = draw_keypoints(img, keypoints, colors="blue", radius=3)
show(res)
