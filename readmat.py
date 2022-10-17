import scipy.io as sio
import numpy as np
import cv2
import torchlm
from utils import draw_bd_pose
transform = torchlm.LandmarksCompose([
    torchlm.LandmarksClip(1,1),
    torchlm.LandmarksRandomScale(),
    torchlm.LandmarksRandomMask(prob=0.3),
    torchlm.LandmarksRandomBlur(kernel_range=(1, 10), prob=0.3),
    torchlm.LandmarksRandomTranslate(translate=0.2,prob=1,diff=True),
    torchlm.LandmarksRandomRotate((-45,45), prob=0.5),
    torchlm.LandmarksRandomCenterCrop((0.5, 1.0), (0.5, 1.0), prob=0.1),
])

# data = sio.loadmat('./data/joints.mat')
# arr = np.array(data.get('joints'))
# print(arr.shape)
# label = arr[:, :2, 15].reshape(14, 2)
# x = np.random.randint(1000,9999)
label = np.load('./data/label/im09999.npy')
img = cv2.imread('./data/images/im09999.jpg')

img1 , lm = transform(img,label)
if 0 in label:
    for i in range(14):
        if label[i][0] == label[i][1] and label[i][0] == 0:
            lm[i][0] = 0
            lm[i][1] = 0
h,w,c = img1.shape
lm[:,:1] = lm[:,:1]/w
lm[:,1:] = lm[:,1:]/h
img1 = cv2.resize(img1,(256,256),interpolation=cv2.INTER_AREA)
for i in range(14):
    x = int(label[i][0])
    y = int(label[i][1])
    x2 = int(lm[i][0]*256)
    y2 = int(lm[i][1]*256)
    cv2.putText(img,'%s'%i,(x,y),1,1,(0,0,255),1)

    cv2.circle(img,(x,y),2,(0,0,255),2)
    cv2.circle(img1,(x2,y2),2,(0,0,255),2)

draw_bd_pose(img1,lm)
img1 = cv2.resize(img1,(256,256),interpolation=cv2.INTER_AREA)
cv2.imshow('1',img)
cv2.imshow('2',img1)
cv2.waitKey(0)