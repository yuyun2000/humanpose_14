import numpy
import tensorflow as tf
import cv2
import numpy as np
from utils import draw_bd_pose
from Flops import try_count_flops

model = tf.keras.models.load_model("./h5/pose-point-33.h5")
# model.summary()

# # flops = try_count_flops(model)
# # print(flops)
# imagegt = cv2.imread('./data/test/11.jpg')
# # imagegt = np.pad(imagegt, ((500, 500), (500, 500), (0, 0)))
# s = 256
# imagegt = cv2.resize(imagegt,(s,s),interpolation=cv2.INTER_AREA)
# image = imagegt.astype(np.float32)
# img = image / 255
# img = img.reshape(1,s,s,3)
# out = model(img,training=False)
# out = np.array(tf.reshape(out[0:1, :], (14,2)))
#
# point  = np.zeros((14,2))
# # for k in range(14):
# #     max = np.max(out[:, :, k:k + 1].flatten())
# #     for i in range(64):
# #         for j in range(64):
# #             if out[i][j][k] >= max:
# #                 point[k][0] = j*4+2
# #                 point[k][1] = i*4+2
# #                 cv2.circle(imagegt, (j * 4 + 2, i * 4 + 2), 1, (0, 0, 255), 2)
#
# for i in range(14):
#         point[i] = (int((out[i][0]+0.5)*256),int((out[i][1]+0.5)*256))
#         x = int((out[i][0]+0.5)*256)
#         y = int((out[i][1]+0.5)*256)
#         cv2.circle(imagegt, (x, y), 1, (0, 0, 255), 2)
#
# draw_bd_pose(imagegt,point)
# imagegt = cv2.resize(imagegt,(512,512))
# cv2.imshow('1',imagegt)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#### ----------------------video
import time
t1 = time.time()
s = 256
vid = cv2.VideoCapture('./data/test/tji.mp4')
fourcc = cv2.VideoWriter_fourcc(*'I420')
outv = cv2.VideoWriter('output.avi',fourcc,20,(s,s))
while True:
    flag,img = vid.read(0)
    if not flag:
        break
    imagegt = cv2.resize(img, (s, s))
    image = imagegt.astype(np.float32)
    img = image / 255
    img = img.reshape(1, s, s, 3)
    out = model(img, training=False)
    out = (np.array(tf.reshape(out[0:1, :], (14, 2)))+0.5)*s
    draw_bd_pose(imagegt,out)
    # for i in range(14):
        # x = int(out[i][0])
        # y = int(out[i][1])
        # cv2.circle(imagegt, (x, y), 1, (0, 0, 255), 2)

    img0 = cv2.resize(imagegt, (s, s))
    outv.write(img0)
    cv2.imshow('1', img0)
    if ord('q') == cv2.waitKey(1):
        break
vid.release()
outv.release()
#销毁所有的数据
cv2.destroyAllWindows()

t2 = time.time()
print(t2-t1)

