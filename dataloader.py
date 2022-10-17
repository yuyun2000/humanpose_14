import tensorflow as tf
import numpy as np
import os
import cv2
import torchlm

transform = torchlm.LandmarksCompose([
    torchlm.LandmarksClip(1,1),
    torchlm.LandmarksRandomScale(),
    torchlm.LandmarksRandomMask(prob=0.3),
    torchlm.LandmarksRandomBlur(kernel_range=(1, 10), prob=0.3),
    torchlm.LandmarksRandomTranslate(translate=0.6),
    torchlm.LandmarksRandomRotate((-45,45), prob=0.7),
    torchlm.LandmarksRandomCenterCrop((0.5, 1.0), (0.5, 1.0), prob=0.5),
])
def makeGaussian(size, fwhm = 2, center=None):
    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]
    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]
    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

def load_list(list_path='./label.txt',image_root_path='./data/images/'):
    images = []
    labels = []
    with open(list_path, 'r') as f:
        for line in f:
            # print(line)
            images.append(os.path.join(image_root_path, line[13:-6]+'.jpg'))
            labels.append(line[:-2])
    return images, labels

def load_image(image_path, label_path):

    # print(image_path.numpy().decode())
    image = cv2.imread(image_path.numpy().decode())
    preds = np.load(label_path.numpy().decode())

    image,preds = transform(image,preds)
    image=image.astype(np.float32)
    preds=preds.astype(np.float32)
    h,w,c = image.shape
    image = cv2.resize(image,(256,256),interpolation=cv2.INTER_AREA)
    image = image / 255

    # preds[:,0:1] = preds[:,0:1]/w
    # preds[:,1:2] = preds[:,1:2]/h
    # preds = preds.reshape(28)

    label = np.zeros((64,64,14), dtype=np.float32)
    for i in range(14):
        xmin = int(preds[i][0] / w *64) if int(preds[i][0] / w *64) <64 else 63
        ymin = int(preds[i][1] / h *64) if int(preds[i][1] / h *64) < 64 else 63
        label[:,:,i:i+1] = makeGaussian(64,center=(xmin,ymin)).reshape(64,64,1)

    return image, label


def train_iterator():
    images, labels = load_list()
    dataset = tf.data.Dataset.from_tensor_slices((images, labels)).shuffle(len(images))
    dataset = dataset.map(lambda x, y: tf.py_function(load_image, inp=[x, y], Tout=[tf.float32, tf.float32]),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)#.cache('/mnt/ssd0.5T/train/face68.TF-data')
    dataset = dataset.repeat()
    dataset = dataset.batch(50).prefetch(1)
    it = dataset.__iter__()
    return it

if __name__ == '__main__':
    it = train_iterator()
    images, labels = it.next()
    print(np.sum(np.array(tf.reshape(labels[0][:,:,:1],(64,64)))))




