import scipy.io as sio
import numpy as np

data = sio.loadmat('./data/joints.mat')
arr = np.array(data.get('joints'))

for i in range(10000):
    label = arr[:, :2, i].reshape(14, 2)
    # label = arr[:2,:,i].reshape(2,14).transpose(1,0)
    i += 1
    if i < 10:
        np.save('./data/label/im0000%s'%i,label)
    elif i < 100:
        np.save('./data/label/im000%s' % i, label)
    elif i < 1000:
        np.save('./data/label/im00%s' % i, label)
    elif i < 10000:
        np.save('./data/label/im0%s' % i, label)
    else:
        np.save('./data/label/im%s' % i, label)