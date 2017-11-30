from keras.datasets import cifar10
from keras.datasets import cifar100
import numpy as np
import scipy.io as sio
from scipy import linalg
import matplotlib.pyplot as plt

def normalizeDataset(dataset):
    mean=np.reshape(np.mean(x, axis=(0, 1, 2)),[1,1,3])
    std= np.reshape(np.std(x, axis=(0, 1, 2)),[1,1,3])
    x=x-mean
    x=x/(std+1e-08)
    flat_x = np.reshape(x, (x.shape[0], 3072))
    sigma = np.dot(flat_x.T, flat_x) / flat_x.shape[0]
    u, s, _ = linalg.svd(sigma)
    principal_components = np.dot(np.dot(u, np.diag(1. / np.sqrt(s + 1e-6))), u.T)
    whitex = np.dot(flat_x, principal_components)
    return np.reshape(whitex, x.shape)

(x10, y10), (x_test, y_test) = cifar10.load_data()
(x100,y100),(x_test2,y_test2) = cifar100.load_data(label_mode='fine')
x10 = np.reshape(x10,[50000,32,32,3])
x100 = np.reshape(x100,[50000,32,32,3])
svhn_data = sio.loadmat('svhn.mat')
xsvhn = svhn_data['X']
ysvhn = svhn_data['y']
xsvhn = np.reshape(xsvhn,[73257,32,32,3])
ysvhn = np.squeeze(ysvhn)

x10 = normalizeDataset(x10)
print("x10 normalized")
x100 = normalizeDataset(x100)
print("x100 normalized")
xsvhn = normalizeDataset(xsvhn)
print("xsvhn normalized")

np.save("x10", x10)
np.save("y10", y10)
np.save("x100", x100)
np.save("y100", y100)
np.save("xsvhn", xsvhn)
np.save("xsvhn", ysvhn)