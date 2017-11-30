from keras.datasets import cifar10
from keras.datasets import cifar100
import numpy as np
import scipy.io as sio
from scipy import linalg
import matplotlib.pyplot as plt

def normalizeDataset(x):
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

(x10, y10), (x10test, y10test) = cifar10.load_data()
(x100,y100),(x100test,y100test) = cifar100.load_data(label_mode='fine')
x10 = np.reshape(x10,[50000,32,32,3])
x100 = np.reshape(x100,[50000,32,32,3])
x10test = np.reshape(x10test,[10000,32,32,3])
x100test = np.reshape(x100test,[10000,32,32,3])
svhn_data = sio.loadmat('datasets/svhn.mat')
xsvhn = svhn_data['X']
ysvhn = svhn_data['y']
xsvhn = np.reshape(xsvhn,[73257,32,32,3])
ysvhn = np.squeeze(ysvhn)
svhn_data = sio.loadmat('datasets/svhntest.mat')
xsvhntest = svhn_data['X']
ysvhntest = svhn_data['y']
xsvhntest = np.reshape(xsvhntest,[26032,32,32,3])
ysvhntest = np.squeeze(ysvhntest)

x10 = normalizeDataset(x10)
print("x10 normalized")
x10test = normalizeDataset(x10test)
print("x10test normalized")
x100 = normalizeDataset(x100)
print("x100 normalized")
x100test = normalizeDataset(x100test)
print("x100test normalized")

np.save("datasets/x10", x10)
np.save("datasets/x10test",x10test)
np.save("datasets/y10", y10)
np.save("datasets/y10test", y10test)
np.save("datasets/x100", x100)
np.save("datasets/x100test", x100test)
np.save("datasets/y100", y100)
np.save("datasets/y100test", y100test)
np.save("datasets/xsvhn", xsvhn)
np.save("datasets/xsvhntest", xsvhntest)
np.save("datasets/ysvhn", ysvhn)
np.save("datasets/ysvhntest", ysvhntest)
