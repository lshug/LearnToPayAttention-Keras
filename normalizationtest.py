from keras.datasets import cifar10
from keras.datasets import cifar100
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from sklearn.decomposition import PCA

(x10, y10), (x_test, y_test) = cifar10.load_data()
(x100,y100),(x_test2,y_test2) = cifar100.load_data(label_mode='fine')

x10=x10[0:1000]
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
    
def show(i):
    i = i.reshape((32,32,3))
    m,M = i.min(), i.max()
    plt.imshow((i - m) / (M - m))
    plt.show()

plt.imshow(x10[3])
plt.show()
print("Pre-normalization")
X=normalizeDataset(x10)
print("Post-normalization")
plt.clf()
show(X[3])