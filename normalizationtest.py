from keras.datasets import cifar10
from keras.datasets import cifar100
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator


(x10, y10), (x_test, y_test) = cifar10.load_data()
(x100,y100),(x_test2,y_test2) = cifar100.load_data(label_mode='fine')

x10=x10[0:1000]
def normalizeDataset(x):
    x=x.astype('float32')
    g = ImageDataGenerator(featurewise_center=True,featurewise_std_normalization=True,zca_epsilon=1e-6,zca_whitening=True)
    g.fit(x)
    return g.standardize(x)
    
def show(i):
    i = i.reshape((32,32,3))
    m,M = i.min(), i.max()
    plt.imshow((i - m) / (M - m))
    plt.show()

plt.imshow(x10[6])
plt.show()
print("Pre-normalization")
X=normalizeDataset(x10)
print("Post-normalization")
plt.clf()
show(X[6])
plt.clf()
plt.imshow(X[6])
plt.show()
x10 = np.load("datasets/x10.npy")
plt.clf()
show(x10[6])
