from keras.datasets import cifar10
from keras.datasets import cifar100
import numpy as np
import scipy.io as sio
from scipy import linalg
from keras.preprocessing.image import ImageDataGenerator

def normalizeDataset(x):
    x=x.astype('float32')
    g = ImageDataGenerator(featurewise_center=True,featurewise_std_normalization=True,zca_epsilon=1e-6,zca_whitening=True)
    g.fit(x)
    return g.standardize(x)


#todo: normalize and save STL-train, STL-test, Caltech-101, Caltech-256, Event-8, Action-40, Scene-67, Object Discovery 

(x10, y10), (x10test, y10test) = cifar10.load_data()
(x100,y100),(x100test,y100test) = cifar100.load_data(label_mode='fine')
svhn_data = sio.loadmat('datasets/svhn.mat')
xsvhn = svhn_data['X']
ysvhn = svhn_data['y']
xsvhn = np.rollaxis(xsvhn,3,-4)
ysvhn = np.squeeze(ysvhn)
svhn_data = sio.loadmat('datasets/svhntest.mat')
xsvhntest = svhn_data['X']
ysvhntest = svhn_data['y']
xsvhntest = np.rollaxis(xsvhntest,3,-4)
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
