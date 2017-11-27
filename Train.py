import LearnToPayAttention
import numpy as np
import scipy.io as sio
from keras.datasets import cifar10
from keras.datasets import cifar100
from sklearn.decomposition import PCA

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

if __name__ == "__main__":
    
    (x10, y10), (x_test, y_test) = cifar10.load_data()
    (x100,y100),(x_test2,y_test2) = cifar100.load_data(label_mode='fine')
    x10 = np.reshape(x10,[50000,32,32,3])
    x100 = np.reshape(x100,[50000,32,32,3])
    svhn_data = sio.loadmat('svhn.mat')
    xsvhn = svhn_data['X']
    ysvhn = svhn_data['y']
    xsvhn = np.reshape(xsvhn,[73257,32,32,3])
    ysvhn = np.squeeze(ysvhn)
    cubx = np.load("cubimgArr.npy")
    cuby = np.lod("cubclassArr.npy")
    
    x10 = normalizeDataset(x10)
    print("x10 normalized")
    x100 = normalizeDataset(x100)
    print("x100 normalized")
    xsvhn = normalizeDataset(xsvhn)
    print("xsvhn normalized")

    vggatt3concatpcCIFAR10 = AttentionVGG(att='att3', gmode='concat', compatibilityfunction='pc', height=32, width=32, channels=3, outputclasses=10).StandardFit("cifar10",x10,y10)
    vggatt3concatpcCIFAR100 = AttentionVGG(att='att3', gmode='concat', compatibilityfunction='pc', height=32, width=32, channels=3, outputclasses=100).StandardFit("cifar100",x100,y100)
    vggatt3concatpcSVHN = AttentionVGG(att='att3', gmode='concat', compatibilityfunction='pc', height=32, width=32, channels=3, outputclasses=10).StandardFit("svhn",xsvhn,ysvhn)

    vggatt2concatpcCIFAR10 = AttentionVGG(att='att2', gmode='concat', compatibilityfunction='pc', height=32, width=32, channels=3, outputclasses=10).StandardFit("cifar10",x10,y10)
    vggatt2concatpcCIFAR100 = AttentionVGG(att='att2', gmode='concat', compatibilityfunction='pc', height=32, width=32, channels=3, outputclasses=100).StandardFit("cifar100",x100,y100)
    vggatt2concatpcSVHN = AttentionVGG(att='att2', gmode='concat', compatibilityfunction='pc', height=32, width=32, channels=3, outputclasses=10).StandardFit("svhn",xsvhn,ysvhn)

    RNatt2concatpcCIFAR10 = AttentionRN(att='att2', gmode='concat', compatibilityfunction='pc', height=32, width=32, channels=3, outputclasses=10).StandardFit("cifar10",x10,y10)
    RNatt2concatpcCIFAR100 = AttentionRN(att='att2', gmode='concat', compatibilityfunction='pc', height=32, width=32, channels=3, outputclasses=100).StandardFit("cifar100",x100,y100)

    vggatt1concatdpCIFAR10 = AttentionVGG(att='att1', gmode='concat', compatibilityfunction='dp', height=32, width=32, channels=3, outputclasses=10).StandardFit("cifar10",x10,y10)
    vggatt1concatdpCIFAR100 = AttentionVGG(att='att1', gmode='concat', compatibilityfunction='dp', height=32, width=32, channels=3, outputclasses=100).StandardFit("cifar100",x100,y100)

    vggatt1concatpcCIFAR10 = AttentionVGG(att='att1', gmode='concat', compatibilityfunction='pc', height=32, width=32, channels=3, outputclasses=100).StandardFit("cifar10",x10,y10)
    vggatt1concatpcCIFAR100 = AttentionVGG(att='att1', gmode='concat', compatibilityfunction='pc', height=32, width=32, channels=3, outputclasses=100).StandardFit("cifar100",x100,y100)

    vggatt2indepdpCIFAR10 = AttentionVGG(att='att2', gmode='indep', compatibilityfunction='dp', height=32, width=32, channels=3, outputclasses=10).StandardFit("cifar10",x10,y10)
    vggatt2indepdpCIFAR100 = AttentionVGG(att='att2', gmode='indep', compatibilityfunction='dp', height=32, width=32, channels=3, outputclasses=100).StandardFit("cifar100",x100,y100)

    vggatt2indeppcCIFAR10 = AttentionVGG(att='att2', gmode='indep', compatibilityfunction='pc', height=32, width=32, channels=3, outputclasses=10).StandardFit("cifar10",x10,y10)
    vggatt2indeppcCIFAR100 = AttentionVGG(att='att2', gmode='indep', compatibilityfunction='pc', height=32, width=32, channels=3, outputclasses=100).StandardFit("cifar100",x100,y100)

    
    


