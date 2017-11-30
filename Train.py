import LearnToPayAttention
import numpy as np
import scipy.io as sio
from keras.datasets import cifar10
from keras.datasets import cifar100

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
    
    x10 = np.load("datasets/x10.npy")
    x100 = np.load("datasets/x100.npy")
    y10 = np.load("datasets/y10.npy")
    y100 = np.load("datasets/y100.npy")
    xsvhn = np.load("datasets/xsvhn.npy")  
    ysvhn = np.load("datasets/ysvhn.npy")  
    xcub = np.load("datasets/xcub.npy")
    ycub = np.load("datasets/ycub.npy")
    
    vggatt3concatpcCIFAR10 = AttentionVGG(att='att3', gmode='concat', compatibilityfunction='pc', height=32, width=32, channels=3, outputclasses=10).StandardFit("cifar10",x10,y10)
    vggatt3concatpcCIFAR100 = AttentionVGG(att='att3', gmode='concat', compatibilityfunction='pc', height=32, width=32, channels=3, outputclasses=100).StandardFit("cifar100",x100,y100)
    vggatt3concatpcSVHN = AttentionVGG(att='att3', gmode='concat', compatibilityfunction='pc', height=32, width=32, channels=3, outputclasses=10).StandardFit("svhn",xsvhn,ysvhn)
    vggatt3concatpcCUB = AttentionVGG(att='att3', gmode='concat', compatibilityfunction='pc', height=32, width=32, channels=3, outputclasses=10).StandardFit("cub2002011",xcub,ycub,True)

    vggatt2concatpcCIFAR10 = AttentionVGG(att='att2', gmode='concat', compatibilityfunction='pc', height=32, width=32, channels=3, outputclasses=10).StandardFit("cifar10",x10,y10)
    vggatt2concatpcCIFAR100 = AttentionVGG(att='att2', gmode='concat', compatibilityfunction='pc', height=32, width=32, channels=3, outputclasses=100).StandardFit("cifar100",x100,y100)
    vggatt2concatpcSVHN = AttentionVGG(att='att2', gmode='concat', compatibilityfunction='pc', height=32, width=32, channels=3, outputclasses=10).StandardFit("svhn",xsvhn,ysvhn)
    vggatt2concatpcCUB = AttentionVGG(att='att2', gmode='concat', compatibilityfunction='pc', height=32, width=32, channels=3, outputclasses=10).StandardFit("cub2002011",xcub,ycub,True)

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

    
    


