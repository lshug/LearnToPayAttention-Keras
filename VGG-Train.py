import LearnToPayAttention
import numpy as np
import scipy.io as sio
from keras.datasets import cifar10
from keras.datasets import cifar100
import cv2

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

    #todo: normalize and color-normalize x10, x100, and xsvhn.
    #todo: implement and crop cub-200-2011


    vggatt1concatdpCIFAR10 = AttentionVGG(att='att1', gmode='concat', compatibilityfunction='dp', height=32, width=32, channels=3, outputclasses=10).StandardFit("cifar10",x10,y10)
    vggatt1concatdpCIFAR100 = AttentionVGG(att='att1', gmode='concat', compatibilityfunction='dp', height=32, width=32, channels=3, outputclasses=100).StandardFit("cifar100",x100,y100)

    vggatt1concatpcCIFAR10 = AttentionVGG(att='att1', gmode='concat', compatibilityfunction='pc', height=32, width=32, channels=3, outputclasses=100).StandardFit("cifar10",x10,y10)
    vggatt1concatpcCIFAR100 = AttentionVGG(att='att1', gmode='concat', compatibilityfunction='pc', height=32, width=32, channels=3, outputclasses=100).StandardFit("cifar100",x100,y100)

    vggatt2indepdpCIFAR10 = AttentionVGG(att='att2', gmode='indep', compatibilityfunction='dp', height=32, width=32, channels=3, outputclasses=10).StandardFit("cifar10",x10,y10)
    vggatt2indepdpCIFAR100 = AttentionVGG(att='att2', gmode='indep', compatibilityfunction='dp', height=32, width=32, channels=3, outputclasses=100).StandardFit("cifar100",x100,y100)

    vggatt2indeppcCIFAR10 = AttentionVGG(att='att2', gmode='indep', compatibilityfunction='pc', height=32, width=32, channels=3, outputclasses=10).StandardFit("cifar10",x10,y10)
    vggatt2indeppcCIFAR100 = AttentionVGG(att='att2', gmode='indep', compatibilityfunction='pc', height=32, width=32, channels=3, outputclasses=100).StandardFit("cifar100",x100,y100)

    vggatt2concatpcCIFAR10 = AttentionVGG(att='att2', gmode='concat', compatibilityfunction='pc', height=32, width=32, channels=3, outputclasses=10).StandardFit("cifar10",x10,y10)
    vggatt2concatpcCIFAR100 = AttentionVGG(att='att2', gmode='concat', compatibilityfunction='pc', height=32, width=32, channels=3, outputclasses=100).StandardFit("cifar100",x100,y100)
    vggatt2concatpcSVHN = AttentionVGG(att='att2', gmode='concat', compatibilityfunction='pc', height=32, width=32, channels=3, outputclasses=10).StandardFit("svhn",xsvhn,ysvhn)

    vggatt3concatpcCIFAR10 = AttentionVGG(att='att3', gmode='concat', compatibilityfunction='pc', height=32, width=32, channels=3, outputclasses=10).StandardFit("cifar10",x10,y10)
    vggatt3concatpcCIFAR100 = AttentionVGG(att='att3', gmode='concat', compatibilityfunction='pc', height=32, width=32, channels=3, outputclasses=100).StandardFit("cifar100",x100,y100)
    vggatt3concatpcSVHN = AttentionVGG(att='att3', gmode='concat', compatibilityfunction='pc', height=32, width=32, channels=3, outputclasses=10).StandardFit("svhn",xsvhn,ysvhn)



