from LearnToPayAttention import AttentionVGG,AttentionRN,StandardVGG
import numpy as np
import scipy.io as sio
from keras.datasets import cifar10
from keras.datasets import cifar100
  
x10 = np.load("datasets/x10.npy")
x10test = np.load("datasets/x10test.npy")
x100 = np.load("datasets/x100.npy")
x100test = np.load("datasets/x100test.npy")
y10 = np.load("datasets/y10.npy")
y10test = np.load("datasets/y10test.npy")
y100 = np.load("datasets/y100.npy")
y100test = np.load("datasets/y100test.npy")
xsvhn = np.load("datasets/xsvhn.npy")  
xsvhntest = np.load("datasets/xsvhntest.npy")  
ysvhn = np.load("datasets/ysvhn.npy") - 1  
ysvhntest = np.load("datasets/ysvhntest.npy") - 1  
xcub = np.load("datasets/xcub.npy")
xcubtest = np.load("datasets/xcubtest.npy")
ycub = np.load("datasets/ycub.npy")
ycubtest = np.load("datasets/ycubtest.npy")


vggatt3concatpcCIFAR10 = AttentionVGG().StandardFit("cifar10",x10,y10,beep=True, min_delta=0.01, patience=3, validation_data=(x10test,y10test))
vggatt3concatpcCIFAR100 = AttentionVGG(outputclasses=100).StandardFit("cifar100",x100,y100,beep=True, min_delta=0.01, patience=3, validation_data=(x100test,y100test))
vggatt3concatpcSVHN = AttentionVGG().StandardFit("svhn",xsvhn,ysvhn,beep=True, min_delta=0.01, validation_data=(xsvhntest,ysvhntest), patience=7, lrplateaufactor=0.5, lrplateaupatience=3, initial_lr=0.0025)

vggatt2concatpcCIFAR10 = AttentionVGG(att='att2').StandardFit("cifar10",x10,y10,beep=True, min_delta=0.01, patience=3, validation_data=(x10test,y10test))
vggatt2concatpcCUB = AttentionVGG(att='att2', height=80, width=80, outputclasses=200).StandardFit("cub2002011",xcub,ycub,True,beep=True, min_delta=0.01, validation_data=(xcubtest,ycubtest))

RNatt2concatpcCIFAR10 = AttentionRN().StandardFit("cifar10",x10,y10,beep=True, min_delta=0, validation_data=(x10test,y10test))
RNatt2concatpcCIFAR100 = AttentionRN(outputclasses=100).StandardFit("cifar100",x100,y100,beep=True, min_delta=0, validation_data=(x100test,y100test))
    


