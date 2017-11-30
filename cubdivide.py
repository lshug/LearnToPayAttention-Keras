import numpy as np
import os

#todo: using divide cubimgArr.npy and cubclassArr.npy into xcub.npy, ycub.npy, xcubtest.npy, and ycubtest.npy
traintestsplit = open('cub2002011/train_test_split.txt', 'r')
os.chdir('datasets')
images = np.load('cubimgArr.npy')
labels = np.load('cubclassArr.npy')

print(labels.shape)

xcub = np.empty([5994, 80, 80, 3])
ycub = np.empty([5994])

xcubtest = np.empty([5794, 80, 80, 3])
ycubtest = np.empty([5794])

traini=0
testi=0
i=0
for l in traintestsplit:
    if l.split()[1] == '1':
        xcub[traini] = images[i]
        ycub[traini] = labels[i]
        traini = traini+1
    else:
        xcubtest[testi] = images[i]
        ycubtest[testi] = labels[i]
        testi=testi+1
    i=i+1
np.save('xcub',xcub)
np.save('ycub',ycub)
np.save('xcubtest',xcubtest)
np.save('ycubtest',ycubtest)