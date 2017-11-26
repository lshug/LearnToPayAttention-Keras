import cv2
import numpy as np
import os

os.chdir("cub2002011")
image = open('images.txt', 'r')
bbox = open('bounding_boxes.txt', 'r')

images = []
classes = []
i = 1;
for line in image:
	# Extract bounding box conditions
	split = [int(float(x))for x in bbox.readline().split()]
	x, y, w, h = split[1], split[2], split[3], split[4]
	
	# Read and process image #line = image.readline()
	path = 'images/'+line.split()[1]
	classnumber = int(line.split('/')[0].split()[1].split('.')[0])
	img = cv2.imread(path)
	img = img[y: y+h, x: x+w]
	img = cv2.resize(img, (80,80))
	
	#Append to list	
	classes.append(classnumber)
	images.append(img)
	if i % 500 == 0:
		print(i)
	i = i+1

#Convert to numpy array and save to file
images = np.array(images)
clases = np.array(classes)

os.chdir("..")

np.save("cubimgArr", images)
np.save("cubclassArr", classes)
