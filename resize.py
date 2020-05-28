import cv2
import os
import time


for filename in os.listdir('path of your folder that contains the images'):

	img = cv2.imread('path of your folder that contains the images'+filename, cv2.IMREAD_UNCHANGED)
	dim=(200,200)
	resized=cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
	#print('resized dimension: ', resized.shape)
	cv2.imwrite('path of the resized image'+filename, resized)


