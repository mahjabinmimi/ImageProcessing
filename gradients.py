import cv2 as cv
import numpy as  np
img = cv.imread("D:\My Downloads\penguin1.jpg")
cv.imshow("panguin", img)

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('Gray',gray)

# #laplocation 
lap= cv.Laplacian(gray,cv.CV_64F)
lap=np.uint8(np.absolute(lap))
cv.imshow('laplacian',lap)

#sabel
sobelx=cv.Sobel(gray,cv.CV_64F,1,0)
sobely=cv.Sobel(gray,cv.CV_64F,0,1)

cv.imshow('Sobel x',sobelx)
cv.imshow('Sobel y',sobely)
combined_sobel =cv.bitwise_or(sobelx,sobely)
cv.imshow('Sobel combined',combined_sobel)
canny = cv.Canny(gray,100,200)
cv.imshow('Canny',canny)



cv.waitKey(0)