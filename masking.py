import cv2 as cv
import numpy as np

img = cv.imread("D:\My Downloads\penguin1.jpg")
cv.imshow("pengu", img)

blank = np.zeros(img.shape[:2], dtype= 'uint8')
cv.imshow('Blank',blank)


mask = cv.rectangle(blank,(img.shape[1]//2+45,img.shape[0]//2),(img.shape[1]//2,img.shape[0]//2 - 40),255,-1)

cv.imshow('Mask',mask)

circle = cv.circle(blank.copy(),(img.shape[1]//2+45,img.shape[0]//2),100,255,-1)
cv.imshow('Mask',mask)

rectangle = cv.rectangle(blank.copy(),(30,30),(370,370),255,-1)

weird_shape = cv.bitwise_and(circle,rectangle)
cv.imshow('weird shape',weird_shape)
masked = cv.bitwise_and(img,img,mask=weird_shape)
cv.imshow(' weird_shape Masked Image',masked)

cv.waitKey(0)