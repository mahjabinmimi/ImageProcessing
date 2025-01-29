import cv2 as cv

img = cv. imread("D:\My Downloads\penguin1.jpg")
cv.imshow("Penguin", img)

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow("Gray", gray)

#simple Thresholding
threshold, thresh = cv.threshold(gray,150,255, cv.THRESH_BINARY)
cv.imshow('Simple Thresholding', thresh)

threshold, thresh_inv = cv.threshold(gray,150,255, cv.THRESH_BINARY_INV)
cv.imshow('INV THRESHOLD', thresh_inv)

#Adaptive Thresholding
adaptive_thresh = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,11,9)
cv.imshow('Adaptive Thresholding', adaptive_thresh)

adaptive_thresh1 = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,9)
cv.imshow('Adaptive Thresholding', adaptive_thresh1)





cv.waitKey(0)