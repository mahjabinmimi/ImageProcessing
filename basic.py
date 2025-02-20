import cv2 as cv

img = cv.imread("D:\My Downloads\park.jpg")
cv.imshow('park', img)
"""
#converting to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)
"""
#Blur
blur = cv.GaussianBlur(img,(7,7),cv.BORDER_DEFAULT)
cv.imshow('Blur',blur)

#Edge Cascode
#canny = cv.Canny(img, 150, 255)
canny = cv.Canny(blur,125,175)
cv.imshow('Canny Edges',canny)

#Dilated the image
dilated = cv.dilate(canny,(7,7),iterations=3)
cv.imshow('Dilated',dilated)

#Eroding
eroded = cv.erode(dilated, (3,3),iterations=1)
cv.imshow('Eroded',eroded)

#Resize
resized = cv.resize(img, (200,300),interpolation=cv.INTER_CUBIC)
cv.imshow('Resized',resized)

#Cropping
cropped = img[50:200, 200:400]
cv.imshow('cropped', cropped)

cv.waitKey(0)