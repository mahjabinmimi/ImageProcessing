import cv2 as cv
import numpy as np

blank= np.zeros((500,500,3), dtype= 'uint8')
cv.imshow('Blank',blank)
"""
blank[200:300,400:500] = 0,0,255
cv. imshow('Green', blank)
"""
#cv.rectangle(blank,(0,0),(250,500),(0,0,255),thickness=cv.FILLED)
#For small ractangle:
cv.rectangle(blank,(0,0),(blank.shape[1]//2,blank.shape[0]//2),(0,255,0),thickness=-1)
cv.imshow('Rectangle',blank)
#for a circle
cv.circle(blank,(blank.shape[1]//2, blank.shape[0]//2),50,(0,0,255),thickness=cv.FILLED)
cv.imshow('Circle',blank)
#For draw a line:
cv.line(blank,(0,0),(blank.shape[1]//2,blank.shape[0]//2),(255,255,255),thickness=4)
cv.imshow('Line', blank)\
#changing direction a line:
cv.line(blank,(200,350),(400,500),(255,255,255),thickness=5)
cv.imshow('Line',blank)
#write text
cv.putText(blank,'Hello, my name is MIMI!!',(0,255), cv.FONT_HERSHEY_TRIPLEX,1.0,(0,255,0),2)
cv.imshow('Text',blank)
cv.waitKey(0)