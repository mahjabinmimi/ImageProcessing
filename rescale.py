import cv2 as cv

img = cv.imread('D:\\My Downloads\\main.jpg')
def rescaleFrame(frame, scale=0.75):

        width = int(frame.shape[1] * scale)
        height = int(frame.shape[0] * scale)

        dimensions = (width, height)


        return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


resized_img = rescaleFrame(img)

cv.imshow('Original Image', img)
cv.imshow('Rescaled Image', resized_img)

cv.waitKey(0)
cv.destroyAllWindows()

