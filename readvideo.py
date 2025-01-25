import cv2 as cv

capture = cv.VideoCapture('D:\\My Downloads\\catvideo.mp4') 

while True:
 
        isTrue, frame = capture.read()

 
        if not isTrue:
            break

        cv.imshow('catvideo', frame)

        if cv.waitKey(20) & 0xFF == ord('d'):
            print("Exiting video playback.")
            break

capture.release()
cv.destroyAllWindows()
