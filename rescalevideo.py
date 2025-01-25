import cv2 as cv


def rescaleFrame(frame, scale=0.75):
   
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)
def changeRes(width,height):
    capture.set(3,width)
    capture.set(4,height)
    

capture = cv.VideoCapture('D:\\My Downloads\\catvideo.mp4')  
if not capture.isOpened():
    print("Error: Could not open the video file. Please check the file path.")
else:
    while True:
       
        isTrue, frame = capture.read()

       
        if not isTrue:
            print("End of video or cannot read the frame.")
            break

      
        resized_frame = rescaleFrame(frame, scale=0.5) 
        cv.imshow('Original Video', frame)
        cv.imshow('Rescaled Video', resized_frame)

     
        if cv.waitKey(20) & 0xFF == ord('d'):
            print("Exiting video playback.")
            break
capture.release()
cv.destroyAllWindows()
