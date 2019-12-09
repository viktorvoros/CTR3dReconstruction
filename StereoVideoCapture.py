import cv2
import numpy as np

camera1 = 0
camera2 = 1

cap1 = cv2.VideoCapture(camera1)
cap2 = cv2.VideoCapture(camera2)

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
outCam1 = cv2.VideoWriter('outCam1_6.avi', fourcc, 20.0, (640,480), True)
outCam2 = cv2.VideoWriter('outCam2_6.avi', fourcc, 20.0, (640,480), True)

while(True):
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    outCam1.write(frame1)
    outCam2.write(frame2)

    cv2.imshow('frame',frame1)
    cv2.imshow('frame1', frame2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()                                      # close camera, release resources
outCam1.release()
outCam2.release()                                      #close out (video writer)

cv2.destroyAllWindows()
