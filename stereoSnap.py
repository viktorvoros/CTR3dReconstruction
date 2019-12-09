import cv2
import numpy as np

cam1 = cv2.VideoCapture(0)
cam2 = cv2.VideoCapture(1)
img_counter = 0

while True:
    ret1, frame1 = cam1.read()
    ret2, frame2 = cam2.read()
    # hsv1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
    # hsv2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
    # # # segmentation
    # mask1 = cv2.inRange(hsv1, hsvMin, hsvMax)
    # mask2 = cv2.inRange(hsv2, hsvMin, hsvMax)
    # res1 = cv2.bitwise_not(frame1, frame1, mask = mask1)
    # res2 = cv2.bitwise_not(frame2, frame2, mask = mask2)

    cv2.imshow('Camera 1', frame1)
    cv2.imshow('Camera 2', frame2)

    if not ret1 and ret2:
        break
    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name1 = 'Cam1_{}.jpg'.format(img_counter)
        img_name2 = 'Cam2_{}.jpg'.format(img_counter)
        cv2.imwrite(img_name1, frame1)
        cv2.imwrite(img_name2, frame2)
        print('{} written!'.format(img_name1))
        print('{} written!'.format(img_name2))
        img_counter += 1

cam1.release()
cam2.release()
cv2.destroyAllWindows()
print('Images captured!')
