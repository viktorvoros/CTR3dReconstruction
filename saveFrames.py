import cv2
import numpy as np

# run script - video will start playing - press space to capture a frame from the video
# press space as many times as you want - that frame will be saved
# press q to interrupt video
# change the image_name in while loop after elif when saving another video - not to overwrite previous images
# if you don't want to read 2 videos at the same time, just comment out the stuff with 2 at the end


# write the video name you want to read here
videoName1 = 'outCam1_6.avi'
videoName2 = 'outCam2_6.avi'

cam1 = cv2.VideoCapture(videoName1)
cam2 = cv2.VideoCapture(videoName2)
img_counter = 0

while True:
    ret1, frame1 = cam1.read()
    ret2, frame2 = cam2.read()

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
        img_name1 = 'Vid1_{}.jpg'.format(img_counter)
        img_name2 = 'Vid2_{}.jpg'.format(img_counter)
        cv2.imwrite(img_name1, frame1)
        cv2.imwrite(img_name2, frame2)
        print('{} written!'.format(img_name1))
        print('{} written!'.format(img_name2))
        img_counter += 1

cam1.release()
cam2.release()
cv2.destroyAllWindows()
print('Images captured!')
