import cv2

# press esc if you want to stop in the middle of writing the images
# camera records with 20 fps, so it will be thousands of images
# write the video name you want to read here
videoName = 'outCam.avi'

# reads video
cap = cv2.VideoCapture(videoName)
imgCounter = 0

while True:
    ret, frame = cap.read()
    # name of image to be saved, {} will be from 0..1..2..3.......
    imgName = 'Vid1_{}.jpg'.format(imgCounter)
    cv2.imwrite(imgName, frame)
    imgCounter += 1

    # if cv2.waitKey(0) & 0xFF == ord('q'):
    #     break

cap.release()

cv2.destroyAllWindows()
