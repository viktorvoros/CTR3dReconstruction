import cv2
import numpy as np
import glob

# Just run it and the windows will pop up showing the camera images. Give input at the beginning about the calibration paper size.
# There are 3 steps:
# - Calibration of 1st camera
# - Calibration of 2nd camera
# - Stereocalibration
# For each step, the sequence is the following:
# 1. Camera window(s) open.
# 2. Press space to take an image of the calibration paper.
# 3. Do it as many times as you want with different orientation of the paper.
# 4. Press ESC to finish taking images.
# 5. Calibration is done automatically, variables are saved.
# After each step there is a message whether the calibration was successful or not.
# After the cameras are calibrated separately, just set isCalibrated = True at line 14, so that step will be skipped.

################
# INPUT START #
###############

# Are the 2 cameras calibrated separately?
isCalibrated = False
# Is the stereo system calibrated?
isStereoCalibrated = False
# giving calibration paper properties
chessboardSize = (6,5)
rectangleSize = 0.015 # mm
# camera codes
camera1 = 0
camera2 = 1

#############
# INPUT END #
#############

#further variables
isCam1Calibrated = False
isCam2Calibrated = False
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# if the cameras are calibrated, then the calibration steps are skipped
if isCalibrated == True:
    isCam1Calibrated = True
    isCam2Calibrated = True

else:
    isStereoCalibrated = False
    ###################################
    # Calibration of the first camera #
    ###################################
    if isCam1Calibrated == False:

    # capturing images for the calibration procedure
        cam = cv2.VideoCapture(camera1)
        img_counter = 0

        print('To take photos press SPACE as many times as needed, after images were taken, press ESC to continue')

        while True:
            ret, frame = cam.read()
            cv2.imshow("Camera 1 Calibration", frame)
            if not ret:
                break
            k = cv2.waitKey(1)

            if k%256 == 27:
                # ESC pressed
                break
            elif k%256 == 32:
                # SPACE pressed
                img_name = "calibrationcam1_{}.jpg".format(img_counter)
                cv2.imwrite(img_name, frame)
                print("{} written".format(img_name))
                img_counter += 1
        cam.release()
        cv2.destroyAllWindows()

        ################################
        # Calibrating the first camera #
        ################################
        print('First camera calibration in progres...')
        objp = np.zeros((chessboardSize[1]*chessboardSize[0],3), np.float32)            # creating an empty array for object points
        objp[:,:2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1,2)*recatangleSize

        objpoints = []      # array to store object points (3D points in real world)
        imgpoints = []      # array to store image points  (2d points in image space)

        images = glob.glob('calibrationcam1_*.jpg')         # loading all the jpg files collected previously

        for fname in images:
            img = cv2.imread(fname)         #loading all image files
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)            # converting to grayscale colorspace

            ret, corners = cv2.findChessboardCorners(gray, (chessboardSize[0],chessboardSize[1]), None)         # finding the corner points of the chessboard
            #if found, draw corners
            if ret == True:
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
                img = cv2.drawChessboardCorners(img, (chessboardSize[0], chessboardSize[1]), corners2, ret)

                cv2.imshow('img', img)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)          # calibrating the camre: getting the camera matrix, distortion coeff, rot, transl vector

        np.save('camMtx1', mtx)         # save the camera matrix in .npy file
        np.save('distCoeffs1.npy', dist)       # save the distortion coefficients
        np.save('camRvecs1', rvecs)
        np.save('camTvecs1', tvecs)

        isCam1Calibrated = True         # successful calibration of camera 1
        print('Calibration of first camera was successful: ', isCam1Calibrated)

    ###################################
    # Calibration of the second camera #
    ###################################
    if isCam2Calibrated == False:

    # capturing images for the calibration procedure
        cam = cv2.VideoCapture(camera2)
        img_counter = 0

        print('To take photos press SPACE as many times as needed, after images were taken, press ESC to continue')

        while True:
            ret, frame = cam.read()
            cv2.imshow('Camera 2 calibration', frame)
            if not ret:
                break
            k = cv2.waitKey(1)

            if k%256 == 27:
                # ESC pressed
                break
            elif k%256 == 32:
                # SPACE pressed
                img_name = "calibrationcam2_{}.jpg".format(img_counter)
                cv2.imwrite(img_name, frame)
                print("{} written".format(img_name))
                img_counter += 1
        cam.release()
        cv2.destroyAllWindows()

        #################################
        # Calibrating the second camera #
        #################################
        print('Second camera calibration in progres...')
        objp = np.zeros((chessboardSize[1]*chessboardSize[0],3), np.float32)            # creating an empty array for object points
        objp[:,:2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1,2)*recatangleSize

        objpoints = []      # array to store object points (3D points in real world)
        imgpoints = []      # array to store image points  (2d points in image space)

        images = glob.glob('calibrationcam2_*.jpg')         # loading all the jpg files collected previously

        for fname in images:
            img = cv2.imread(fname)         #loading all image files
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)            # converting to grayscale colorspace

            ret, corners = cv2.findChessboardCorners(gray, (chessboardSize[0],chessboardSize[1]), None)         # finding the corner points of the chessboard
            #if found, draw corners
            if ret == True:
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
                img = cv2.drawChessboardCorners(img, (chessboardSize[0], chessboardSize[1]), corners2, ret)

        ret2, mtx2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)          # calibrating the camre: getting the camera matrix, distortion coeff, rot, transl vector

        np.save('camMtx2', mtx2)         # save the camera matrix in .npy file
        np.save('distCoeffs2.npy', dist2)       # save the distortion coefficients
        np.save('camRvecs2', rvecs2)
        np.save('camTvecs2', tvecs2)

        isCam2Calibrated = True         # successful calibration of camera 2
        print('Calibration of second camera is finished: ',isCam2Calibrated)
        # cv2.waitKey(0) & 0xFF == ord('q')
    isCalibrated = True
    print('Calibration process was successful: ', isCalibrated)


#############################
# Stereo camera calibration #
#############################

if isStereoCalibrated == False and isCalibrated== True:
# If both cameras are separately calibrated successfuly, make stereocalibration
    #Capturing the calibration frames with both cameras
    cam1 = cv2.VideoCapture(camera1)
    cam2 = cv2.VideoCapture(camera2)
    img_counter = 0

    print('To take photos press SPACE as many times as needed, after images were taken, press ESC to continue')

    while True:
        ret1, frame1 = cam1.read()
        ret2, frame2 = cam2.read()
        cv2.imshow('Stereocamera 1', frame1)
        cv2.imshow('Stereocamera 2', frame2)
        if not ret1 and ret2:
            break
        k = cv2.waitKey(1)

        if k%256 == 27:
            # ESC pressed
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name1 = 'stereoCalCam1_{}.jpg'.format(img_counter)
            img_name2 = 'stereoCalCam2_{}.jpg'.format(img_counter)
            cv2.imwrite(img_name1, frame1)
            cv2.imwrite(img_name2, frame2)
            print('{} written!'.format(img_name1))
            print('{} written!'.format(img_name2))
            img_counter += 1

    cam1.release()
    cam2.release()
    cv2.destroyAllWindows()
    print('Stereocalibration frames captured!')
    #
    # Sterocalibration procedure
    objp = np.zeros((chessboardSize[1]*chessboardSize[0],3), np.float32)            # creating an empty array for object points
    objp[:,:2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1,2)*recatangleSize

    objpoints = []      # array to store object points (3D points in real world)
    imgpoints1 = []     # array to store image points of 1st camera  (2d points in image space)
    imgpoints2 = []     # array to store image points of 2nd camera


    ####################################################
    # collecting object and image points from camera 1 #
    ####################################################

    images1 = glob.glob('stereoCalCam1_*.jpg')         # loading all the jpg files collected previously

    for fname in images1:
        img1 = cv2.imread(fname)         # reading image files one by one
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)            # converting to grayscale

        ret, corners = cv2.findChessboardCorners(gray1, (chessboardSize[0],chessboardSize[1]), None)         # finding the corner points of the chessboard
        #if found, draw corners
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray1, corners, (11, 11), (-1, -1), criteria)
            imgpoints1.append(corners2)
            img1 = cv2.drawChessboardCorners(img1, (chessboardSize[0], chessboardSize[1]), corners2, ret)

    #########################################
    # collecting image points from camera 2 #
    #########################################

    images2 = glob.glob('stereoCalCam2_*.jpg')         # loading all the jpg files collected previously

    for fname in images2:
        img2 = cv2.imread(fname)         #loading all image files
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)            # converting to grayscale colorspace

        ret, corners3 = cv2.findChessboardCorners(gray2, (chessboardSize[0],chessboardSize[1]), None)         # finding the corner points of the chessboard
        #if found, draw corners
        if ret == True:

            corners4 = cv2.cornerSubPix(gray2, corners3, (11, 11), (-1, -1), criteria)
            imgpoints2.append(corners4)
            img2 = cv2.drawChessboardCorners(img2, (chessboardSize[0], chessboardSize[1]), corners4, ret)

    # the camera matrixes obtained previously from 1-2_cameraCalibration.py
    camMtx1 = np.load('camMtx1.npy')
    camMtx2 = np.load('camMtx2.npy')

    # the distortion coefficients
    distCoeffs1 = np.load('distCoeffs1.npy')
    distCoeffs2 = np.load('distCoeffs2.npy')

    #################################################################################
    # stereocalibration to get essential matrix, rotation, translation vector  ...  #
    #################################################################################

    retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints1, imgpoints2, camMtx1, distCoeffs1, camMtx2, distCoeffs2, gray1.shape[::-1], criteria, flags = cv2.CALIB_FIX_INTRINSIC)

    np.save('Ematrix', E)
    np.save('Fmatrix', F)
    np.save('Rot', R)
    np.save('Transl', T)

    # undistort the images
    undistorted1 = cv2.undistort(img1, camMtx1, distCoeffs1, None, None)
    undistorted2 = cv2.undistort(img2, camMtx2, distCoeffs2, None, None)

    # cv2.imshow('Left', undistorted1)
    # cv2.imshow('Right', undistorted2)

    cv2.waitKey(0) & 0xFF == ord('q')


    #################################################
    # stereo rectification to get rotation matrixes #
    #################################################

    # Camera Calibration to get Intrinsic/Extrinsic Parameters.
    #  Build a rotation matrix Rrect , which can rotate left camera to
    # make left epipole go to infinity (epipolar lines become horizontal).
    #  Apply the same rotation to the right camera
    #  Rotate the right camera by R (between cameras from
    # calibration)
    #  For each left-camera point, calculate a corresponding point
    # in the new stereo rig.
    #  Do the same thing for the right camera.
    #  Adjust the scale in both camera reference frames.

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(camMtx1, distCoeffs1, camMtx2, distCoeffs2, gray1.shape[::-1], R, T, alpha = 1)


    np.save('R1', R1)
    np.save('P1', P1)
    np.save('R2', R2)
    np.save('P2', P2)
    np.save('Q', Q)

    isStereoCalibrated = True
    print('The stereo calibration was successful: ', isStereoCalibrated)
