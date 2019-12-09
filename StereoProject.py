import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

###############
# INPUT START #
###############
# Are the 2 cameras calibrated separately?
isCalibrated = True
# Is the stereo system calibrated?
isStereoCalibrated = False
# giving calibration paper properties
chessboardSize = (6,5)
recatangleSize = 0.015
# camera codes
camera1 = 0
camera2 = 1
# Do you want to analyze images or video?
ImgOrVid = True         # True - image, False - video
# video reading - file or camera code for real time
video1 = 0
video2 = 0
# boundaries for Segmentation #bgr now
hsvMin = np.array([110, 110, 110])
hsvMax = np.array([200, 200, 200])
#############
# INPUT END #
#############

#further variables
isCam1Calibrated = False
isCam2Calibrated = False
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''
def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'w') as f:
        f.write(ply_header % dict(vert_num=len(verts)))
        np.savetxt(f, verts, '%f %f %f %d %d %d')


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
                #
                # if cv2.waitKey(0) & 0xFF == ord('q'):
                #     break

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

                # cv2.imshow('img', img)

                # if cv2.waitKey(0) & 0xFF == ord('q'):
                #     break

        ret2, mtx2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)          # calibrating the camre: getting the camera matrix, distortion coeff, rot, transl vector

        np.save('camMtx2', mtx2)         # save the camera matrix in .npy file
        np.save('distCoeffs2.npy', dist2)       # save the distortion coefficients
        np.save('camRvecs2', rvecs2)
        np.save('camTvecs2', tvecs2)
        # print('Camera matrix: ', mtx)
        # print('Distortion coefficients: ', dist)
        # Cam1 = np.load('camMtx1.npy')           # load the saved camera matrix
        # print(Cam1)
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

            # cv2.imshow('img', img1)
            #
            # if cv2.waitKey(0) & 0xFF == ord('q'):
            #     break


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

            # cv2.imshow('img', img2)
            #
            # if cv2.waitKey(0) & 0xFF == ord('q'):
            #     break

    # the camera matrixes obtained previously from 1-2_cameraCalibration.py
    camMtx1 = np.load('camMtx1.npy')
    camMtx2 = np.load('camMtx2.npy')

    # the distirtion coefficients
    distCoeffs1 = np.load('distCoeffs1.npy')
    distCoeffs2 = np.load('distCoeffs2.npy')


    #################################################################################
    # stereocalibration to get essential matrix, rotation, translation vector  ...  #
    #################################################################################

    retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints1, imgpoints2, camMtx1, distCoeffs1, camMtx2, distCoeffs2, gray1.shape[::-1], criteria, flags = cv2.CALIB_FIX_INTRINSIC)

    ## printing the results
    # print('Stereo camera mtx 1: ', cameraMatrix1)
    # print('Stereo camera mtx 2: ', cameraMatrix2)
    # print('Essential matrix: ', E)
    # print('Fundamental matrix: ', F)
    # print('Rotation parameters: ', R)
    # print('Translation parameters: ', T)

    np.save('Ematrix', E)
    np.save('Fmatrix', F)
    np.save('Rot', R)
    np.save('Transl', T)

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
    # print('T: ', T)
    # camMtx1= np.load('camMtx1.npy')
    # camMtx2= np.load('camMtx2.npy')
    # distCoeffs1= np.load('distCoeffs1.npy')
    # distCoeffs2= np.load('distCoeffs2.npy')
    # R= np.load('Rot.npy')
    # T = np.array([-0.33866645, 0.01231482, 0.08237773])
    # print(R)
    # print(T)
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(camMtx1, distCoeffs1, camMtx2, distCoeffs2, gray1.shape[::-1], R, T, alpha = 1)
    # alpha = 1

    np.save('R1', R1)
    np.save('P1', P1)
    np.save('R2', R2)
    np.save('P2', P2)
    np.save('Q', Q)

    isStereoCalibrated = True
    print('The stereo calibration was successful: ', isStereoCalibrated)

if isStereoCalibrated == True:
    print('Everything is calibrated, ready to take and/or read images/videos')
    # Loading the camera matrixes, distortion coefficients and data from stereocalibration: rotation data
    camMtx1 = np.load('camMtx1.npy')
    camMtx2 = np.load('camMtx2.npy')

    distCoeffs1 = np.load('distCoeffs1.npy')
    distCoeffs2 = np.load('distCoeffs2.npy')

    R1 = np.load('R1.npy')
    P1 = np.load('P1.npy')
    R2 = np.load('R2.npy')
    P2 = np.load('P2.npy')
    Q = np.load('Q.npy')


    ##################
    # Image analysis #
    ##################
    if ImgOrVid == True:
        # taking images
        #     Capturing the calibration frames with both cameras q
        cam1 = cv2.VideoCapture(camera1)
        cam2 = cv2.VideoCapture(camera2)
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

        # loading the images from the 2 cameras
        img1 = cv2.imread('Cam1_0.jpg')
        img2 = cv2.imread('Cam2_0.jpg')

        # converting images to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        # converting images to hsv
        hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
        # segmentation
        mask1 = cv2.inRange(img1, hsvMin, hsvMax)
        mask2 = cv2.inRange(img2, hsvMin, hsvMax)
        res1 = cv2.bitwise_and(img1, img1, mask = mask1)
        res2 = cv2.bitwise_and(img2, img2, mask = mask2)

        width, height, chanels = img1.shape


        #thresholding
        # ret,thresh1 = cv2.threshold(res1,180,255,cv2.THRESH_TOZERO)
        # ret2,thresh2 = cv2.threshold(res2,180,255,cv2.THRESH_TOZERO)
        # undistorting the images
        undistorted1 = cv2.undistort(img1, camMtx1, distCoeffs1, None, None)
        undistorted2 = cv2.undistort(img2, camMtx2, distCoeffs2, None, None)

        # plt.imshow(undistorted2)
        # plt.title('undistorted2')
        # plt.show()
        cv2.imshow('Left', undistorted1)
        cv2.imshow('Right', undistorted2)

        cv2.waitKey(0) & 0xFF == ord('q')

        # # compute the pixel mappings to the rectified versions of the images
        mapL1, mapL2 = cv2.initUndistortRectifyMap(camMtx1, distCoeffs1, R1, P1, gray1.shape[::-1], cv2.CV_32FC1);
        mapR1, mapR2 = cv2.initUndistortRectifyMap(camMtx2, distCoeffs2, R2, P2, gray2.shape[::-1], cv2.CV_32FC1);
        #
        # # undistort and rectify based on the mappings
        undistorted_rectifiedL = cv2.remap(res1, mapL1, mapL2, cv2.INTER_LINEAR)
        undistorted_rectifiedR = cv2.remap(res2, mapR1, mapR2, cv2.INTER_LINEAR)

        # # display images
        cv2.imshow('Undistorted Rectified cam 1',undistorted_rectifiedL)
        cv2.imshow('Undistorted Rectified cam 2',undistorted_rectifiedR)
        #
        # cv2.waitKey(0) & 0xFF == ord('q')

        ## disparity depth map
        window_size = 3
        num_disp = 32
        # stereo = cv2.StereoSGBM_create(numDisparities=num_disp, blockSize=7)
        stereo = cv2.StereoSGBM_create(numDisparities=num_disp, blockSize=15, P1 = 8*3*window_size**2, P2 = 32*3*window_size**2, disp12MaxDiff = 100, preFilterCap=50, uniquenessRatio = 2, speckleWindowSize = 50, speckleRange = 16)

        disparity = stereo.compute(undistorted_rectifiedL,undistorted_rectifiedR)
        # plt.imshow(disparity,'hot')
        # plt.colorbar()
        # plt.show()
        # disparity_colour_mapped = cv2.applyColorMap((disparity).astype(np.uint8), cv2.COLORMAP_BONE)
        # cv2.imshow('Disparity', disparity_colour_mapped)
        # cv2.imshow('disparity', (disparity-0)/num_disp)
        ######################################################
        # generating disparity map and its 3D reconstruction #
        ######################################################
        # min_disp = 16
        # num_disp = 112 - min_disp
        h, w = img1.shape[:2]
        f = camMtx1[1,1]                         # guess for focal length from camera matrix
        # Q = np.float32([[1, 0, 0, -0.5*w],
        #                 [0,-1, 0,  0.5*h],      # turn points 180 deg around x-axis,
        #                 [0, 0, 0,     -f],      # so that y-axis looks up
        #                 [0, 0, 1,      0]])
        # disparity = disparity.astype(np.float32) * 1.0/255
        # points = cv2.reprojectImageTo3D(disparity, Q)
        # cv2.imshow('depth', points)
        # print(points)
        right_matcher = cv2.ximgproc.createRightMatcher(stereo)
         # FILTER Parameters
        lmbda = 10000
        sigma = 2.0
        # visual_multiplier = 1

        wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
        wls_filter.setLambda(lmbda)
        wls_filter.setSigmaColor(sigma)

        displ = stereo.compute(undistorted_rectifiedL, undistorted_rectifiedR).astype(np.float32)/16
        dispr = right_matcher.compute(undistorted_rectifiedR, undistorted_rectifiedL).astype(np.float32)/16
        displ = np.int16(displ)
        dispr = np.int16(dispr)
        filteredImg = wls_filter.filter(dispr, undistorted_rectifiedL, None, displ)  # important to put "imgL" here!!!
        filteredImg2 = wls_filter.filter(displ, undistorted_rectifiedR, None, dispr)
        # need to normalize the image, so its values are between 0-255
        filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
        filteredImg2 = cv2.normalize(src=filteredImg2, dst=filteredImg2, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
        # need to convert back to 8b integer values
        filteredImg = np.uint8(filteredImg)
        filteredImg2 = np.uint8(filteredImg2)
        plt.imshow(filteredImg2, cmap='hot')
        plt.title('wls filtered image')
        plt.show()

        depth_map = cv2.normalize(src=filteredImg2, dst=filteredImg2, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
        depth_map = np.uint8(depth_map)
        depth_map = cv2.bitwise_not(depth_map)
        # np.savetxt("depth.csv", depth_map, delimiter=";")
        # cv2.imshow('depthm', depth_map)
        plt.imshow(depth_map, cmap = 'gray')
        plt.title('depthmap')
        plt.show()

        # depth_map2 = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
        # depth_map2 = np.uint8(depth_map2)
        # depth_map2 = cv2.bitwise_not(depth_map2)
        # # np.savetxt("depth.csv", depth_map, delimiter=";")
        # # cv2.imshow('depthm', depth_map)
        # plt.imshow(depth_map2, cmap = 'gray')
        # plt.title('depthmap2')
        # plt.show()

        width, height = depth_map.shape
        # thresholding the filtered image
        for y in range(width):
            for x in range(height):
            # threshold the pixel
                if depth_map[y, x] > 80:
                    depth_map[y, x] = 0
                else:
                    depth_map[y,x] = 255
                depth_map[y,x] = depth_map[y,x]

        plt.imshow(depth_map, cmap = 'gray')
        plt.title('threshold')
        plt.show()

        # print(depth_map)
        points = cv2.reprojectImageTo3D(depth_map, Q)
        # cv2.imshow('depth', points)
        # print(points)
        # np.savetxt("points.csv", points, delimiter=";")
        # # # print(f)
        colors = cv2.cvtColor(depth_map, cv2.COLOR_BGR2RGB)
        mask = dispr > dispr.min()
        out_points = points[mask]
        # cv2.imshow('out_points',out_points)
        out_colors = colors[mask]
        write_ply('out.ply', out_points, out_colors)
        # write_ply('points.ply', points, out_colors)
        # x = points[9:,0]
        # y = points[9:,1]
        # z = points[9:,2]
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(x, y, z, c='r',marker='o')
        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')
        # plt.show()

        #
        # X, Y, Z = [], [], []
        # for line in open('imgpcloud', 'r'):
        #   values = [float(s) for s in line.split()]
        #   X.append(values[0])
        #   Y.append(values[1])
        #   Z.append(values[2])
        # #
        #
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.plot(depth_map)
        # plt.show()

        # Create the x, y, and z coordinate arrays.  We use
        # numpy's broadcasting to do all the hard work for us.
        # # We could shorten this even more by using np.meshgrid.
        # x = np.arange(points.shape[0])[:, None, None]
        # y = np.arange(points.shape[1])[None, :, None]
        # z = np.arange(points.shape[2])[None, None, :]
        # x, y, z = np.broadcast_arrays(x, y, z)
        #
        # # Turn the volumetric data into an RGB array that's
        # # just grayscale.  There might be better ways to make
        # # ax.scatter happy.
        # c = np.tile(points.ravel()[:, None], [1, 3])
        #
        # # Do the plotting in a single call.
        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # ax.scatter(x.ravel(),
        #            y.ravel(),
        #            z.ravel())
        # plt.show()

        cv2.waitKey(0) & 0xFF == ord('q')


        #############################
        # Segmentation of depth map #???????????????????????????????????
        #############################

        # hsvDisparity = cv2.cvtColor(disparity_colour_mapped, cv2.COLOR_BGR2HSV)
        #
        # min = np.array([0,0,150])
        # max = np.array([0,100,255])
        #
        # mask = cv2.inRange(disparity_colour_mapped, min, max)
        # res = cv2.bitwise_not(disparity_colour_mapped, disparity_colour_mapped, mask = mask)
        # resgray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('segm', resgray)
        # print(disparity_colour_mapped)
        #
        # ret,thresh1 = cv2.threshold(resgray,200,255,cv2.THRESH_BINARY)
        # # cv2.imshow('thresh', thresh1)
        #
        # cv2.waitKey(0) & 0xFF == ord('q')


    ##################
    # Video analysis #
    ##################
    else:
        cap1 = cv2.VideoCapture(video1)
        cap2 = cv2.VideoCapture(video2)

        while(True):
            _, frame1 = cap1.read()
            _, frame2 = cap2.read()
            hsv1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
            hsv2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)

            # converting images to grayscale
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            # converting images to hsv
            hsv1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
            hsv2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
            # segmentation
            mask1 = cv2.inRange(hsv1, hsvMin, hsvMax)
            mask2 = cv2.inRange(hsv2, hsvMin, hsvMax)
            res1 = cv2.bitwise_not(frame1, frame1, mask = mask1)
            res2 = cv2.bitwise_not(frame2, frame2, mask = mask2)

            undistortedL = cv2.undistort(res1, camMtx1, distCoeffs1, None, None)
            undistortedR = cv2.undistort(res2, camMtx2, distCoeffs2, None, None)

            cv2.imshow('Left', undistortedL)
            cv2.imshow('Right', undistortedR)

            # compute the pixel mappings to the rectified versions of the images
            mapL1, mapL2 = cv2.initUndistortRectifyMap(camMtx1, distCoeffs1, R1, P1, gray1.shape[::-1], cv2.CV_32FC1);
            mapR1, mapR2 = cv2.initUndistortRectifyMap(camMtx2, distCoeffs2, R2, P2, gray2.shape[::-1], cv2.CV_32FC1);

            # undistort and rectify based on the mappings
            undistorted_rectifiedL = cv2.remap(res1, mapL1, mapL2, cv2.INTER_LINEAR)
            undistorted_rectifiedR = cv2.remap(res2, mapR1, mapR2, cv2.INTER_LINEAR)

            # display images
            # cv2.imshow('Undistorted Rectified cam 1',undistorted_rectifiedL)
            # cv2.imshow('Undistorted Rectified cam 2',undistorted_rectifiedR)

            # disparity depth map
            stereo = cv2.StereoSGBM_create(numDisparities=16, blockSize=15)
            disparity = stereo.compute(res1,res2)

            disparity_colour_mapped = cv2.applyColorMap((disparity).astype(np.uint8), cv2.COLORMAP_BONE)
            cv2.imshow('depth', disparity_colour_mapped)

            #############################
            # Segmentation of depth map #
            #############################
            #
            # hsvDisparity = cv2.cvtColor(disparity_colour_mapped, cv2.COLOR_BGR2HSV)
            #
            # min = np.array([0,0,150])
            # max = np.array([0,100,255])
            #
            # mask = cv2.inRange(disparity_colour_mapped, min, max)
            # res = cv2.bitwise_not(disparity_colour_mapped, disparity_colour_mapped, mask = mask)
            # resgray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
            # cv2.imshow('segm', resgray)
            # print(disparity_colour_mapped)
            #
            # ret,thresh1 = cv2.threshold(resgray,200,255,cv2.THRESH_BINARY)
            # # cv2.imshow('thresh', thresh1)

            if cv2.waitKey(1) & 0xFF == ord('q'):          #if q is pressed, close window
                break
#
cv2.destroyAllWindows()
