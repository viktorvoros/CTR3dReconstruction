import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# import glob
from math import sqrt
from skimage.measure import compare_ssim
import datetime
# from skimage.morphology import skeletonize
# from skimage import data
# from skimage.filters import threshold_otsu
# from skimage.util import invert
import scipy

###############################################################################################
################################### PRE-PROCESSING ############################################
###############################################################################################
# a = datetime.datetime.now()
# loading the calibration data
C1 = np.load('camMtx1.npy')
C2 = np.load('camMtx2.npy')
d1 = np.load('distCoeffs1.npy')
d2 = np.load('distCoeffs2.npy')

R1 = np.load('R1.npy')
P1 = np.load('P1.npy')
R2 = np.load('R2.npy')
P2 = np.load('P2.npy')
Q = np.load('Q.npy')
T = np.load('Transl.npy')


# loading images
img1 = cv2.imread('Cam1_9.jpg')
img2 = cv2.imread('Cam2_9.jpg')
# cv2.imshow('Left image', img1)
# cv2.imshow('Right image', img2)

# start counting time of execution
# a = datetime.datetime.now()

###############################################################################################
################################### PRE-PROCESSING ############################################
###############################################################################################

# converting images tog rayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# rectification of the stereo images
mapL1, mapL2 = cv2.initUndistortRectifyMap(C1, d1, R1, P1, gray1.shape[::-1], cv2.CV_32FC1)
mapR1, mapR2 = cv2.initUndistortRectifyMap(C2, d2, R2, P2, gray2.shape[::-1], cv2.CV_32FC1)

undistRect1 = cv2.remap(gray1, mapL1, mapL2, cv2.INTER_LINEAR, borderValue = 255)
undistRect2 = cv2.remap(gray2, mapR1, mapR2, cv2.INTER_LINEAR, borderValue = 255)

# Thresholding the rectified images
ret, undistRectThresh1 = cv2.threshold(undistRect1, 90, 255, cv2.THRESH_BINARY)
ret, undistRectThresh2 = cv2.threshold(undistRect2, 100, 255, cv2.THRESH_BINARY)

# crop thresholded and rectified images
undistRectThresh1 = undistRectThresh1[0:335, :]
undistRectThresh2 = undistRectThresh2[0:335, :]

# convert back to colored image
undistRect1 = cv2.cvtColor(undistRect1, cv2.COLOR_GRAY2BGR)
undistRect1 = undistRect1[0:335, :]

undistRect2 = cv2.cvtColor(undistRect2, cv2.COLOR_GRAY2BGR)
undistRect2 = undistRect2[0:335, :]

# cv2.imshow('Left thresholded rectified', undistRectThresh1)
# cv2.imshow('Right thresholded rectified', undistRectThresh2)
# cv2.waitKey(0)

###############################################################################################
################################### IMAGE PROCESSING ##########################################
###############################################################################################

# a = datetime.datetime.now()
# find the robot origin - xO - mark it with red color
def findOrigin(imgThresh, imgColor):
    ######################## METHOD ########################
    # check where the robot enters the image -- 4 if parts
    # find all black pixel coordinates on thresholded image
    # find last black pixel coordinate [y, x]
    # find all the black pixel coordinates on bottom row using last black pixel y coordinate
    # find first black pixel coordinate on bottom row [y, x]
    # origin point coordinate in [x, y]
    # mark origin on colored image with red circle
    #########################################################
    # note: make it just a number --- if: 0 - bottom, 1 - top, 2 - left, 3 - right
    # if the robot enters on bottom
    if np.count_nonzero(imgThresh[imgThresh.shape[0]-1,:]) != len(imgThresh[imgThresh.shape[0]-1,:]):

        black_pixels = np.array(np.where(imgThresh == 0))
        last_black_pixel = black_pixels[:,-1]

        bottom_black_pixels = np.array(np.where(black_pixels[0] == last_black_pixel[0]))
        first_black_pixel = [last_black_pixel[0], black_pixels[1, bottom_black_pixels[:,0]]]

        origin = (np.array([[int((last_black_pixel[1] + first_black_pixel[1])/2)], [last_black_pixel[0]]])).astype(int)

        imgColor = cv2.circle(imgColor, (origin[0], origin[1]), 3, (0, 0, 255), -1)

    # if the robot enters on top
    if np.count_nonzero(imgThresh[0,:]) != len(imgThresh[0,:]):
        black_pixels = np.array(np.where(imgThresh[0,:] == 0))
        last_black_pixel = black_pixels[:,0]

        bottom_black_pixels = np.array(np.where(black_pixels[0] == last_black_pixel[0]))
        first_black_pixel = [last_black_pixel[0], black_pixels[1, bottom_black_pixels[:,0]]]

        origin = (np.array([[int((last_black_pixel[1] + first_black_pixel[1])/2)], [last_black_pixel[0]]])).astype(int)

        imgColor = cv2.circle(imgColor, (origin[0], origin[1]), 3, (0, 0, 255), -1)

    # if the robot enters on the right
    if np.count_nonzero(imgThresh[:,imgThresh.shape[1]-1]) != len(imgThresh[:,imgThresh.shape[1]-1]):
        black_pixels = np.array(np.where(imgThresh[:,imgThresh.shape[1]-1] == 0))
        last_black_pixel = black_pixels[:,-1]

        bottom_black_pixels = np.array(np.where(black_pixels[0] == last_black_pixel[0]))
        first_black_pixel = [last_black_pixel[0], black_pixels[1, bottom_black_pixels[:,0]]]

        origin = (np.array([[int((last_black_pixel[1] + first_black_pixel[1])/2)], [last_black_pixel[0]]])).astype(int)

        imgColor = cv2.circle(imgColor, (origin[0], origin[1]), 3, (0, 0, 255), -1)

    # if the robot enters on the left
    if np.count_nonzero(undistRectThresh1[:,0]) != len(imgThresh[:,0]):
        black_pixels = np.array(np.where(imgThresh[:,0] == 0))
        last_black_pixel = black_pixels[:,-1]

        bottom_black_pixels = np.array(np.where(black_pixels[0] == last_black_pixel[0]))
        first_black_pixel = [last_black_pixel[0], black_pixels[1, bottom_black_pixels[:,0]]]

        origin = (np.array([[int((last_black_pixel[1] + first_black_pixel[1])/2)], [last_black_pixel[0]]])).astype(int)

        imgColor = cv2.circle(imgColor, (origin[0], origin[1]), 3, (0, 0, 255), -1)

    return origin

def findCenterPolar(origin, imgThresh, imgColor):
    black_pixels = np.array(np.where(imgThresh == 0))
    # find all black pixel coordinates and then transform origin to x0L,
    # !!! black_pixels is [y, x], while origin is [x, y] !!!
    new_originX = black_pixels[1] - origin[0]
    new_originY = black_pixels[0] - origin[1]

    # transform coordinates to polar coordinates [r, fi]
    polarR = (np.sqrt(new_originX**2 + new_originY**2)).astype(int)
    polarTh = np.arctan2(new_originY, new_originX)

    # find centerline points in polar coordinates
    centerTh = np.array([])
    centerR = np.array([])
    centerX = np.array([])
    centerY = np.array([])

    for r in range(0, polarR[1]):
        radius = np.array(np.where(polarR == r))
        th = polarTh[radius]
        meanTh = np.mean(th)

        centerTh = np.append(centerTh, meanTh)
        centerR = np.append(centerR, r)

    centerX = (centerR * np.cos(centerTh) + origin[0]).astype(int)
    centerY = (centerR * np.sin(centerTh) + origin[1]).astype(int)
    centerX = centerX[np.where(centerX > 0)]
    centerY = centerY[np.where(centerY > 0)]

    centerX = (centerX).astype(int)
    centerY = (centerY).astype(int)
    center = (np.array([centerX, centerY])).astype(int)
    center = np.transpose(center)
    # for i in range(0, len(centerX)):
    #     imgColor = cv2.circle(imgColor, (centerX[i], centerY[i]), 2, (0, 255, 0), -1)
    # imgColor[centerY, centerX] = (255,0,255)
    return center

def findWidthVect(pC1, imgThresh, imgColor):
    # previous and next point coordinate array
    pC0 = np.roll(pC1, 1, axis = 0)
    pC2 = np.roll(pC1, -1, axis = 0)

    # vectors from prev to next point
    v02 = (pC2 - pC0)[1:-1]


    # normalize vectors
    v02norm = v02 / np.sqrt(v02[:,0]**2 + v02[:,1]**2)[:, np.newaxis]

    # rotate by 90° -- -y -- switch columns and *-1 one of the columns
    v02norm[:,1] *= -1
    v02norm = np.roll(v02norm, 1, axis=1)

    # search range
    l = 5
    search = np.arange(-l, l)
    searchWidth = v02norm[...,np.newaxis]*search

    # search coordinates for each centerline point
    points = np.round(pC1[1:-1,:,np.newaxis] + searchWidth).astype(int)

    # img = np.zeros((img1.shape))
    # img[points[:,1], points[:,0]] = (255,0,0)
    # img[p0L[1], p0L[0]] = (0,0,255)
    # cv2.imshow('img', img)
    # undistRect2[points[l:-1,1], points[l:-1,0]] = (255,0,0)
    # cv2.imshow('w', undistRect1)
    # cv2.imwrite('wR.jpg', undistRect2)
    # find non zero values in each row -- nonzero - length of search range = width
    nonzero_w = np.count_nonzero(imgThresh[points[l:-1,1], points[l:-1,0]], axis = 1)
    # print(nonzero_w)
    width_curr = len(search) - nonzero_w
    width_next = np.roll(width_curr, -1)
    # in order to make it work properly the first element changed to second after rolling if widthPrev + widthCurr
    # width_prev[0] = width_prev[1]
    # sum of current and previous width for accurate joint search
    widthCurrNext = width_curr + width_next
    # print('v02',np.shape(widthCurrNext))

    return widthCurrNext

# find threshold values for the width dataset, will be used to determine the joint points
# adaptive thresholding based on the moving average of the width data
def findThresholdValues(widthData, N):
    # N = 10
    cumsum, moving_aves = [0], []

    for i, x in enumerate(widthData, 1):
        cumsum.append(cumsum[i-1] + x)
        if i>=N:
            moving_ave = (cumsum[i] - cumsum[i-N])/N
            #can do stuff with moving_ave here
            moving_aves.append(moving_ave)

    mn = (np.mean(widthData)).astype(int)

    th1 = ((np.max(moving_aves) + mn)/2 - 1).astype(int)
    th2 = ((mn + np.min(moving_aves))/2 - 1).astype(int)
    # print(th1)
    # print(th2)
    return th1, th2, moving_aves

# find the joint and tip points X2, X4 and X6 - mark them with blue, green and purple colors
# using threshold values of findThresholdValues()
def findJointPoints(movingAvg, centerPointX, centerPointY, th1, th2, imgColor):
    jointStage = 1
    j1 = np.array([])
    j2 = np.array([])
    jTip = np.array([])
    # another method - same speed
    # j1 = np.min(np.array(np.where(widthData < th1)))
    # j1x = centerPointX[j1]
    # j1y = centerPointY[j1]
    # # imgColor = cv2.circle(imgColor, (j1x,j1y), 3, (255,0,0),-1)
    # imgColor[j1y, j1x] = (255, 0, 0)
    #
    # j2 = np.min(np.array(np.where(widthData < th2)))
    # j2x = centerPointX[j2]
    # j2y = centerPointY[j2]
    # # imgColor = cv2.circle(imgColor, (j2x,j2y), 3, (0, 255, 0),-1)
    # imgColor[j2y, j2x] = (0, 255, 0)
    for i in range(0,len(movingAvg)):
        w = movingAvg[i]
        if w < th1 and jointStage == 1:
            imgColor = cv2.circle(imgColor, (centerPointX[i],centerPointY[i]), 3, (255,0,0),-1)
            # imgColor[centerPointY[i], centerPointX[i]] = (255, 0, 0)
            jointStage = 2
            j1 = np.array([[centerPointX[i]], [centerPointY[i]]])

        if w <= th2 and jointStage == 2:
            imgColor = cv2.circle(imgColor, (centerPointX[i],centerPointY[i]), 3, (0,255,0),-1)
            # imgColor[centerPointY[i], centerPointX[i]] = (0, 255, 0)
            jointStage = 3
            j2 = np.array([[centerPointX[i]], [centerPointY[i]]])

    imgColor = cv2.circle(imgColor, (centerPointX[-1], centerPointY[-1]), 3, (255, 255, 0),-1)
    jTip = (np.array([[centerPointX[-1]], [centerPointY[-1]]])).astype(int)
    # imgColor[centerPointY[-1], centerPointX[-1]] = (255, 255, 0)

    return j1, j2, jTip

# find mid point of each section - x1, x3 and x5 - mark with yellow dot on colored image
def findMidPoints(imgThresh, joint1, joint2, imgColor):
    midCoord = np.array([])
    if joint1.size != 0 and joint2.size != 0:
        # find vector from j1 to j2
        v12x = joint1[0] - joint2[0]
        v12y = joint1[1] - joint2[1]
        # normalize vector
        mag = sqrt(v12x**2 + v12y**2)
        v12x = v12x/mag
        v12y = v12y/mag
        # rotate 90°
        v12 = np.array([v12y, -v12x])
        # determine search range and find search points along vector from centerpoint with given search width -- search line
        l = 20
        search = np.arange(-l, l)
        # search points and centerpoint
        searchWidth = v12[...,np.newaxis]*search
        cpoint = ((joint1 + joint2)/2).astype(int)
        # search line
        sw = np.round(cpoint[...,np.newaxis] + searchWidth).astype(int)
        # find the robot pixels along the search line
        black_pix = np.array(np.where(imgThresh[sw[1,:,:], sw[0,:,:]] == 0))
        # find mid point based on mean values of the robot points along search line
        midCoord = np.array([[(np.mean(sw[0,:,black_pix[1]])).astype(int)], [(np.mean(sw[1,:, black_pix[1]])).astype(int)]])
        imgColor = cv2.circle(imgColor, (midCoord[0], midCoord[1]), 3, (0, 255, 255), -1)

    return midCoord

# 3D coordinates and 3D plot of the robot
def find3dCoords(xyL, xyR):
    # x coordinates in mm
    x = xyR[0] / 3.780
    # y coordinates in mm
    y = xyR[1] / 3.780

    # z coordinates based on disparity calculation - differences in the x coordinates when matching the 2 images
    diff = xyR[0] - xyL[0]
    # focus of camera
    f = C1[0,0]
    # baseline - distance between the 2 cameras
    b = sqrt(T[0]**2 + T[1]**2 + T[2]**2)
    k = f*b
    # z coordinates in mm
    z = k/diff * 1000
    # print(z)
    # z = z - z[0]
    # print(z)
    xyz = np.array([[x], [y], [z]])

    if len(z) == 7:
        c = ['r', 'y', 'b', 'y', 'g', 'y', 'c']
    elif len(z) == 13:
        c = ['r', 'y', 'y', 'y', 'b', 'y', 'y', 'y', 'g', 'y', 'y', 'y', 'c']
    elif len(z) == 4:
        c = ['r', 'b', 'g', 'c']
    # # plot 3d point cloud
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 2, 1, projection='3d')
    # ax.plot(x, y, z)
    #
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    #
    # ax = fig.add_subplot(1, 2, 2, projection='3d')
    # ax.scatter(x, y, z, c = c, marker='o')
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    # plt.show()

    # plot 3d point cloud in mm
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(x, y, z, c = c, marker = 'o')
    ax.set_xlabel('X label [mm]')
    ax.set_ylabel('Y label [mm]')
    ax.set_zlabel('Z label [mm]')
    ax.set_xlim(0, 640/3.780)
    ax.set_ylim(0,335/3.780)
    ax.set_zlim(0, np.min(z))
    # ax.set_title('3D pointcloud of CTR')
    plt.show()
    # in mm
    return xyz

###############################################################################################
################################### 3D RECONSTRUCTION  ########################################
###############################################################################################

if __name__ == '__main__':

    a = datetime.datetime.now()
    # finding robot coordinates
    # origin points
    p0L = findOrigin(undistRectThresh1, undistRect1)
    p0R = findOrigin(undistRectThresh2, undistRect2)

    # find all the centerline points with polar coordinates method
    pL = findCenterPolar(p0L, undistRectThresh1, undistRect1)
    pR = findCenterPolar(p0R, undistRectThresh2, undistRect2)

    # find the width values along the robot
    w1 = findWidthVect(pL, undistRectThresh1, undistRect1)
    w2 = findWidthVect(pR, undistRectThresh2, undistRect2)

    # median filter on dataset - optional
    # w1 = scipy.signal.medfilt(w1)
    # w2 = scipy.signal.medfilt(w2)

    # find the threshold values to find the width reduction --> find joint points
    th11, th12, movingAvg1 = findThresholdValues(w1, 6)
    th21, th22, movingAvg2 = findThresholdValues(w2, 5)

    # # a = datetime.datetime.now()
    # find the two joint and the tip points
    joint11, joint12, tip1 = findJointPoints(movingAvg1, pL[:,0], pL[:,1], th11, th12, undistRect1)
    joint21, joint22, tip2 = findJointPoints(movingAvg2, pR[:,0], pR[:,1], th21, th22, undistRect2)

    # find the midpoints of left tube
    p1L = findMidPoints(undistRectThresh1, p0L, joint11,  undistRect1)
    p3L = findMidPoints(undistRectThresh1, joint11, joint12, undistRect1)
    p5L = findMidPoints(undistRectThresh1, joint12, tip1, undistRect1)

    # # find the midpoints of right tube
    p1R = findMidPoints(undistRectThresh2, p0R, joint21, undistRect2)
    p3R = findMidPoints(undistRectThresh2, joint21, joint22, undistRect2)
    p5R = findMidPoints(undistRectThresh2, joint22, tip2, undistRect2)
    #
    # find additional points on left tube - in between current ones
    p2L = findMidPoints(undistRectThresh1, joint11, p1L, undistRect1)
    p4L = findMidPoints(undistRectThresh1, p0L, p1L, undistRect1)
    p6L = findMidPoints(undistRectThresh1, joint11, p3L, undistRect1)
    p8L = findMidPoints(undistRectThresh1, joint12, p3L, undistRect1)
    p9L = findMidPoints(undistRectThresh1, joint12, p5L, undistRect1)
    p10L = findMidPoints(undistRectThresh1, tip1, p5L, undistRect1)

    # # find additional points on right tube - in between current ones
    p2R = findMidPoints(undistRectThresh2, joint21, p1R, undistRect2)
    p4R = findMidPoints(undistRectThresh2, p0R, p1R, undistRect2)
    p6R = findMidPoints(undistRectThresh2, joint21, p3R, undistRect2)
    p8R = findMidPoints(undistRectThresh2, joint22, p3R,  undistRect2)
    p9R = findMidPoints(undistRectThresh2, joint22, p5R, undistRect2)
    p10R = findMidPoints(undistRectThresh2, tip2, p5R, undistRect2)

    # collect xy coordinates of left image in 1 array [2xn]
    xyCoordinatesL = p0L
    xyCoordinatesL = np.column_stack((xyCoordinatesL, p2L))
    xyCoordinatesL = np.column_stack((xyCoordinatesL, p1L))
    xyCoordinatesL = np.column_stack((xyCoordinatesL, p4L))
    xyCoordinatesL = np.column_stack((xyCoordinatesL, joint11))
    xyCoordinatesL = np.column_stack((xyCoordinatesL, p6L))
    xyCoordinatesL = np.column_stack((xyCoordinatesL, p3L))
    xyCoordinatesL = np.column_stack((xyCoordinatesL, p8L))
    xyCoordinatesL = np.column_stack((xyCoordinatesL, joint12))
    xyCoordinatesL = np.column_stack((xyCoordinatesL, p9L))
    xyCoordinatesL = np.column_stack((xyCoordinatesL, p5L))
    xyCoordinatesL = np.column_stack((xyCoordinatesL, p10L))
    xyCoordinatesL = np.column_stack((xyCoordinatesL, tip1))

    # collect xy coordinates of right image in 1 array [2xn]
    xyCoordinatesR = p0R
    xyCoordinatesR = np.column_stack((xyCoordinatesR, p2R))
    xyCoordinatesR = np.column_stack((xyCoordinatesR, p1R))
    xyCoordinatesR = np.column_stack((xyCoordinatesR, p4R))
    xyCoordinatesR = np.column_stack((xyCoordinatesR, joint21))
    xyCoordinatesR = np.column_stack((xyCoordinatesR, p6R))
    xyCoordinatesR = np.column_stack((xyCoordinatesR, p3R))
    xyCoordinatesR = np.column_stack((xyCoordinatesR, p8R))
    xyCoordinatesR = np.column_stack((xyCoordinatesR, joint22))
    xyCoordinatesR = np.column_stack((xyCoordinatesR, p9R))
    xyCoordinatesR = np.column_stack((xyCoordinatesR, p5R))
    xyCoordinatesR = np.column_stack((xyCoordinatesR, p10R))
    xyCoordinatesR = np.column_stack((xyCoordinatesR, tip2))

    # calculate 3d coordinates and plot 3D pointcloud - returns x, y, z coordinates in mm !!
    xyz = find3dCoords(xyCoordinatesL, xyCoordinatesR)

    # get time value -- how long it takes to complete script until this point
    print('time: {}'.format((datetime.datetime.now() - a).total_seconds()))

    # # generating time range for time series
    # xAxis1 = range(0,len(movingAvg1))
    # xAxis2 = range(0,len(movingAvg2))
    #
    # # plot the width values
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # fig.suptitle('moving averages on left and right width data')
    # ax1.plot(xAxis1, movingAvg1)
    # ax2.plot(xAxis2, movingAvg2)
    # plt.show()

    # plot marked images
    cv2.imshow('left', undistRect1)
    cv2.imshow('right', undistRect2)
    cv2.imwrite('13L.jpg', undistRect1)
    cv2.imwrite('13R.jpg', undistRect2)
    cv2.waitKey(0) & 0xFF == ord('q')
