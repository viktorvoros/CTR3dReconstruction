import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
from math import sqrt
from skimage.measure import compare_ssim
import datetime
from skimage.morphology import skeletonize
from skimage import data
from skimage.filters import threshold_otsu
from skimage.util import invert
import scipy
import pandas
# from joblib import Parallel, delayed
from skimage.draw import circle
import multiprocessing as mp
from multiprocessing import Pool as p
import psutil

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

# a = datetime.datetime.now()
# loading images
img1 = cv2.imread('Cam1_4.jpg')
img2 = cv2.imread('Cam2_4.jpg')

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
ret, undistRectThresh1 = cv2.threshold(undistRect1, 100, 255, cv2.THRESH_BINARY)
ret, undistRectThresh2 = cv2.threshold(undistRect2, 100, 255, cv2.THRESH_BINARY)
# # inverse treshold for skeleton images
# ret, undistRectThresh = cv2.threshold(undistRect2, 90, 255, cv2.THRESH_BINARY_INV)
# ret, undistRectThresh3 = cv2.threshold(undistRect2, 100, 255, cv2.THRESH_BINARY_INV)

# crop thresholded and rectified images
undistRectThresh1 = undistRectThresh1[0:335, :]
undistRectThresh2 = undistRectThresh2[0:335, :]

# undistRectThresh = undistRectThresh[135:335, :]
# undistRectThresh3 = undistRectThresh3[135:335, :]

undistRect1 = cv2.cvtColor(undistRect1, cv2.COLOR_GRAY2BGR)
undistRect1 = undistRect1[0:335, :]

undistRect2 = cv2.cvtColor(undistRect2, cv2.COLOR_GRAY2BGR)
undistRect2 = undistRect2[0:335, :]
# edges = cv2.Canny(undistRectThresh,0,50)
# cv2.imshow('canny edge', edges)

cv2.imshow('Left thresholded rectified', undistRectThresh1)
cv2.imshow('Right threshold rectified', undistRectThresh2)
cv2.waitKey(0)

# a = datetime.datetime.now()
# skeletonization
# skeleton = (skeletonize(undistRectThresh2//255) * 255).astype(np.uint8)
# skeleton2 = (skeletonize(undistRectThresh3//255) * 255).astype(np.uint8)
# skeleton = np.invert(skeleton)
# skeleton2 = np.invert(skeleton2)

# filled = scipy.ndimage.binary_fill_holes(skeleton).astype(np.uint8)
# cv2.imshow('filled', undistRectThresh)

# plt.imshow(skeleton, cmap=plt.cm.gray)
# plt.show()

# a = datetime.datetime.now()
###############################################################################################
################################### IMAGE PROCESSING ##########################################
###############################################################################################
# a = datetime.datetime.now()
# find the robot origin - xO - mark it with red color
def findOrigin(img, imgColor):
    ######################## METHOD ########################
    # find all black pixel coordinates on thresholded image
    # find last black pixel coordinate [y, x]
    # find all the black pixel coordinates on bottom row using last black pixel y coordinate
    # find first black pixel coordinate on bottom row [y, x]
    # origin point coordinate in [x, y] !!!!!
    # mark origin on colored image with red circle
    #########################################################

    black_pixels = np.array(np.where(img == 0))
    last_black_pixel = black_pixels[:,-1]

    bottom_black_pixels = np.array(np.where(black_pixels[0] == last_black_pixel[0]))
    first_black_pixel = [last_black_pixel[0], black_pixels[1, bottom_black_pixels[:,0]]]

    origin = (np.array([int((last_black_pixel[1] + first_black_pixel[1])/2), last_black_pixel[0]])).astype(int)

    imgColor = cv2.circle(imgColor, (origin[0], origin[1]), 3, (0, 0, 255), -1)

    return origin

# find centerline points with a given iteration increment
def findCenterPoints(imgColor, imgThresh, origin):
    centerPointX = []
    centerPointY = []
    rows, columns = imgThresh.shape

    incr = 1
    rmin = incr
    rmax = 225

    for r in range(rmin, rmax, incr):
        result = np.zeros((rows, columns)) + 255
        mask = cv2.circle(result, (origin[0], origin[1]), r, (0, 0, 0), 1)

        # sim = ~np.array(imgThresh / 255, 'bool') & np.array(img / 255, 'bool')
        sim = imgThresh + mask

        commonPoints = np.array(np.where(sim == 0))
        centerPointCoord = [(np.mean(commonPoints[0])).astype(int), np.mean(commonPoints[1]).astype(int)]
        # imgColor = cv2.circle(imgColor, (jointCoord[1], jointCoord[0]), 2, (0, 255, 0), -1)
        centerPointX = (np.append(centerPointX, [centerPointCoord[1]], axis = 0)).astype(int)
        centerPointY = (np.append(centerPointY, [centerPointCoord[0]], axis = 0)).astype(int)

    centerPointX = centerPointX[np.array(np.where(centerPointX > 0))]
    centerPointY = centerPointY[np.array(np.where(centerPointY > 0))]
    centerPointX = np.transpose(centerPointX)
    centerPointY = np.transpose(centerPointY)
    centerPoint = np.array([centerPointX, centerPointY]).astype(int)
    # centerPointX = np.transpose(centerPointX)
    # centerPointY = np.transpose(centerPointY)

    return centerPointX, centerPointY

# find the width of the robot at each centerline point found previously
def findWidth(imgThresh, xLx, xLy, imgColor):
    wPrev = 6
    jointStage = 1
    widthVal = []
    rows, columns = imgThresh.shape

    for i in range(1,len(xLx[:,0])-1):
            v13x = xLx[i+1] - xLx[i-1]
            v13y = xLy[i+1] - xLy[i-1]
            mag = sqrt(v13x*v13x + v13y*v13y)
            v13x = v13x/mag
            v13y = v13y/mag

            v13 = np.array([v13y, -v13x])

            cx = xLx[i] + v13[0]*5
            cy = xLy[i] + v13[1]*5
            dx = xLx[i] - v13[0]*5
            dy = xLy[i] - v13[1]*5

            img = np.zeros((rows, columns))
            img = cv2.line(img,((dx).astype(int),(dy).astype(int)), ((cx).astype(int),(cy).astype(int)), (255,255,255), 1)

            com = ~np.array(imgThresh / 255, 'bool') & np.array(img / 255, 'bool')

            black_pix = np.array(np.where(com))

            w = np.sqrt((np.max(black_pix[0]) - np.min(black_pix[0]))**2 + (np.max(black_pix[1]) - np.min(black_pix[1]))**2) + 1

            wPrev = w
            widthVal.append(w + wPrev)

    return widthVal

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
    th2 = ((mn + np.min(moving_aves))/2).astype(int)
    print(th1)
    print(th2)

    return th1, th2, moving_aves

# find the joint and tip points X2, X4 and X6 - mark them with blue, green and purple colors
# using threshold values of findThresholdValues()
def findJointPoints(widthData, centerPointX, centerPointY, th1, th2, imgColor):
    jointStage = 1
    # # another method - same speed
    # j1 = np.min(np.array(np.where(moving_aves < th1)))
    # j1x = centerPointX[j1]
    # j1y = centerPointY[j1]
    # imgColor = cv2.circle(imgColor, (j1x,j1y), 3, (255,0,0),-1)
    #
    # j2 = np.min(np.array(np.where(moving_aves < th2)))
    # j2x = centerPointX[j2]
    # j2y = centerPointY[j2]
    # imgColor = cv2.circle(imgColor, (j2x,j2y), 3, (0, 255, 0),-1)
    for i in range(0,len(widthData)):
        w = widthData[i]
        if w < th1 and jointStage == 1:
            imgColor = cv2.circle(imgColor, (centerPointX[i],centerPointY[i]), 3, (255,0,0),-1)
            jointStage = 2
            j1 = np.array([centerPointX[i], centerPointY[i]])

        if w <= th2 and jointStage == 2:
            imgColor = cv2.circle(imgColor, (centerPointX[i],centerPointY[i]), 3, (0,255,0),-1)
            jointStage = 3
            j2 = np.array([centerPointX[i], centerPointY[i]])
    imgColor = cv2.circle(imgColor, (centerPointX[-1],centerPointY[-1]), 3, (255, 255, 0),-1)
    jTip = (np.array([centerPointX[-1], centerPointY[-1]])).astype(int)

    return j1, j2, jTip

# find mid point of each section - x1, x3 and x5 - mark with yellow dot on colored image
def findMidPoints(imgThresh, joint1, joint2, imgColor):
    v12x = joint1[0] - joint2[0]
    v12y = joint1[1] - joint2[1]
    mag = sqrt(v12x*v12x + v12y*v12y)
    v12x = v12x/mag
    v12y = v12y/mag

    v12 = np.array([v12y, -v12x])

    cx = (joint1[0] + joint2[0])/2 + v12[0]*20
    cy = (joint1[1] + joint2[1])/2 + v12[1]*20
    dx = (joint1[0] + joint2[0])/2 - v12[0]*20
    dy = (joint1[1] + joint2[1])/2 - v12[1]*20

    rows, columns = imgThresh.shape
    img = np.zeros((rows, columns))
    img = cv2.line(img,(dx.astype(int),dy.astype(int)), (cx.astype(int),cy.astype(int)), (255,255,255), 1)

    com = ~np.array(imgThresh / 255, 'bool') & np.array(img / 255, 'bool')

    black_pix = np.array(np.where(com))

    midCoord = np.array([(np.mean(black_pix[1])).astype(int), np.mean(black_pix[0]).astype(int)])
    imgColor = cv2.circle(imgColor, (midCoord[0], midCoord[1]), 3, (0, 255, 255), -1)

    return midCoord

# 3D plot of the robot based on 7 points - origin, 2 joint, 3 mid and tip
def get3Dp7(x0L, x1L, joint11, x3L, joint12, x5L, tip1, x0R, x1R, joint21, x3R, joint22, x5R, tip2):
    # data processing
    joint11 = np.transpose(joint11)
    joint12 = np.transpose(joint12)
    tip1 = np.transpose(tip1)

    joint21 = np.transpose(joint21)
    joint22 = np.transpose(joint22)
    tip2 = np.transpose(tip2)

    # collecting the x coordinates of each point into an array
    left7 = np.transpose([x0L[0], x1L[0], joint11[0,0], x3L[0], joint12[0,0], x5L[0], tip1[0,0]])
    right7 = np.transpose([x0R[0], x1R[0], joint21[0,0], x3R[0], joint22[0,0], x5R[0], tip2[0,0]])

    # x coordinates of left image points
    xL0 = x0L[0]
    xL1 = x1L[0]
    xL2 = joint11[0,0]
    xL3 = x3L[0]
    xL4 = joint12[0,0]
    xL5 = x5L[0]
    xL6 = tip1[0,0]

    xL = [xL0, xL1, xL2, xL3, xL4, xL5, xL6]

    # right image x-y coordinates
    xR0 = x0R[0]
    xR1 = x1R[0]
    xR2 = joint21[0,0]
    xR3 = x3R[0]
    xR4 = joint22[0,0]
    xR5 = x5R[0]
    xR6 = tip2[0,0]

    x = [xR0, xR1, xR2, xR3, xR4, xR5, xR6]

    yR0 = x0R[1]
    yR1 = x1R[1]
    yR2 = joint21[0,-1]
    yR3 = x3R[1]
    yR4 = joint22[0,-1]
    yR5 = x5R[1]
    yR6 = tip2[0,-1]

    y = [yR0, yR1, yR2, yR3, yR4, yR5, yR6]

    # z coordinates based on disparity calculation - differences in the x coordinates when matching the 2 images
    diff =  right7 - left7

    # focus of camera
    f = C1[0,0]
    # baseline - distance between the 2 cameras
    b = sqrt(T[0]**2 + T[1]**2 + T[2]**2)
    k = f*b

    z0 = k/diff[0]
    z1 = k/diff[1]
    z2 = k/diff[2]
    z3 = k/diff[3]
    z4 = k/diff[4]
    z5 = k/diff[5]
    z6 = k/diff[6]

    z = [z0, z1, z2, z3, z4, z5, z6]

    # plot 3d point cloud
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot(x, y, z)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.scatter(x, y, z, c=['r', 'y', 'b', 'y', 'g', 'y', 'c'], marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

    return x, y, z

# 3D plot of the robot based on 13 points - origin, 2 joint, 3 mid, 6 extra and tip
def get3Dp13(x0L, x2L, x1L, x4L, joint11, x6L, x3L, x8L, joint12, x9L, x5L, x10L, tip1, x0R, x2R, x1R, x4R, joint21, x6R, x3R, x8R, joint22, x9R, x5R, x10R, tip2):
    # data processing
    joint11 = np.transpose(joint11)
    joint12 = np.transpose(joint12)
    tip1 = np.transpose(tip1)

    joint21 = np.transpose(joint21)
    joint22 = np.transpose(joint22)
    tip2 = np.transpose(tip2)
    # 3D reconstruction with 13 points
    left13 = np.transpose([x0L[0], x2L[0], x1L[0], x4L[0],  joint11[0,0], x6L[0], x3L[0], x8L[0], joint12[0,0], x9L[0], x5L[0], x10L[0], tip1[0,0]])
    right13 = np.transpose([x0R[0], x2R[0], x1R[0], x4R[0],  joint21[0,0], x6R[0], x3R[0], x8R[0], joint22[0,0], x9R[0], x5R[0], x10R[0], tip2[0,0]])

    xR0 = x0R[0]
    xR1 = x2R[0]
    xR2 = x1R[0]
    xR3 = x4R[0]
    xR4 = joint21[0,0]
    xR5 = x6R[0]
    xR6 = x3R[0]
    xR7 = x8R[0]
    xR8 = joint22[0,0]
    xR9 = x9R[0]
    xR10 = x5R[0]
    xR11 = x10R[0]
    xR12 = tip2[0,0]

    x = [xR0, xR1, xR2, xR3, xR4, xR5, xR6, xR7, xR8, xR9, xR10, xR11, xR12]

    yR0 = x0R[1]
    yR1 = x2R[1]
    yR2 = x1R[1]
    yR3 = x4R[1]
    yR4 = joint21[0,-1]
    yR5 = x6R[1]
    yR6 = x3R[1]
    yR7 = x8R[1]
    yR8 = joint22[0,-1]
    yR9 = x9R[1]
    yR10 = x5R[1]
    yR11 = x10R[1]
    yR12 = tip2[0,-1]

    y = [yR0, yR1, yR2, yR3, yR4, yR5, yR6, yR7, yR8, yR9, yR10, yR11, yR12]

    diff =  right13 - left13

    # focus of camera
    f = C1[0,0]
    # baseline - distance between the 2 cameras
    b = sqrt(T[0]**2 + T[1]**2 + T[2]**2)
    k = f*b

    z0 = k/diff[0]
    z1 = k/diff[1]
    z2 = k/diff[2]
    z3 = k/diff[3]
    z4 = k/diff[4]
    z5 = k/diff[5]
    z6 = k/diff[6]
    z7 = k/diff[7]
    z8 = k/diff[8]
    z9 = k/diff[9]
    z10 = k/diff[10]
    z11 = k/diff[11]
    z12 = k/diff[12]

    z = [z0, z1, z2, z3, z4, z5, z6, z7, z8, z9, z10, z11, z12]

    # plot 3d point cloud
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot(x, y, z)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.scatter(x, y, z, c=['r', 'y', 'y', 'y', 'b', 'y', 'y', 'y', 'g', 'y', 'y', 'y', 'c'], marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

    return x, y, z

# # skeleton coordinates
# skelCoord = np.array(np.where(skeleton == 255))
# skelCoord = np.transpose(skelCoord)
# # skelCoord2 = np.array(np.where(skeleton2 == 255))

###############################################################################################
################################### 3D RECONSTRUCTION  ########################################
###############################################################################################

a = datetime.datetime.now()
if __name__ == '__main__':
    # mp.set_start_method('spawn')

    # finding robot coordinates
    # origin points
    x0L = findOrigin(undistRectThresh1, undistRect1)
    x0R = findOrigin(undistRectThresh2, undistRect2)


    # with Pool(5) as p:

    # with mp.Pool(processes=5) as p:
    #     xLy = p.map(findCenterPoints, zip(undistRect1, undistRectThresh1, x0L))
    # with mp.Pool(processes=5) as p:
    #     xLy = p.map(multi_run_wrapper, [(undistRect1, undistRectThresh1, x0L)])
    # num_cpus = psutil.cpu_count(logical=False)
    # print(num_cpus)
    # pool = p(num_cpus)
    # for _ in range(10):
    #     results = pool.map(findCenterPoints, zip(undistRect1, undistRectThresh1, x0L))
    #     xLx = [result[0] for result in results]
    #     xLy = [result[1] for result in results]

    # find all the centerline points
    xLx, xLy = findCenterPoints(undistRect1, undistRectThresh1, x0L)
    xRx, xRy = findCenterPoints(undistRect2, undistRectThresh2, x0R)

    # find the width values along the robot and applying median filter on dataset
    w1 = findWidth(undistRectThresh1, xLx, xLy, undistRect1)
    w2 = findWidth(undistRectThresh2, xRx, xRy, undistRect2)
    w1 = scipy.signal.medfilt(w1)
    w2 = scipy.signal.medfilt(w2)

    # find the threshold values to find the width reduction --> find joint points
    th11, th12, movingAvg1 = findThresholdValues(w1, 4)
    th21, th22, movingAvg2 = findThresholdValues(w2, 4)

    # a = datetime.datetime.now()
    # find the two joint and the tip points
    joint11, joint12, tip1 = findJointPoints(movingAvg1, xLx, xLy, th11, th12, undistRect1)
    joint21, joint22, tip2 = findJointPoints(movingAvg2, xRx, xRy, th21, th22, undistRect2)

    # get time value -- how long it takes to complete script until this point
    # print('time: {}'.format((datetime.datetime.now() - a).total_seconds()))

    # find the midpoints of left tube
    x1L = findMidPoints(undistRectThresh1, x0L, joint11,  undistRect1)
    x3L = findMidPoints(undistRectThresh1, joint11, joint12, undistRect1)
    x5L = findMidPoints(undistRectThresh1, joint12, tip1, undistRect1)

    # find the midpoints of right tube
    x1R = findMidPoints(undistRectThresh2, x0R, joint21, undistRect2)
    x3R = findMidPoints(undistRectThresh2, joint21, joint22, undistRect2)
    x5R = findMidPoints(undistRectThresh2, joint22, tip2, undistRect2)

    # print('time: {}'.format((datetime.datetime.now() - a).total_seconds()))
    # calculate 3d coordinates and plot 3D pointcloud - returns x, y, z coordinates - based on 7 points
    x7, y7, z7 = get3Dp7(x0L, x1L, joint11, x3L, joint12, x5L, tip1, x0R, x1R, joint21, x3R, joint22, x5R, tip2)
    #
    # # find additional points on left tube
    # x2L = findMidPoints(undistRectThresh1, joint11, x1L,  undistRect1)
    # x4L = findMidPoints(undistRectThresh1, x0L, x1L,  undistRect1)
    # x6L = findMidPoints(undistRectThresh1, joint11, x3L,  undistRect1)
    # x8L = findMidPoints(undistRectThresh1, joint12, x3L,  undistRect1)
    # x9L = findMidPoints(undistRectThresh1, joint12, x5L, undistRect1)
    # x10L = findMidPoints(undistRectThresh1, tip1, x5L, undistRect1)
    # # # find additional points on right tube
    # x2R = findMidPoints(undistRectThresh2, joint21, x1R,  undistRect2)
    # x4R = findMidPoints(undistRectThresh2, x0R, x1R,  undistRect2)
    # x6R = findMidPoints(undistRectThresh2, joint21, x3R,  undistRect2)
    # x8R = findMidPoints(undistRectThresh2, joint22, x3R,  undistRect2)
    # x9R = findMidPoints(undistRectThresh2, joint22, x5R, undistRect2)
    # x10R = findMidPoints(undistRectThresh2, tip2, x5R, undistRect2)
    # print('time: {}'.format((datetime.datetime.now() - a).total_seconds()))
    # # calculate 3d coordinates and plot 3D pointcloud - returns x, y, z coordinates - based on 13 points
    # x13, y13, z13 = get3Dp13(x0L, x2L, x1L, x4L, joint11, x6L, x3L, x8L, joint12, x9L, x5L, x10L, tip1, x0R, x2R, x1R, x4R, joint21, x6R, x3R, x8R, joint22, x9R, x5R, x10R, tip2)

    # print('time: {}'.format((datetime.datetime.now() - a).total_seconds()))
    # # get time value -- how long it takes to complete script until this point
    # print('time: {}'.format((datetime.datetime.now() - a).total_seconds()))

    # generating time range for time series
    xAxis1 = range(0,len(movingAvg1))
    xAxis2 = range(0,len(movingAvg2))

    # plot the width values
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('moving averages on left and right width data')
    ax1.plot(xAxis1, movingAvg1)
    ax2.plot(xAxis2, movingAvg2)
    plt.show()

    # plot marked images
    cv2.imshow('left', undistRect1)
    cv2.imshow('right', undistRect2)
    cv2.waitKey(0) & 0xFF == ord('q')
