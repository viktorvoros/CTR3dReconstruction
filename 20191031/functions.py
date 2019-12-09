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

def findCenterPolar(imgcolor, imgThresh, origin):
    black_pixels = np.array(np.where(imgThresh == 0))
    # find all black pixel coordinates and then transform origin to x0L, !!! black_pixels is [y, x], while origin is [x, y]
    new_originX = black_pixels[1] - origin[0]
    new_originY = black_pixels[0] - origin[1]

    # transform coordinates to polar coordinates [r, fi]
    polarR = (np.sqrt(new_originX**2 + new_originY**2)).astype(int)
    polarFi = np.arctan2(new_originY, new_originX)

    # find centerline points in polar coordinates
    centerFi = np.array([])
    centerR = np.array([])
    centerX = np.array([])
    centerY = np.array([])

    for r in range(0, polarR[1]):
        radius = np.array(np.where(polarR == r))
        fi = polarFi[radius]
        meanFi = np.mean(fi)

        centerFi = np.append(centerFi, meanFi)
        centerR = np.append(centerR, r)

    centerX = (centerR * np.cos(centerFi) + origin[0]).astype(int)
    centerY = (centerR * np.sin(centerFi) + origin[1]).astype(int)
    centerX = centerX[np.where(centerX > 0)]
    centerY = centerY[np.where(centerY > 0)]

    centerX = (centerX).astype(int)
    centerY = (centerY).astype(int)
    center = (np.array([centerX, centerY])).astype(int)
    center = np.transpose(center)
    # for i in range(0, len(centerX)):
    #     imgcolor = cv2.circle(imgcolor, (centerX[i], centerY[i]), 2, (0, 255, 0), -1)
    # imgcolor[centerY, centerX] = (255,0,255)
    return center

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

def findWidthVect(imgThresh, xL1, imgColor):
    # previous and next point coordinate array
    xL0 = np.roll(xL1, 1, axis = 0)
    xL2 = np.roll(xL1, -1, axis = 0)

    # vectors from prev to next point
    v02 = (xL2 - xL0)[1:-1]

    # normalize vectors
    v02norm = v02 / np.sqrt(v02[:,0]**2 + v02[:,1]**2)[:, np.newaxis]

    # rotate by 90° -- -y -- switch columns
    v02norm[:,1] *= -1
    v02norm = np.roll(v02norm, 1, axis=1)

    # search range
    l = 5
    search = np.arange(-l, l)
    searchWidth = v02norm[...,np.newaxis]*search
    # search coordinates for each centerline point
    points = np.round(xL1[1:-1,:,np.newaxis] + searchWidth).astype(int)

    # img = np.zeros((img1.shape))
    # img[points[:,1], points[:,0]] = (255,0,0)
    # img[x0L[1], x0L[0]] = (0,0,255)
    # undistRect1[points[l:-1,1], points[l:-1,0]] = (255,0,255)

    # find non zero values in each row -- nonzero - length of search range = width
    nonzero_w = np.count_nonzero(imgThresh[points[l:-1,1], points[l:-1,0]], axis = 1)

    width_curr = len(search) - nonzero_w
    width_next = np.roll(width_curr, -1)
    # in order to make it work properly the first element changed to second after rolling
    # width_prev[0] = width_prev[1]
    # sum of current and previous width for accurate joint search
    widthCurrNext = width_curr + width_next

    return widthCurrNext

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
    for i in range(0,len(widthData)):
        w = widthData[i]
        if w < th1 and jointStage == 1:
            imgColor = cv2.circle(imgColor, (centerPointX[i],centerPointY[i]), 2, (255,0,0),-1)
            # imgColor[centerPointY[i], centerPointX[i]] = (255, 0, 0)
            jointStage = 2
            j1 = np.array([[centerPointX[i]], [centerPointY[i]]])

        if w <= th2 and jointStage == 2:
            imgColor = cv2.circle(imgColor, (centerPointX[i],centerPointY[i]), 2, (0,255,0),-1)
            # imgColor[centerPointY[i], centerPointX[i]] = (0, 255, 0)
            jointStage = 3
            j2 = np.array([[centerPointX[i]], [centerPointY[i]]])
    imgColor = cv2.circle(imgColor, (centerPointX[-1], centerPointY[-1]), 2, (255, 255, 0),-1)
    jTip = (np.array([[centerPointX[-1]], [centerPointY[-1]]])).astype(int)
    # imgColor[centerPointY[-1], centerPointX[-1]] = (255, 255, 0)

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


# find mid point of each section - x1, x3 and x5 - mark with yellow dot on colored image
def findMidPointsVect(imgThresh, joint1, joint2, imgColor):
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
    imgColor = cv2.circle(imgColor, (midCoord[0], midCoord[1]), 2, (0, 255, 255), -1)

    return midCoord

# 3D plot of the robot based on 7 points - origin, 2 joint, 3 mid and tip
def get3Dp7(x0L, x1L, joint11, x3L, joint12, x5L, tip1, x0R, x1R, joint21, x3R, joint22, x5R, tip2):

    # collecting the x coordinates of each point into an array
    left7 = np.transpose([x0L[0,0], x1L[0,0], joint11[0,0], x3L[0,0], joint12[0,0], x5L[0,0], tip1[0,0]])
    right7 = np.transpose([x0R[0,0], x1R[0,0], joint21[0,0], x3R[0,0], joint22[0,0], x5R[0,0], tip2[0,0]])

    x = right7

    yR0 = x0R[1,0]
    yR1 = x1R[1,0]
    yR2 = joint21[1,0]
    yR3 = x3R[1,0]
    yR4 = joint22[1,0]
    yR5 = x5R[1,0]
    yR6 = tip2[1,0]

    y = [yR0, yR1, yR2, yR3, yR4, yR5, yR6]

    # z coordinates based on disparity calculation - differences in the x coordinates when matching the 2 images
    diff =  right7 - left7

    # focus of camera
    f = C1[0,0]
    # baseline - distance between the 2 cameras
    b = sqrt(T[0]**2 + T[1]**2 + T[2]**2)
    k = f*b

    z = k/diff

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
    # ax.scatter(x, y, z, c=['r', 'y', 'b', 'y', 'g', 'y', 'c'], marker='o')
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    # plt.show()
    #
    return x, y, z

# 3D plot of the robot based on 13 points - origin, 2 joint, 3 mid, 6 extra and tip
def get3Dp13(x0L, x2L, x1L, x4L, joint11, x6L, x3L, x8L, joint12, x9L, x5L, x10L, tip1, x0R, x2R, x1R, x4R, joint21, x6R, x3R, x8R, joint22, x9R, x5R, x10R, tip2):
    # 3D reconstruction with 13 points
    left13 = np.transpose([x0L[0,0], x2L[0,0], x1L[0,0], x4L[0,0],  joint11[0,0], x6L[0,0], x3L[0,0], x8L[0,0], joint12[0,0], x9L[0,0], x5L[0,0], x10L[0,0], tip1[0,0]])
    right13 = np.transpose([x0R[0,0], x2R[0,0], x1R[0,0], x4R[0,0],  joint21[0,0], x6R[0,0], x3R[0,0], x8R[0,0], joint22[0,0], x9R[0,0], x5R[0,0], x10R[0,0], tip2[0,0]])

    x = right13

    yR0 = x0R[1,0]
    yR1 = x2R[1,0]
    yR2 = x1R[1,0]
    yR3 = x4R[1,0]
    yR4 = joint21[1,0]
    yR5 = x6R[1,0]
    yR6 = x3R[1,0]
    yR7 = x8R[1,0]
    yR8 = joint22[1,0]
    yR9 = x9R[1,0]
    yR10 = x5R[1,0]
    yR11 = x10R[1,0]
    yR12 = tip2[1,0]

    y = [yR0, yR1, yR2, yR3, yR4, yR5, yR6, yR7, yR8, yR9, yR10, yR11, yR12]

    diff =  right13 - left13

    # focus of camera
    f = C1[0,0]
    # baseline - distance between the 2 cameras
    b = sqrt(T[0]**2 + T[1]**2 + T[2]**2)
    k = f*b

    z = k/diff
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
    # ax.scatter(x, y, z, c=['r', 'y', 'y', 'y', 'b', 'y', 'y', 'y', 'g', 'y', 'y', 'y', 'c'], marker='o')
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    # plt.show()

    return x, y, z

# 3D coordinates and 3D plot of the robot
def find3dCoords(xyL, xyR):
    # x coordinates
    x = xyR[0]
    # y coordinates
    y = xyR[1]

    # z coordinates based on disparity calculation - differences in the x coordinates when matching the 2 images
    diff =  xyR[0] - xyL[0]

    # focus of camera
    f = C1[0,0]
    # baseline - distance between the 2 cameras
    b = sqrt(T[0]**2 + T[1]**2 + T[2]**2)
    k = f*b
    # z coordinates
    z = k/diff

    xyz = np.array([[x], [y], [z]])

    if len(z) == 7:
        c = ['r', 'y', 'b', 'y', 'g', 'y', 'c']
    else:
        c = ['r', 'y', 'y', 'y', 'b', 'y', 'y', 'y', 'g', 'y', 'y', 'y', 'c']
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

    return xyz
