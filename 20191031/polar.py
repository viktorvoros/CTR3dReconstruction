import numpy as np
import cv2
import datetime

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
img1 = cv2.imread('Cam1_5.jpg')
# img2 = cv2.imread('Cam2_8.jpg')

# start counting time of execution
# a = datetime.datetime.now()

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

    origin = (np.array([[int((last_black_pixel[1] + first_black_pixel[1])/2)], [last_black_pixel[0]]])).astype(int)

    imgColor = cv2.circle(imgColor, (origin[0], origin[1]), 3, (0, 0, 255), -1)

    return origin
def findCenterPolar(imgcolor, imgThresh, origin):
    black_pixels = np.array(np.where(imgThresh == 0))
    # find all black pixel coordinates and then transform origin to x0L, !!! black_pixels is [y, x], while origin is [x, y]
    new_originX = black_pixels[1] - origin[0]
    new_originY = black_pixels[0] - origin[1]

    # transform coordinates to polar coordinates [r, fi]
    polarR = np.round(np.sqrt(new_originX**2 + new_originY**2)).astype(int)
    polarFi = np.arctan2(new_originY, new_originX)

    # find centerline points in polar coordinates
    centerFi = np.array([])
    centerR = np.array([])
    centerX = np.array([])
    centerY = np.array([])

    for r in range(0, polarR[1]):
        radius = np.array(np.where(polarR == r))
        # print(radius)
        fi = polarFi[radius]
        meanFi = np.mean(fi)
        # print(mean)
        centerCartX = (r * np.cos(meanFi) + origin[0]).astype(int)
        centerCartY = (r * np.sin(meanFi) + origin[1]).astype(int)
        centerFi = np.append(centerFi, meanFi)
        centerR = np.append(centerR, r)
        centerX = np.append(centerX, centerCartX)
        centerY = np.append(centerY, centerCartY)

    centerX = centerX[np.where(centerX > 0)]
    centerY = centerY[np.where(centerY > 0)]
    # centerX = np.transpose(centerX)
    # centerY = np.transpose(centerY)

    # centerX = (centerX).astype(int)
    # centerY = (centerY).astype(int)
    center = (np.array([centerX, centerY])).astype(int)
    center = np.transpose(center)
    # for i in range(0, len(centerX)):
    #     imgcolor = cv2.circle(imgcolor, (centerX[i], centerY[i]), 2, (0, 255, 0), -1)

    return center
###############################################################################################
################################### PRE-PROCESSING ############################################
###############################################################################################

# converting images tog rayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# rectification of the stereo images
mapL1, mapL2 = cv2.initUndistortRectifyMap(C1, d1, R1, P1, gray1.shape[::-1], cv2.CV_32FC1)
# mapR1, mapR2 = cv2.initUndistortRectifyMap(C2, d2, R2, P2, gray2.shape[::-1], cv2.CV_32FC1)

undistRect1 = cv2.remap(gray1, mapL1, mapL2, cv2.INTER_LINEAR, borderValue = 255)
# undistRect2 = cv2.remap(gray2, mapR1, mapR2, cv2.INTER_LINEAR, borderValue = 255)

# Thresholding the rectified images
ret, undistRectThresh1 = cv2.threshold(undistRect1, 100, 255, cv2.THRESH_BINARY)
# ret, undistRectThresh2 = cv2.threshold(undistRect2, 110, 255, cv2.THRESH_BINARY)

# crop thresholded and rectified images
undistRectThresh1 = undistRectThresh1[0:335, :]
# undistRectThresh2 = undistRectThresh2[135:335, :]
undistRect1 = cv2.cvtColor(undistRect1, cv2.COLOR_GRAY2BGR)
undistRect1 = undistRect1[0:335, :]
#
a = datetime.datetime.now()
x0L = findOrigin(undistRectThresh1, undistRect1)
print(x0L.shape)
xL1 = findCenterPolar(undistRect1, undistRectThresh1, x0L)


# previous and next point coordinate array
xL0 = np.roll(xL1, 1, axis = 0)
xL2 = np.roll(xL1, -1, axis = 0)

# vectors from prev to next point
v02 = (xL2 - xL0)[1:-1]
print(xL1)
print(v02.shape)

# normalize vectors
v02norm = v02 / np.sqrt(v02[:,0]**2 + v02[:,1]**2)[:, np.newaxis]

# rotate by 90°
v02norm[:,1] *= -1
v02norm = np.roll(v02norm, 1, axis=1)
print(v02norm.shape)

# search range for width
l = 5

search = np.arange(-l, l)
print(search.shape)

searchWidth = v02norm[...,np.newaxis]*search
print(searchWidth.shape)

points = np.round(xL1[1:-1,:,np.newaxis] + searchWidth).astype(int)
# print(xL1)
# print('search: ', points)

img = np.zeros((img1.shape))

img[points[:,1], points[:,0]] = (255,0,0)
img[x0L[1], x0L[0]] = (0,0,255)
# print(undistRectThresh1)
undistRect1[points[l:-1,1], points[l:-1,0]] = (255,0,255)

nonzero_w = np.count_nonzero(undistRectThresh1[points[l:-1,1], points[l:-1,0]], axis = 1)

width_curr = len(search) - nonzero_w
width_prev = np.roll(width_curr, 1)
width_prev[0] = width_prev[1]
# print(width_curr)
# print(width_prev)

width = width_curr + width_prev

print(width)





# cv2.imshow('u', cv2.resize(img, None, fx=2, fy=2))
# cv2.waitKey(0)

cv2.imshow('u', undistRect1)
cv2.waitKey(0)
